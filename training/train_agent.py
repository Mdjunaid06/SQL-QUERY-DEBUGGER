"""
training/train_agent.py — SQL Database Engineer Agent
Unsloth + GRPO training script.
Run on venue GPU (April 25-26) with compute credits.

FIXES applied:
  1. Robust JSON extraction via regex (kills PARSE FALLBACK)
  2. task_id from kwargs directly — not from kwargs["batch"] (kills only-easy_s001)
  3. Reward calls /grader (stateless) instead of /reset+/step (kills race condition + flat 0.500)
  4. Format bonus so valid JSON gets non-zero reward even before agent learns DBA actions
"""

import os
import re
import json
import warnings
import requests
from datasets import Dataset

# ── Try importing Unsloth (GPU only) ─────────────────────────
try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available. Run: pip install unsloth trl")

# Keep demo/training logs readable by silencing known upstream deprecation noise.
warnings.filterwarnings(
    "ignore",
    message=r".*Both `max_new_tokens`.*and `max_length`.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*AttentionMaskConverter.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*use_return_dict.*deprecated.*",
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

ENV_URL    = os.getenv("ENV_URL",    "https://junaid0600-sql-db-engineer-agent.hf.space")
HF_TOKEN   = os.getenv("HF_TOKEN",  "")
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./sdea-trained")

# Valid Round 2 action types — model must use one of these
VALID_ACTION_TYPES = {
    "inspect_query", "analyze_indexes", "create_index",
    "rewrite_query", "add_column", "drop_index",
    "partition_table", "analyze_statistics",
    "request_hint", "submit_report",
}

SYSTEM_PROMPT = """You are a senior database engineer.
Given a database scenario with slow queries, choose the BEST single action to improve performance.

Investigation pattern (follow this order):
1. Use inspect_query to understand WHY a query is slow (scan type, rows examined)
2. Use analyze_indexes to see what indexes exist and what is missing
3. Use create_index to add the missing index on WHERE/JOIN columns
4. Use rewrite_query if the SQL itself is inefficient
5. Use partition_table for tables with 1M+ rows and range queries
6. Use submit_report when performance target is reached

RESPOND WITH VALID JSON ONLY. No explanation. No markdown. No preamble.
Examples:
{"action_type": "inspect_query", "payload": {"query_id": "q1"}}
{"action_type": "analyze_indexes", "payload": {"table": "users"}}
{"action_type": "create_index", "payload": {"table": "users", "columns": ["email"]}}
{"action_type": "create_index", "payload": {"table": "orders", "columns": ["user_id", "status"]}}
{"action_type": "submit_report", "payload": {"summary": "Added composite index on orders(user_id, status). Performance improved from 5.0 to 85.0."}}"""


# ─────────────────────────────────────────────
#  JSON EXTRACTION  (FIX 1 — kills PARSE FALLBACK)
# ─────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """
    Robustly extract a JSON object from model output.
    Handles: pure JSON, markdown blocks, JSON buried in text, partial JSON.
    Returns parsed dict or None if nothing parseable found.
    """
    if not text:
        return None

    # Strip common markdown wrappers
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Try 1: entire text is valid JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try 2: balanced-brace scan from every opening '{'
    # This handles nested payload objects/arrays more reliably than regex.
    starts = [idx for idx, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "action_type" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def _is_valid_action(action: dict) -> bool:
    """Check action has correct structure before sending to /grader."""
    if not isinstance(action, dict):
        return False
    if "action_type" not in action:
        return False
    if action["action_type"] not in VALID_ACTION_TYPES:
        return False
    if "payload" not in action or not isinstance(action.get("payload"), dict):
        return False
    return True


def _completion_to_text(completion) -> str:
    """
    Normalize TRL completion formats into raw text.
    Handles dict/list/string and content blocks emitted by some chat templates.
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for block in content:
                if isinstance(block, dict):
                    chunks.append(str(block.get("text", "")))
                else:
                    chunks.append(str(block))
            return "".join(chunks)
        return str(content)

    if isinstance(completion, list):
        # Common shape: [{"role":"assistant","content":"..."}]
        if not completion:
            return ""
        return "".join(_completion_to_text(item) for item in completion)

    return str(completion)


# ─────────────────────────────────────────────
#  REWARD FUNCTION  (FIX 2 + FIX 3)
# ─────────────────────────────────────────────

def reward_fn(prompts, completions, **kwargs):
    """
    GRPO reward function — calls /grader (STATELESS).

    FIX 2: task_ids from kwargs["task_id"] directly (TRL passes dataset
            columns as direct kwargs, NOT inside a "batch" key).

    FIX 3: calls /grader instead of /reset + /step.
            /grader is stateless — no race condition, no global env mutation,
            no flat reward from concurrent resets overwriting each other.
    """
    rewards  = []

    # ── FIX 2: correct task_id extraction ────────────────────────
    # TRL GRPO passes dataset columns directly as kwargs.
    # With num_generations=4, each task_id is repeated 4x in the list.
    raw_task_ids = kwargs.get("task_id", [])
    if isinstance(raw_task_ids, str):
        raw_task_ids = [raw_task_ids]

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        task_id = (
            raw_task_ids[i]
            if i < len(raw_task_ids)
            else "easy_s001"
        )

        # ── Extract text from completion ──────────────────────────
        text = _completion_to_text(completion)

        # ── FIX 1: robust JSON parse ──────────────────────────────
        action = _extract_json(text)

        if action is None:
            # Complete parse failure — 0.001 (not 0.0, avoids GRPO div-by-zero)
            print(f"  [REWARD] scenario={task_id} | INVALID JSON | score=0.001")
            rewards.append(0.001)
            continue

        # Format bonus: valid JSON with correct structure = small positive signal
        # This gives the model SOMETHING to learn from even before it learns
        # the right actions, avoiding the all-zero gradient problem.
        if not _is_valid_action(action):
            # JSON parsed but action_type is wrong/missing
            print(f"  [REWARD] scenario={task_id} | INVALID ACTION | score=0.05")
            rewards.append(0.05)
            continue

        # ── FIX 3: stateless /grader call ────────────────────────
        try:
            resp = requests.post(
                f"{ENV_URL}/grader",
                json={"task_id": task_id, "action": action},
                timeout=20,
            )
            resp.raise_for_status()
            score = float(resp.json().get("score", 0.001))
            score = max(0.001, min(0.999, score))
            print(f"  [REWARD] scenario={task_id} | action={action['action_type']} | score={score:.3f}")
            rewards.append(score)

        except requests.exceptions.Timeout:
            print(f"  [REWARD] scenario={task_id} | GRADER TIMEOUT | score=0.05")
            rewards.append(0.05)   # grader timed out — give format credit
        except Exception as e:
            print(f"[reward_fn] grader call failed for {task_id}: {e}")
            rewards.append(0.001)

    return rewards


# ─────────────────────────────────────────────
#  BUILD TRAINING DATASET
# ─────────────────────────────────────────────

def build_dataset():
    """
    Build training examples from all Round 2 scenario JSON files.
    Each example: {"prompt": "...", "task_id": "easy_s001"}.
    task_id is passed through to reward_fn via kwargs (TRL behaviour).
    """
    scenarios = []

    for fname in [
        "dataset/easy_scenarios.json",
        "dataset/medium_scenarios.json",
        "dataset/hard_scenarios.json",
    ]:
        try:
            with open(fname) as f:
                loaded = json.load(f)
                scenarios.extend(loaded)
                print(f"  Loaded {len(loaded)} scenarios from {fname}")
        except FileNotFoundError:
            print(f"  {fname} not found, skipping")

    if not scenarios:
        print("  Falling back to /tasks endpoint...")
        try:
            resp     = requests.get(f"{ENV_URL}/tasks", timeout=15)
            tasks    = resp.json().get("tasks", [])
            scenarios = [{"id": t["id"], "description": t.get("description", "")}
                         for t in tasks if "_s" in t["id"]]
        except Exception as e:
            print(f"  /tasks fallback failed: {e}")
            # Minimal fallback so training doesn't crash
            scenarios = [{"id": "easy_s001",
                          "description": "User lookup query taking 2s. Add index.",
                          "tables": [{"name": "users", "rows": 10000, "indexes": ["PRIMARY"]}],
                          "slow_queries": [{"id": "q1", "sql": "SELECT * FROM users WHERE email=?", "avg_ms": 2000}],
                          "performance_score_baseline": 8.0,
                          "target_score": 80.0}]

    examples = []
    for s in scenarios:
        tables_txt = json.dumps(s.get("tables", []), separators=(",", ":"))
        queries_txt = json.dumps(s.get("slow_queries", []), separators=(",", ":"))
        baseline    = s.get("performance_score_baseline", s.get("performance_score", 0))
        target      = s.get("target_score", 85)
        max_steps   = s.get("max_steps", 50)

        prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Scenario: {s.get('id')} — {s.get('description','')}\n"
        f"Slow queries: {', '.join(q['id'] for q in s.get('slow_queries',[]))}\n"
        f"Tables: {', '.join(t['name'] for t in s.get('tables',[]))}\n"
        f"Missing indexes: {json.dumps(s.get('missing_index_hints',[]))}\n"
        f"Performance: {baseline}/100 → target {target}/100\n\n"
        f"Respond with JSON only:"
    )


        examples.append({
            "prompt":  prompt,
            "task_id": s.get("id", "easy_s001"),
        })

    print(f"Built {len(examples)} training examples from {len(scenarios)} scenarios")
    return Dataset.from_list(examples)


# ─────────────────────────────────────────────
#  REWARD WRAPPER  (FIX 2 continued)
# ─────────────────────────────────────────────

def reward_wrapper(prompts, completions, **kwargs):
    """
    Thin wrapper — passes kwargs straight through.
    TRL GRPO sends dataset columns (including task_id) as direct kwargs.
    DO NOT use kwargs.get("batch") — that key does not exist in TRL GRPO.
    """
    return reward_fn(prompts, completions, **kwargs)


# ─────────────────────────────────────────────
#  MAIN TRAINING
# ─────────────────────────────────────────────

def train():
    if not UNSLOTH_AVAILABLE:
        print("Cannot train — Unsloth not installed")
        print("Run: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' trl transformers datasets accelerate")
        return

    print(f"🚀 Loading model: {MODEL_NAME}")
    print(f"🌐 Environment:   {ENV_URL}")

    # Sanity check — make sure environment is reachable
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        print(f"✅ Environment health: {r.json()}")
    except Exception as e:
        print(f"⚠️  Cannot reach environment at {ENV_URL}: {e}")
        print("   Training will likely fail — check ENV_URL")

    # Load model with Unsloth 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = 4096,
        load_in_4bit   = True,
        token          = HF_TOKEN or None,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        lora_alpha     = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # Build dataset
    dataset = build_dataset()

    # GRPO config
    # Avoid generation warnings from simultaneously using max_length and max_new_tokens.
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    if hasattr(model, "config"):
        # Some stacks still read generation defaults from model.config.
        model.config.max_length = None

    config = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 100,      # was 3  → now 100 → gives ~120 steps
        per_device_train_batch_size = 1,        # was 2  → frees memory for 1.5B
        gradient_accumulation_steps = 2,        # was 8  → was hiding steps, now shows them
        learning_rate               = 2e-5,     # was 5e-5 → lower = more stable for small model
        max_completion_length       = 120,
        num_generations             = 4,
        logging_steps               = 1,        # was 10 → see every step
        save_steps                  = 20,
        warmup_steps                = 10,       # was warmup_ratio (deprecated)
        report_to                   = "none",
    )

    trainer = GRPOTrainer(
        model         = model,
        tokenizer     = tokenizer,
        reward_funcs  = reward_wrapper,
        args          = config,
        train_dataset = dataset,
    )

    print("🏋️  Starting GRPO training...")
    print("   Expected reward progression:")
    print("   Steps  10: ~0.05-0.15 (model still outputting free text)")
    print("   Steps  50: ~0.20-0.35 (learning JSON format)")
    print("   Steps 100: ~0.35-0.50 (learning correct action types)")
    print("   Steps 200: ~0.55-0.70 (learning DBA investigation pattern)")
    print("   Steps 300: ~0.70-0.82 (strategic multi-action planning)")

    trainer.train()

    # Save
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"✅ Training complete. Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    train()