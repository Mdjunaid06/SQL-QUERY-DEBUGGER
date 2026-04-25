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
from collections import defaultdict

# ── Try importing Unsloth (GPU only) ─────────────────────────
try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available. Run: pip install unsloth trl")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

ENV_URL    = os.getenv("ENV_URL",    "https://junaid0600-sql-db-engineer-agent.hf.space")
HF_TOKEN   = os.getenv("HF_TOKEN",  "")
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./sdea-trained")
MAX_STEPS  = int(os.getenv("MAX_STEPS", "50"))

# Valid Round 2 action types — model must use one of these
VALID_ACTION_TYPES = {
    "inspect_query", "analyze_indexes", "create_index",
    "rewrite_query", "add_column", "drop_index",
    "partition_table", "analyze_statistics",
    "request_hint", "submit_report",
}

_last_reward_by_task = {}
_scenario_hits = defaultdict(int)

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


def _configure_warning_filters() -> None:
    """
    Suppress noisy upstream deprecation warnings that do not affect training
    correctness right now. Keep real runtime errors visible.
    """
    filters = [
        r"Passing `generation_config` together with generation-related arguments",
        r"Both `max_new_tokens` \(=.+\) and `max_length`\(.+\) seem to have been set",
        r"The attention mask API under `transformers\.modeling_attn_mask_utils`",
        r"`use_return_dict` is deprecated! Use `return_dict` instead!",
    ]
    for message in filters:
        warnings.filterwarnings("ignore", message=message, category=FutureWarning)
        warnings.filterwarnings("ignore", message=message, category=UserWarning)


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

    # Try 2: scan for JSON object starts and raw-decode from each "{"
    # This handles nested objects/arrays better than regex-only extraction.
    decoder = json.JSONDecoder()
    brace_positions = [m.start() for m in re.finditer(r"\{", text)]
    for pos in brace_positions:
        try:
            obj, _ = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except Exception:
            continue

    # Try 3: greedy — find first { to last }
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict) and "action_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _recover_action_from_text(text: str) -> dict | None:
    """
    Heuristic fallback when strict JSON parse fails.
    Helps GRPO get useful gradient instead of all-invalid outputs.
    """
    if not text:
        return None

    action_match = re.search(
        r"(inspect_query|analyze_indexes|create_index|rewrite_query|add_column|drop_index|partition_table|analyze_statistics|request_hint|submit_report)",
        text,
        flags=re.IGNORECASE,
    )
    if not action_match:
        return None

    action_type = action_match.group(1).lower()
    payload = {}

    query_match = re.search(r"\bq\d+\b", text, flags=re.IGNORECASE)
    table_match = re.search(
        r"\b(users|orders|products|sessions|logs|tickets|events|posts|authors|transactions|accounts|customers|audit_log|workspaces|projects|tasks|comments|attachments|activity_log|notifications|patients|appointments|prescriptions|clinical_notes|players|matches|leaderboards|achievements|shipments|routes|drivers|vehicles|warehouses|tracking)\b",
        text,
        flags=re.IGNORECASE,
    )
    cols_match = re.search(r"\[(.*?)\]", text, flags=re.DOTALL)

    if action_type == "inspect_query":
        payload["query_id"] = query_match.group(0) if query_match else "q1"
    elif action_type in {"analyze_indexes", "partition_table", "analyze_statistics"}:
        payload["table"] = table_match.group(1) if table_match else "orders"
    elif action_type == "create_index":
        payload["table"] = table_match.group(1) if table_match else "orders"
        if cols_match:
            cols = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cols_match.group(1))
            payload["columns"] = cols[:4] if cols else ["user_id", "status"]
        else:
            payload["columns"] = ["user_id", "status"]
    elif action_type == "submit_report":
        payload["summary"] = "Applied index and query optimization recommendations."
    else:
        payload = {}

    return {"action_type": action_type, "payload": payload}


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
        if raw_task_ids:
            task_id = raw_task_ids[i % len(raw_task_ids)]
        else:
            # Last-resort fallback should not pin all rewards to one scenario.
            task_id = f"fallback_{i}"
        _scenario_hits[task_id] += 1

        # ── Extract text from completion ──────────────────────────
        if isinstance(completion, list):
            # Standard TRL format: [{"role": "assistant", "content": "..."}]
            text = completion[0].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        # ── FIX 1: robust JSON parse ──────────────────────────────
        action = _extract_json(text)
        recovered = False
        if action is None:
            action = _recover_action_from_text(text)
            recovered = action is not None

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
            print(
                f"  [REWARD] scenario={task_id} | action={action.get('action_type', 'unknown')} "
                f"| INVALID ACTION STRUCTURE | score=0.05"
            )
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
            rewards.append(score)
            delta_reward = score - _last_reward_by_task.get(task_id, score)
            _last_reward_by_task[task_id] = score
            parse_tag = "RECOVERED" if recovered else "JSON_OK"
            print(
                f"  [REWARD] scenario={task_id} | action={action['action_type']} | {parse_tag} "
                f"| score={score:.3f} | delta_reward={delta_reward:+.3f}"
            )

        except requests.exceptions.Timeout:
            print(f"  [REWARD] scenario={task_id} | grader TIMEOUT | score=0.05")
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
    _configure_warning_filters()

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
    dataset = build_dataset().shuffle(seed=42)

    # GRPO config
    config = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        # Set explicit optimizer steps so "Total steps" is predictable.
        max_steps                   = MAX_STEPS,
        num_train_epochs            = 100,      # ignored when max_steps is reached first
        per_device_train_batch_size = 1,        # was 2  → frees memory for 1.5B
        gradient_accumulation_steps = 2,        # was 8  → was hiding steps, now shows them
        learning_rate               = 2e-5,     # was 5e-5 → lower = more stable for small model
        max_completion_length       = 512,
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
    print(f"   Target optimizer steps: {MAX_STEPS}")
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