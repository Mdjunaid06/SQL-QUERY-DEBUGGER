"""
training/train_agent.py — SQL Database Engineer Agent
Unsloth + GRPO — Fixed Version
- 100 steps (not 700)
- Local DatabaseSimulator for variance (not just /grader)
- Real delta rewards, milestones shown in logs
- Generates loss_curve.png automatically
"""

import os, re, json, sys, warnings
from pathlib import Path

# ── Suppress known warnings ───────────────────────────────────
warnings.filterwarnings("ignore", message=r".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=r".*AttentionMaskConverter.*", category=FutureWarning)

# ── GPU check + Unsloth ───────────────────────────────────────
UNSLOTH_AVAILABLE = False
try:
    import torch
    if not torch.cuda.is_available():
        print("❌ No GPU. Unsloth requires CUDA.")
        sys.exit(1)
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset
    UNSLOTH_AVAILABLE = True
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
except ImportError as e:
    print(f"❌ {e}")
    sys.exit(1)

# ── Add project root to path ──────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.db_simulator import DatabaseSimulator

# ── Config ────────────────────────────────────────────────────
ENV_URL    = os.getenv("ENV_URL",    "https://junaid0600-sql-db-engineer-agent.hf.space")
HF_TOKEN   = os.getenv("HF_TOKEN",  "")
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./sdea-trained")
MAX_STEPS  = int(os.getenv("MAX_STEPS", "100"))

print(f"\n[CONFIG] Model:     {MODEL_NAME}")
print(f"[CONFIG] Max steps: {MAX_STEPS}")
print(f"[CONFIG] Output:    {OUTPUT_DIR}\n")

VALID_ACTIONS = {
    "inspect_query", "analyze_indexes", "create_index",
    "rewrite_query", "add_column", "drop_index",
    "partition_table", "analyze_statistics",
    "request_hint", "submit_report",
}

SYSTEM_PROMPT = """You are a senior database engineer fixing slow database queries.

Investigation pattern:
1. inspect_query  → understand WHY query is slow
2. analyze_indexes → see what indexes are missing
3. create_index   → add composite index on WHERE/JOIN columns
4. submit_report  → when performance target is reached

RESPOND WITH VALID JSON ONLY. No markdown. No explanation.
Examples:
{"action_type": "inspect_query", "payload": {"query_id": "q1"}}
{"action_type": "create_index", "payload": {"table": "users", "columns": ["email"]}}
{"action_type": "create_index", "payload": {"table": "orders", "columns": ["user_id", "status"]}}"""


# ── Load all 15 scenarios ─────────────────────────────────────
def load_scenarios() -> list:
    scenarios = []
    for fname in ["easy_scenarios.json", "medium_scenarios.json", "hard_scenarios.json"]:
        path = os.path.join(ROOT, "dataset", fname)
        try:
            with open(path) as f:
                data = json.load(f)
                scenarios.extend(data)
                print(f"  ✅ {len(data)} from {fname}")
        except FileNotFoundError:
            print(f"  ⚠️  {fname} not found")
    print(f"  Total: {len(scenarios)} scenarios\n")
    return scenarios

ALL_SCENARIOS = load_scenarios()


# ── Parse LLM output → action dict ───────────────────────────
def parse_action(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Try full parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try extract from partial text
    for start in [i for i,c in enumerate(text) if c == "{"]:
        depth, in_str, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape: escape = False
                elif ch == "\\": escape = True
                elif ch == '"': in_str = False
                continue
            if ch == '"': in_str = True; continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i+1])
                        if isinstance(obj, dict) and "action_type" in obj:
                            return obj
                    except: pass
                    break
    return None


# ── Local reward using DatabaseSimulator ──────────────────────
def compute_reward(action: dict, scenario: dict) -> tuple:
    """
    Compute reward LOCALLY — no HTTP, no shared state, deterministic.
    Returns (score, improvement_pts, milestone_bonus, description)
    """
    sim      = DatabaseSimulator(scenario)
    baseline = sim.get_performance_score()
    action_type = action.get("action_type", "")
    payload     = action.get("payload", {})

    # Apply action
    delta = 0.0
    if action_type == "create_index":
        result = sim.apply_action("create_index", payload)
        delta  = result.get("delta", 0.0)
    elif action_type == "rewrite_query":
        result = sim.apply_action("rewrite_query", payload)
        delta  = result.get("delta", 0.0)
    elif action_type == "partition_table":
        result = sim.apply_action("partition_table", payload)
        delta  = result.get("delta", 0.0)
    elif action_type == "analyze_statistics":
        result = sim.apply_action("analyze_statistics", payload)
        delta  = result.get("delta", 0.0)
    elif action_type in ("inspect_query", "analyze_indexes"):
        delta = 0.0  # investigation — no DB change
    elif action_type == "submit_report":
        delta = max(0, sim.get_performance_score() - baseline)

    final       = sim.get_performance_score()
    improvement = max(0.0, final - baseline)
    max_possible = max(1.0, 100.0 - baseline)
    ratio = improvement / max_possible

    # Step reward
    step_r = {
        "inspect_query":     0.10,
        "analyze_indexes":   0.10,
        "create_index":      0.15,
        "rewrite_query":     0.20,
        "analyze_statistics":0.08,
        "partition_table":   0.15,
        "submit_report":     0.05,
    }.get(action_type, 0.001)

    # Delta reward — key signal
    delta_r = min(0.65, ratio * 0.65)

    # Milestone bonus
    milestone = 0.0
    milestone_str = ""
    if ratio >= 0.75:
        milestone = 0.40
        milestone_str = "🎯 75% milestone!"
    elif ratio >= 0.50:
        milestone = 0.25
        milestone_str = "🎯 50% milestone!"
    elif ratio >= 0.25:
        milestone = 0.15
        milestone_str = "🎯 25% milestone!"

    # Wrong index penalty
    wrong_pen = -0.15 if (action_type == "create_index" and delta <= 0.0) else 0.0

    total = max(0.001, min(0.999, step_r + delta_r + milestone + wrong_pen))
    desc  = f"+{improvement:.1f}pts delta={delta:.1f} {milestone_str}"

    return total, improvement, milestone, desc


# ── GRPO reward function ──────────────────────────────────────
def reward_fn(prompts, completions, **kwargs):
    """
    LOCAL reward — DatabaseSimulator directly.
    Rewards vary 0.001 to 0.999 giving GRPO real gradient signal.
    """
    rewards = []

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            # Get text
            if isinstance(completion, list):
                text = completion[0].get("content","") if completion else ""
            else:
                text = str(completion)

            # Pick scenario
            scenario = ALL_SCENARIOS[i % len(ALL_SCENARIOS)]
            sid      = scenario["id"]

            # Parse
            action = parse_action(text)

            if action is None:
                print(f"  [REWARD] {sid} | INVALID JSON | score=0.001")
                rewards.append(0.001)
                continue

            if action.get("action_type") not in VALID_ACTIONS:
                print(f"  [REWARD] {sid} | UNKNOWN ACTION | score=0.05")
                rewards.append(0.05)
                continue

            # Compute locally
            score, improvement, milestone, desc = compute_reward(action, scenario)
            rewards.append(score)

            print(f"  [REWARD] {sid} | "
                  f"action={action['action_type']} | "
                  f"{desc} | score={score:.3f}")

        except Exception as e:
            print(f"  [REWARD] Error: {e}")
            rewards.append(0.001)

    return rewards


# ── Build dataset ─────────────────────────────────────────────
def build_dataset() -> Dataset:
    examples = []
    for s in ALL_SCENARIOS:
        tables_str  = json.dumps(s.get("tables", []))
        queries_str = json.dumps(s.get("slow_queries", []))
        hints_str   = json.dumps(s.get("missing_index_hints", []))
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== DATABASE STATE ===\n"
            f"Scenario: {s['id']} — {s.get('description','')}\n"
            f"Tables: {tables_str}\n"
            f"Slow Queries: {queries_str}\n"
            f"Missing Index Hints: {hints_str}\n"
            f"Performance: {s.get('performance_score_baseline',0)}/100 "
            f"→ Target: {s.get('target_score',85)}/100\n\n"
            f"What is your next action? JSON only:"
        )
        examples.append({"prompt": prompt, "scenario_id": s["id"]})

    print(f"  ✅ Dataset: {len(examples)} examples")
    return Dataset.from_list(examples)


# ── Generate loss + reward plots ──────────────────────────────
def generate_plots(trainer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = [l for l in trainer.state.log_history if "loss" in l]
    if not logs:
        print("⚠️ No logs to plot")
        return

    steps   = [l.get("step", i) for i,l in enumerate(logs)]
    losses  = [l.get("loss", 0) for l in logs]
    rewards = [l.get("reward", 0) for l in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("GRPO Training — SQL Database Engineer Agent",
                 fontsize=13, fontweight="bold")

    ax1.plot(steps, losses, "b-o", lw=2, ms=4, label="Loss")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss (↓ = model learning)")
    ax1.grid(True, alpha=0.3)
    if losses:
        ax1.annotate(f"Start: {losses[0]:.4f}", xy=(steps[0], losses[0]),
                    xytext=(steps[0]+1, losses[0]*1.1), fontsize=8, color="red")
        ax1.annotate(f"End: {losses[-1]:.4f}", xy=(steps[-1], losses[-1]),
                    xytext=(steps[-1]-8, losses[-1]*1.15), fontsize=8, color="green")

    ax2.plot(steps, rewards, "g-o", lw=2, ms=4, label="Avg Reward")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Reward")
    ax2.set_title("Reward During Training (↑ = improving)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
    print("✅ loss_curve.png saved")
    if losses:
        print(f"   Loss:   {losses[0]:.4f} → {losses[-1]:.4f}")
    if rewards:
        valid = [r for r in rewards if r > 0]
        if valid:
            print(f"   Reward: {valid[0]:.4f} → {valid[-1]:.4f}")


# ── Main ──────────────────────────────────────────────────────
def train():
    if not ALL_SCENARIOS:
        print("❌ No scenarios loaded. Check dataset/ folder.")
        sys.exit(1)

    print(f"⏳ Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = 2048,
        load_in_4bit   = True,
        token          = HF_TOKEN or None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("✅ Model + LoRA ready\n")

    dataset = build_dataset()

    config = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        max_steps                   = MAX_STEPS,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        learning_rate               = 2e-5,
        max_completion_length       = 150,
        num_generations             = 4,
        temperature                 = 1.0,
        logging_steps               = 1,
        save_steps                  = 25,
        save_total_limit            = 3,
        warmup_steps                = 10,
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    trainer = GRPOTrainer(
        model         = model,
        tokenizer     = tokenizer,
        reward_funcs  = reward_fn,
        args          = config,
        train_dataset = dataset,
    )

    print(f"🏋️  GRPO training — {MAX_STEPS} steps")
    print("Expected rewards:")
    print("  inspect_query (always):        ~0.10")
    print("  create_index (wrong columns):  ~0.001")
    print("  create_index (right columns):  ~0.75-0.99")
    print("  GRPO learns: right create_index >> everything else\n")

    trainer.train()
    print("\n✅ Training complete!")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"✅ Saved to {OUTPUT_DIR}/final")

    generate_plots(trainer)

    print("\n" + "="*50)
    print("NEXT:")
    print("  python training/evaluate_agent.py")
    print("  git add loss_curve.png reward_curve.png")
    print("  git commit -m 'Real GRPO training evidence'")
    print("  git push origin main")
    print("="*50)


if __name__ == "__main__":
    train()
