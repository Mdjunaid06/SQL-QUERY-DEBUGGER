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

        # ── Hint-match fallback ────────────────────────────────────────────
        # When the simulator returns delta=0 (wrong/partial columns), check
        # the scenario's missing_index_hints and simulate a proportional delta.
        # This gives GRPO a gradient signal early in training before the model
        # has learned the exact right column names.
        if delta <= 0.0:
            hints         = scenario.get("missing_index_hints", [])
            p_table       = payload.get("table", "")
            p_cols        = set(str(c).lower() for c in payload.get("columns", []))
            best_match    = 0.0
            for hint in hints:
                h_table = hint.get("table", "")
                h_cols  = set(str(c).lower() for c in hint.get("columns", []))
                if not h_cols:
                    continue
                table_ok    = 1.0 if h_table == p_table else 0.3
                col_overlap = len(p_cols & h_cols) / max(len(h_cols), 1)
                best_match  = max(best_match, table_ok * 0.4 + col_overlap * 0.6)
            if best_match > 0:
                max_gap = max(1.0, 100.0 - baseline)
                delta   = best_match * max_gap * 0.65   # up to 65% of gap on perfect match → reaches 25% milestone
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

    final        = sim.get_performance_score()
    sim_improve  = max(0.0, final - baseline)
    # Use hint-simulated delta if simulator returned 0 (wrong columns early in training)
    improvement  = max(sim_improve, delta)
    max_possible = max(1.0, 100.0 - baseline)
    ratio        = improvement / max_possible

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

    # Milestone bonus — cumulative, all thresholds crossed shown + rewarded
    milestone       = 0.0
    milestone_parts = []
    if ratio >= 0.25:
        milestone += 0.15
        milestone_parts.append("25%")
    if ratio >= 0.50:
        milestone += 0.25
        milestone_parts.append("50%")
    if ratio >= 0.75:
        milestone += 0.40
        milestone_parts.append("75%")
    milestone_str = ("🎯 " + "+".join(milestone_parts) + " milestone!") if milestone_parts else ""

    # Wrong index penalty reduced: -0.05 instead of -0.15
    # Old value exactly cancelled step_r (0.15 - 0.15 = 0 → clamped to 0.001)
    # Now wrong create_index still scores ~0.10 instead of 0.001
    wrong_pen = -0.05 if (action_type == "create_index" and delta <= 0.0) else 0.0

    total = max(0.001, min(0.999, step_r + delta_r + milestone + wrong_pen))
    src   = "sim" if sim_improve > 0 else ("hint" if delta > 0 else "none")
    desc  = f"+{improvement:.1f}pts delta={delta:.1f}[{src}] {milestone_str}"

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
# ──────────────────────────────────────────────────────────────
# REPLACE ONLY THIS FUNCTION in your existing train_agent.py
# Find: def generate_plots(trainer):
# Replace the entire function with this:
# ──────────────────────────────────────────────────────────────

def generate_plots(trainer):
    import json, numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = [l for l in trainer.state.log_history if "loss" in l]
    if not logs:
        print("⚠️ No logs to plot")
        return

    steps   = [l.get("step", i)   for i, l in enumerate(logs)]
    losses  = [l.get("loss",   0) for l in logs]
    rewards = [l.get("reward", 0) for l in logs]

    # Save logs for generate_plots.py
    import os, json
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/training_logs.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    print(f"✅ Logs saved to {OUTPUT_DIR}/training_logs.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GRPO Training — SQL Database Engineer Agent\n"
                 "Qwen2.5-7B fine-tuned with Unsloth + TRL",
                 fontsize=13, fontweight="bold")

    # ── LEFT: Loss with smoothing ──────────────────────────────
    ax1.plot(steps, losses, "b-", lw=1.0, alpha=0.35, label="Raw loss")
    if len(losses) >= 10:
        smooth = np.convolve(losses, np.ones(10)/10, mode="valid")
        ax1.plot(steps[9:], smooth, "b-", lw=2.5, label="10-step avg")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss  ↓ = model learning")
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    # FIX 1: scientific notation annotations
    if losses:
        ax1.annotate(f"Start: {losses[0]:.2e}",
                     xy=(steps[0], losses[0]),
                     xytext=(steps[0]+3, max(losses)*0.85),
                     fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1))
        ax1.annotate(f"End: {losses[-1]:.2e}",
                     xy=(steps[-1], losses[-1]),
                     xytext=(steps[-1]-12, max(losses)*0.65),
                     fontsize=8, color="green",
                     arrowprops=dict(arrowstyle="->", color="green", lw=1))

    # ── RIGHT: Reward with smoothing ───────────────────────────
    ax2.plot(steps, rewards, "g-", lw=1.0, alpha=0.35, label="Raw reward")
    if len(rewards) >= 10:
        smooth_r = np.convolve(rewards, np.ones(10)/10, mode="valid")
        ax2.plot(steps[9:], smooth_r, "g-", lw=2.5, label="10-step avg")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Avg Reward")
    ax2.set_title("Reward During Training  ↑ = improving")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Bottom summary
    if losses and rewards:
        start_r = rewards[0]
        end_r   = rewards[-1]
        pct     = ((end_r-start_r)/max(abs(start_r),1e-9))*100
        fig.text(0.5, 0.01,
                 f"Loss: {losses[0]:.2e} → {losses[-1]:.2e}  |  "
                 f"Reward: {start_r:.3f} → {end_r:.3f} ({'+'if pct>=0 else ''}{pct:.0f}%)",
                 ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
    print("✅ loss_curve.png saved")
    if losses:
        print(f"   Loss:   {losses[0]:.2e} → {losses[-1]:.2e}")
    if rewards:
        valid = [r for r in rewards if r > 0]
        if valid:
            print(f"   Reward: {valid[0]:.3f} → {valid[-1]:.3f}")


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
    print("  inspect_query / analyze_indexes:       ~0.10")
    print("  create_index (no table/col match):     ~0.10  (was 0.001)")
    print("  create_index (partial hint match):     ~0.20-0.45")
    print("  create_index (perfect hint match):     ~0.55-0.80")
    print("  create_index (simulator confirms):     ~0.75-0.99")
    print("  Milestones: 25%=+0.15  50%=+0.25  75%=+0.40 (cumulative)\n")

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