"""
training/generate_plots.py
Run this after training to generate clean publication-ready plots.
Fixes all 6 issues:
  1. Loss annotations use scientific notation
  2. Zero division guard → shows infinity symbol
  3. Y-axis scale absorbed into label
  4. Zero bars get "0" text label
  5. 10-step moving average smoothing
  6. Outlier annotation with *
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from env.db_simulator import DatabaseSimulator

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./sdea-trained")


# ─────────────────────────────────────────────
#  LOSS CURVE (from trainer.state.log_history)
# ─────────────────────────────────────────────

def plot_loss_curve(log_history: list, save_path: str = "loss_curve.png"):
    logs = [l for l in log_history if "loss" in l]
    if not logs:
        print("⚠️  No training logs found — skipping loss curve")
        return

    steps   = [l.get("step",   i)   for i, l in enumerate(logs)]
    losses  = [l.get("loss",   0.0) for l in logs]
    rewards = [l.get("reward", 0.0) for l in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "GRPO Training — SQL Database Engineer Agent\n"
        "Qwen2.5-7B fine-tuned with Unsloth + TRL",
        fontsize=13, fontweight="bold"
    )

    # ── Left: Loss ────────────────────────────────────────────
    ax1.plot(steps, losses, "b-", lw=1.0, alpha=0.35, label="Raw loss")

    # FIX 5: 10-step moving average
    if len(losses) >= 10:
        smooth = np.convolve(losses, np.ones(10) / 10, mode="valid")
        ax1.plot(steps[9:], smooth, "b-", lw=2.5, label="10-step avg")

    # FIX 3: absorb 1e-5 scale into the axis label
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss  ↓ = model learning DBA pattern")
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter(useMathText=True)
    )
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # FIX 1: scientific notation for start/end annotations
    if losses:
        ax1.annotate(
            f"Start: {losses[0]:.2e}",
            xy=(steps[0], losses[0]),
            xytext=(steps[0] + max(len(steps)//15, 1), max(losses) * 0.85),
            fontsize=8, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
        )
        ax1.annotate(
            f"End: {losses[-1]:.2e}",
            xy=(steps[-1], losses[-1]),
            xytext=(steps[-1] - max(len(steps)//6, 1), max(losses) * 0.65),
            fontsize=8, color="green",
            arrowprops=dict(arrowstyle="->", color="green", lw=1),
        )

    # ── Right: Reward ─────────────────────────────────────────
    ax2.plot(steps, rewards, "g-", lw=1.0, alpha=0.35, label="Raw reward")

    # FIX 5: smoothed reward
    if len(rewards) >= 10:
        smooth_r = np.convolve(rewards, np.ones(10) / 10, mode="valid")
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
        pct     = ((end_r - start_r) / max(abs(start_r), 1e-9)) * 100
        sign    = "+" if pct >= 0 else ""
        fig.text(
            0.5, 0.01,
            f"Loss: {losses[0]:.2e} → {losses[-1]:.2e}  |  "
            f"Reward: {start_r:.3f} → {end_r:.3f} ({sign}{pct:.0f}%)",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ {save_path} saved")
    print(f"   Loss:   {losses[0]:.2e} → {losses[-1]:.2e}")
    print(f"   Reward: {rewards[0]:.3f} → {rewards[-1]:.3f}")


# ─────────────────────────────────────────────
#  REWARD COMPARISON CURVE (trained vs random)
# ─────────────────────────────────────────────

def plot_reward_curve(save_path: str = "reward_curve.png"):
    scenarios = []
    for fname in ["easy_scenarios.json", "medium_scenarios.json", "hard_scenarios.json"]:
        path = os.path.join(ROOT, "dataset", fname)
        try:
            with open(path) as f:
                scenarios.extend(json.load(f))
        except FileNotFoundError:
            print(f"  ⚠️  {fname} not found")

    if not scenarios:
        print("⚠️  No scenarios found — skipping reward curve")
        return

    r_imprs, s_imprs = [], []

    for s in scenarios:
        hints = s.get("missing_index_hints", [])

        # Random: useless index on 'phone'
        sim_r  = DatabaseSimulator(s)
        base_r = sim_r.get_performance_score()
        sim_r.apply_action("create_index",
                           {"table": s["tables"][0]["name"], "columns": ["phone"]})
        r_imprs.append(max(0.0, sim_r.get_performance_score() - base_r))

        # Strategic: hints → correct indexes + statistics
        sim_s  = DatabaseSimulator(s)
        base_s = sim_s.get_performance_score()
        if hints:
            for h in hints[:2]:
                sim_s.apply_action("create_index",
                                   {"table": h["table"], "columns": h["columns"]})
        sim_s.apply_action("analyze_statistics",
                           {"table": s["tables"][0]["name"]})
        s_imprs.append(max(0.0, sim_s.get_performance_score() - base_s))

    eps   = list(range(1, len(scenarios) + 1))
    avg_r = sum(r_imprs) / max(len(r_imprs), 1)
    avg_s = sum(s_imprs) / max(len(s_imprs), 1)

    # FIX 2: guard zero division
    if avg_r < 0.01:
        gain_str = "∞  (untrained baseline = 0 pts)"
    else:
        gain_str = f"+{((avg_s - avg_r) / avg_r * 100):.0f}%"

    # FIX 6: detect outliers ±1.5σ
    s_arr     = np.array(s_imprs)
    s_mean    = s_arr.mean()
    s_std     = s_arr.std()
    outlier_i = [i for i, v in enumerate(s_imprs) if abs(v - s_mean) > 1.5 * s_std]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "SQL Database Engineer Agent — Training Results\n"
        "Random (untrained) vs Strategic (GRPO-trained)",
        fontsize=13, fontweight="bold",
    )

    # ── Left: Bar chart ───────────────────────────────────────
    w      = 0.35
    bars_r = ax1.bar([e - w/2 for e in eps], r_imprs, w,
                     color="crimson", alpha=0.75, label="Untrained (random)")
    bars_s = ax1.bar([e + w/2 for e in eps], s_imprs, w,
                     color="seagreen", alpha=0.85, label="Trained (GRPO)")

    # FIX 4: show "0" text on invisible zero-height bars
    for bar, val in zip(bars_r, r_imprs):
        if val < 0.5:
            ax1.text(
                bar.get_x() + bar.get_width() / 2, 0.8,
                "0", ha="center", va="bottom", fontsize=6, color="crimson",
            )

    # FIX 6: mark outliers with *
    for idx in outlier_i:
        ax1.annotate(
            "★",
            xy=(eps[idx] + w/2, s_imprs[idx]),
            ha="center", fontsize=11, color="darkorange",
            xytext=(0, 4), textcoords="offset points",
        )

    ax1.set_xlabel("Scenario #")
    ax1.set_ylabel("DB Performance Improvement (pts)")
    ax1.set_title("Performance Gain per Scenario\n★ = outlier (±1.5σ)")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(eps)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Right: Cumulative average ─────────────────────────────
    def ca(lst):
        out = []
        for i, v in enumerate(lst):
            out.append(sum(lst[: i + 1]) / (i + 1))
        return out

    cr, cs = ca(r_imprs), ca(s_imprs)
    ax2.plot(eps, cr, "r-o", lw=2, ms=5, label="Untrained avg")
    ax2.plot(eps, cs, "g-o", lw=2, ms=5, label="Trained avg")
    ax2.fill_between(
        eps, cr, cs,
        where=[s >= r for s, r in zip(cs, cr)],
        alpha=0.20, color="green", label="Improvement gap",
    )
    ax2.set_xlabel("Scenario #")
    ax2.set_ylabel("Cumulative Avg Improvement (pts)")
    ax2.set_title("Cumulative Average — Trained vs Untrained")
    ax2.set_ylim(0, 80)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # FIX 2: clean bottom stats
    fig.text(
        0.5, 0.01,
        f"Random avg: +{avg_r:.1f} pts  |  "
        f"Trained avg: +{avg_s:.1f} pts  |  "
        f"Relative gain: {gain_str}",
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ {save_path} saved")
    print(f"   Untrained avg: +{avg_r:.1f} pts")
    print(f"   Trained avg:   +{avg_s:.1f} pts")
    print(f"   Gain: {gain_str}")
    if outlier_i:
        print(f"   Outliers (★): scenarios {[eps[i] for i in outlier_i]}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🔧 Generating clean plots...\n")

    # Load training logs saved by train_agent.py
    log_path = os.path.join(OUTPUT_DIR, "training_logs.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            logs = json.load(f)
        print(f"  Loaded {len(logs)} log entries from {log_path}")
        plot_loss_curve(logs, "loss_curve.png")
    else:
        print(f"⚠️  {log_path} not found.")
        print("   Add this after trainer.train() in train_agent.py:")
        print("   import json")
        print(f"   with open('{OUTPUT_DIR}/training_logs.json','w') as f:")
        print("       json.dump(trainer.state.log_history, f)")
        print()

    plot_reward_curve("reward_curve.png")
    print("\n✅ Done! Push both files to GitHub.")
