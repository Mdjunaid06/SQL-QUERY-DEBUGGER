from env.models import Action, Reward, DifficultyLevel, ActionType
from env.graders import grade

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

MAX_STEPS             = 50    # Round 2: long-horizon episodes
HINT_PENALTY          = -0.10  # Per hint requested (increased from Round 1)
LOOP_PENALTY          = -0.08  # Same action on same target 2+ times, no improvement
INVALID_PENALTY       = -0.10  # Null / malformed action
BACKTRACK_PENALTY     = -0.05  # Action makes score worse than previous best
BUDGET_EXHAUSTION_PEN = -0.15  # Reaching max_steps without submitting report
EFFICIENCY_BONUS      =  0.10  # Solved in < 70% of max_steps

# Milestone thresholds: {improvement_fraction: bonus_reward}
MILESTONE_THRESHOLDS = {
    0.25: 0.15,   # 25% improvement → +0.15 bonus
    0.50: 0.25,   # 50% improvement → +0.25 bonus
    0.75: 0.40,   # 75% improvement → +0.40 bonus
}

# Step rewards for Round 2 actions (dense signal)
STEP_REWARDS = {
    # ── Round 2 actions ──────────────────────────
    ActionType.INSPECT_QUERY:    0.05,   # Investigation rewarded
    ActionType.ANALYZE_INDEXES:  0.05,   # Investigation rewarded
    ActionType.CREATE_INDEX:     0.10,   # Core optimization action
    ActionType.REWRITE_QUERY:    0.15,   # High-value rewrite
    ActionType.ADD_COLUMN:       0.08,   # Denormalization
    ActionType.DROP_INDEX:       0.05,   # Clean up overhead
    ActionType.PARTITION_TABLE:  0.15,   # Big structural improvement
    ActionType.ANALYZE_STATS:    0.05,   # Maintenance action
    ActionType.SUBMIT_REPORT:    0.00,   # Terminal — score comes from grader
    ActionType.REQUEST_HINT:     0.00,   # No reward, only penalty
    # ── Round 1 backward compat ──────────────────
    ActionType.IDENTIFY_ERROR:   0.15,
    ActionType.PROPOSE_FIX:      0.25,
    ActionType.SUBMIT_ANSWER:    0.00,
    ActionType.EXPLAIN_ISSUE:    0.10,
    ActionType.OPTIMIZE_QUERY:   0.20,
}

# Terminal actions that end the episode
TERMINAL_ACTIONS = {
    ActionType.SUBMIT_ANSWER,
    ActionType.OPTIMIZE_QUERY,
    ActionType.SUBMIT_REPORT,
}


# ─────────────────────────────────────────────
#  MILESTONE TRACKER
# ─────────────────────────────────────────────

def check_milestones(
    baseline_score: float,
    new_score:      float,
    earned:         set,
) -> tuple[float, list[float]]:
    """
    Returns (total_bonus, newly_earned_thresholds).
    One-time bonuses — each milestone only paid once per episode.
    """
    max_possible   = max(1.0, 100.0 - baseline_score)
    improvement    = (new_score - baseline_score) / max_possible
    bonus          = 0.0
    newly_earned   = []

    for threshold, reward in MILESTONE_THRESHOLDS.items():
        if improvement >= threshold and threshold not in earned:
            bonus        += reward
            newly_earned.append(threshold)
            earned.add(threshold)

    return round(bonus, 4), newly_earned


# ─────────────────────────────────────────────
#  LOOP DETECTOR
# ─────────────────────────────────────────────

def _detect_loop(previous_actions: list[str], current_action: str) -> bool:
    """Returns True if agent has done the same action 2+ times in a row."""
    if len(previous_actions) < 1:
        return False
    last = previous_actions[-1]
    return last == current_action


def _count_consecutive(previous_actions: list[str], current_action: str) -> int:
    count = 1
    for a in reversed(previous_actions):
        if a == current_action:
            count += 1
        else:
            break
    return count


# ─────────────────────────────────────────────
#  EFFICIENCY BONUS
# ─────────────────────────────────────────────

def _efficiency_bonus(step_count: int, max_steps: int) -> float:
    """Bonus if agent finishes in < 70% of budget."""
    threshold = max_steps * 0.70
    if step_count <= threshold:
        ratio = step_count / max(1, max_steps)
        return round(EFFICIENCY_BONUS * (1.0 - ratio), 4)
    return 0.0


# ─────────────────────────────────────────────
#  MAIN REWARD FUNCTION
# ─────────────────────────────────────────────

def compute_reward(
    action:           Action,
    task_id:          str,
    difficulty:       DifficultyLevel,
    step_count:       int,
    previous_actions: list[str],
    hints_used:       int,
    estimated_steps:  int,
    action_counts:    dict[str, int],
    # Round 2 extras (optional — backward compatible)
    db_delta:         float = 0.0,     # Performance score delta from DatabaseSimulator
    baseline_score:   float = 0.0,     # Scenario baseline score
    current_score:    float = 0.0,     # Current DB performance score
    milestones_earned: set  = None,    # Set of already-earned milestone thresholds
) -> Reward:
    """
    Computes dense reward signal for every step.

    Components:
    1. Step reward     — small reward for valid action type
    2. Delta reward    — proportional to DB performance improvement (Round 2)
    3. Milestone bonus — one-time bonus at 25%/50%/75% improvement
    4. Grader score    — full score on terminal actions (Round 1 compat)
    5. Loop penalty    — repeated same action with no improvement
    6. Hint penalty    — cost per hint
    7. Backtrack penalty — action made things worse
    8. Budget penalty  — approaching max_steps without submitting
    9. Efficiency bonus — solved fast
    """

    if milestones_earned is None:
        milestones_earned = set()

    breakdown      = {}
    feedback_parts = []
    final_score    = 0.0

    # ── Edge case: null action ────────────────────────────────────
    if action is None or action.payload is None:
        return Reward(
            score=0.001,
            breakdown={"invalid_action": 0.001},
            feedback="Invalid or null action received."
        )

    action_type_val  = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
    action_type_enum = action.action_type

    # ── 1. Step reward ────────────────────────────────────────────
    step_reward = STEP_REWARDS.get(action_type_enum, 0.05)
    breakdown["step_reward"] = round(step_reward, 4)
    final_score += step_reward
    if step_reward > 0:
        feedback_parts.append(f"Action '{action_type_val}' +{step_reward}.")

    # ── 2. Delta reward (Round 2 DB performance change) ───────────
    if db_delta != 0.0:
        delta_reward = round((db_delta / 100.0) * 0.40, 4)
        delta_reward = max(-0.40, min(0.40, delta_reward))
        breakdown["delta_reward"] = delta_reward
        final_score += delta_reward
        if delta_reward > 0:
            feedback_parts.append(f"DB improved +{db_delta:.1f} pts. Delta reward +{delta_reward}.")
        elif delta_reward < 0:
            feedback_parts.append(f"DB worsened {db_delta:.1f} pts. Penalty {delta_reward}.")

    # ── 3. Milestone bonuses ──────────────────────────────────────
    if baseline_score > 0 and current_score > 0:
        milestone_bonus, newly_earned = check_milestones(
            baseline_score, current_score, milestones_earned
        )
        if milestone_bonus > 0:
            breakdown["milestone_bonus"] = milestone_bonus
            final_score += milestone_bonus
            pct = int(max(newly_earned) * 100)
            feedback_parts.append(f"🎯 Milestone! {pct}% improvement. Bonus +{milestone_bonus}!")

    # ── 4. Grader score for terminal actions (Round 1 compat) ─────
    grader_score = 0.0
    is_terminal  = action_type_enum in TERMINAL_ACTIONS

    if is_terminal and action_type_enum != ActionType.SUBMIT_REPORT:
        raw_score, grader_breakdown, grader_feedback = grade(action, task_id)
        grader_score = raw_score
        breakdown["grader_score"]     = round(grader_score, 4)
        breakdown["grader_breakdown"] = grader_breakdown
        final_score += grader_score
        feedback_parts.append(grader_feedback)

        if grader_score >= 0.5:
            eff_bonus = _efficiency_bonus(step_count, MAX_STEPS)
            if eff_bonus > 0:
                final_score += eff_bonus
                breakdown["efficiency_bonus"] = round(eff_bonus, 4)
                feedback_parts.append(f"Efficiency bonus +{eff_bonus}.")

    elif is_terminal and action_type_enum == ActionType.SUBMIT_REPORT:
        # Round 2 terminal: compute from DB performance
        if baseline_score > 0 and current_score > 0:
            perf_improvement = (current_score - baseline_score) / max(1.0, 100.0 - baseline_score)
            step_efficiency  = 1.0 - (step_count / max(1, MAX_STEPS))
            terminal_score   = round(
                (perf_improvement * 0.60) + (step_efficiency * 0.20) + 0.10, 4
            )
            terminal_score = max(0.001, min(0.999, terminal_score))
            breakdown["terminal_score"]    = terminal_score
            breakdown["perf_improvement"]  = round(perf_improvement, 4)
            breakdown["step_efficiency"]   = round(step_efficiency, 4)
            final_score += terminal_score
            feedback_parts.append(
                f"Report submitted. Performance: {baseline_score:.1f} → {current_score:.1f}. "
                f"Terminal score: {terminal_score}."
            )
            # Efficiency bonus on submit_report too
            eff_bonus = _efficiency_bonus(step_count, MAX_STEPS)
            if eff_bonus > 0:
                final_score += eff_bonus
                breakdown["efficiency_bonus"] = round(eff_bonus, 4)
                feedback_parts.append(f"Efficiency bonus +{eff_bonus}.")
        else:
            breakdown["terminal_score"] = 0.10
            final_score += 0.10
            feedback_parts.append("Report submitted.")

    elif action_type_enum == ActionType.PROPOSE_FIX:
        raw_score, grader_breakdown, _ = grade(action, task_id)
        partial = round(raw_score * 0.4, 4)
        breakdown["partial_grader_score"] = partial
        final_score += partial

    elif action_type_enum == ActionType.IDENTIFY_ERROR:
        raw_score, _, _ = grade(action, task_id)
        partial = round(raw_score * 0.2, 4)
        breakdown["identification_score"] = partial
        final_score += partial

    # ── 5. Loop penalty ───────────────────────────────────────────
    if _detect_loop(previous_actions, action_type_val):
        consecutive = _count_consecutive(previous_actions, action_type_val)
        loop_pen    = LOOP_PENALTY * min(consecutive - 1, 3)
        final_score += loop_pen
        breakdown["loop_penalty"] = round(loop_pen, 4)
        feedback_parts.append(f"Loop detected ({consecutive}x). Penalty {loop_pen}.")

    # ── 6. Hint penalty ───────────────────────────────────────────
    if action_type_enum == ActionType.REQUEST_HINT:
        final_score += HINT_PENALTY
        breakdown["hint_penalty"] = HINT_PENALTY
        feedback_parts.append(f"Hint requested. Penalty {HINT_PENALTY}.")

    # ── 7. Backtrack penalty ──────────────────────────────────────
    if db_delta < -1.0:
        final_score += BACKTRACK_PENALTY
        breakdown["backtrack_penalty"] = BACKTRACK_PENALTY
        feedback_parts.append(f"Performance regressed. Backtrack penalty {BACKTRACK_PENALTY}.")

    # ── 8. Budget exhaustion penalty ─────────────────────────────
    if step_count >= MAX_STEPS - 2 and not is_terminal:
        final_score += BUDGET_EXHAUSTION_PEN
        breakdown["budget_penalty"] = BUDGET_EXHAUSTION_PEN
        feedback_parts.append("Budget nearly exhausted. Submit report now!")

    # ── Clamp to (0.001, 0.999) ───────────────────────────────────
    final_score = round(max(0.001, min(0.999, final_score)), 4)
    breakdown["total"] = final_score

    feedback = " ".join(feedback_parts) if feedback_parts else "Step processed."

    return Reward(score=final_score, breakdown=breakdown, feedback=feedback)


# ─────────────────────────────────────────────
#  EPISODE DONE CONDITION
# ─────────────────────────────────────────────

def is_done(
    action_type:  ActionType,
    step_count:   int,
    grader_score: float = 0.0,
    target_reached: bool = False,
) -> bool:
    """
    Episode ends when:
    1. Agent submits report / final answer
    2. Max steps reached
    3. Perfect score / target reached
    """
    if action_type in TERMINAL_ACTIONS:
        return True
    if step_count >= MAX_STEPS:
        return True
    if grader_score >= 1.0:
        return True
    if target_reached:
        return True
    return False
