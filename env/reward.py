from env.models import Action, Reward, DifficultyLevel, ActionType
from env.graders import grade

#  CONSTANTS

MAX_STEPS        = 20
HINT_PENALTY     = -0.05   # Per hint requested
LOOP_PENALTY     = -0.05   # Same action 3+ times in a row
INVALID_PENALTY  = -0.10   # Null / malformed action
STEP_EFFICIENCY_BONUS = 0.10  # Bonus for solving in fewer steps than estimated

# Dense reward per action type (before grader score)
STEP_REWARDS = {
    ActionType.IDENTIFY_ERROR:  0.15,  # Rewarded for diagnosing
    ActionType.PROPOSE_FIX:     0.25,  # Rewarded for attempting fix
    ActionType.SUBMIT_ANSWER:   0.00,  # Final score comes from grader
    ActionType.REQUEST_HINT:    0.00,  # No reward, only penalty
    ActionType.EXPLAIN_ISSUE:   0.10,  # Rewarded for explaining
    ActionType.OPTIMIZE_QUERY:  0.20,  # Rewarded for optimization attempt
}


#  LOOP DETECTOR


def _detect_loop(previous_actions: list[str], current_action: str) -> bool:
    """
    Returns True if the agent has submitted the same action type
    3 or more times in a row — indicating a stuck loop.
    """
    if len(previous_actions) < 2:
        return False
    last_two = previous_actions[-2:]
    return all(a == current_action for a in last_two)


def _count_consecutive(previous_actions: list[str], current_action: str) -> int:
    """Count how many times the current action has been repeated consecutively."""
    count = 1
    for a in reversed(previous_actions):
        if a == current_action:
            count += 1
        else:
            break
    return count


#  EFFICIENCY BONUS


def _efficiency_bonus(step_count: int, estimated_steps: int) -> float:
    """
    Bonus reward if agent solves faster than estimated.
    Encourages efficient reasoning, not just correct answers.
    """
    if step_count <= 0 or estimated_steps <= 0:
        return 0.0
    if step_count <= estimated_steps:
        ratio = step_count / estimated_steps
        # More bonus the faster — scales from 0.10 down to 0.0
        return round(STEP_EFFICIENCY_BONUS * (1.0 - ratio + 0.1), 4)
    return 0.0


#  MAIN REWARD FUNCTION

def compute_reward(
    action:           Action,
    task_id:          str,
    difficulty:       DifficultyLevel,
    step_count:       int,
    previous_actions: list[str],
    hints_used:       int,
    estimated_steps:  int,
    action_counts:    dict[str, int],
) -> Reward:
    """
    Computes a DENSE reward signal for every step.
    Never returns 0.0 for all steps — reward varies at each step.

    Dense reward components:
    1. Step reward     — small reward just for taking valid action
    2. Grader score    — full grader score on submit_answer / optimize_query
    3. Loop penalty    — repeated same action 3+ times
    4. Hint penalty    — accumulated hint cost
    5. Efficiency bonus — solved faster than estimated steps
    6. Invalid penalty — null / malformed action

    Score is always clamped to [-1.0, 1.0].
    """

    breakdown     = {}
    feedback_parts = []
    final_score   = 0.0

    # ── Edge case: null action ────────────────────────────────────
    if action is None or action.payload is None:
        return Reward(
            score=-0.1,
            breakdown={"invalid_action": -0.1},
            feedback="Invalid or null action received. Penalty applied."
        )

    action_type_val = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
    action_type_enum = action.action_type

    # ── 1. Step reward (dense signal) ────────────────────────────
    step_reward = STEP_REWARDS.get(action_type_enum, 0.05)
    breakdown["step_reward"] = round(step_reward, 4)
    final_score += step_reward
    if step_reward > 0:
        feedback_parts.append(f"Action '{action_type_val}' rewarded +{step_reward}.")

    # ── 2. Grader score for terminal actions ──────────────────────
    grader_score = 0.0
    is_terminal  = action_type_enum in (ActionType.SUBMIT_ANSWER, ActionType.OPTIMIZE_QUERY)

    if is_terminal:
        raw_score, grader_breakdown, grader_feedback = grade(action, task_id)
        grader_score = raw_score
        breakdown["grader_score"]    = round(grader_score, 4)
        breakdown["grader_breakdown"] = grader_breakdown
        final_score += grader_score
        feedback_parts.append(grader_feedback)

        # Efficiency bonus — only on correct terminal action
        if grader_score >= 0.5:
            eff_bonus = _efficiency_bonus(step_count, estimated_steps)
            if eff_bonus > 0:
                final_score += eff_bonus
                breakdown["efficiency_bonus"] = round(eff_bonus, 4)
                feedback_parts.append(f"Efficiency bonus +{eff_bonus} for solving in {step_count} steps.")

    elif action_type_enum == ActionType.PROPOSE_FIX:
        # Partial grader score for propose_fix — encourages iterative improvement
        raw_score, grader_breakdown, _ = grade(action, task_id)
        partial = round(raw_score * 0.4, 4)  # 40% of full grader score
        grader_score = partial
        breakdown["partial_grader_score"] = partial
        final_score += partial
        if partial > 0:
            feedback_parts.append(f"Partial fix credit +{partial}.")

    elif action_type_enum == ActionType.IDENTIFY_ERROR:
        # Small grader check on error identification
        raw_score, _, _ = grade(action, task_id)
        partial = round(raw_score * 0.2, 4)  # 20% for identification step
        breakdown["identification_score"] = partial
        final_score += partial

    # ── 3. Loop penalty ───────────────────────────────────────────
    if _detect_loop(previous_actions, action_type_val):
        consecutive = _count_consecutive(previous_actions, action_type_val)
        loop_pen    = LOOP_PENALTY * min(consecutive - 2, 3)  # Cap at 3x penalty
        final_score += loop_pen
        breakdown["loop_penalty"] = round(loop_pen, 4)
        feedback_parts.append(f"Loop detected ({consecutive}x same action). Penalty {loop_pen}.")

    # ── 4. Hint penalty ───────────────────────────────────────────
    if action_type_enum == ActionType.REQUEST_HINT:
        hint_pen     = HINT_PENALTY
        final_score += hint_pen
        breakdown["hint_penalty"] = round(hint_pen, 4)
        feedback_parts.append(f"Hint requested. Penalty {hint_pen}.")

    # ── 5. Max steps penalty ──────────────────────────────────────
    if step_count >= MAX_STEPS - 1:
        final_score += -0.10
        breakdown["max_steps_penalty"] = -0.10
        feedback_parts.append("Approaching max steps limit. Penalty applied.")

    # ── Clamp to [-1.0, 1.0] ─────────────────────────────────────
    final_score = round(max(-1.0, min(1.0, final_score)), 4)
    breakdown["total"] = final_score

    feedback = " ".join(feedback_parts) if feedback_parts else "Step processed."

    return Reward(
        score=final_score,
        breakdown=breakdown,
        feedback=feedback
    )


#  EPISODE DONE CONDITION

def is_done(
    action_type:      ActionType,
    step_count:       int,
    grader_score:     float = 0.0,
) -> bool:
    """
    Episode ends when:
    1. Agent submits final answer (submit_answer / optimize_query)
    2. Max steps reached
    3. Perfect score achieved
    """
    if action_type in (ActionType.SUBMIT_ANSWER, ActionType.OPTIMIZE_QUERY):
        return True
    if step_count >= MAX_STEPS:
        return True
    if grader_score >= 1.0:
        return True
    return False