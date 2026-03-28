import os
import time
from env.environment import SQLDebuggerEnvironment
from env.models import (
    Action, ActionType, DifficultyLevel,
    BaselineResponse, BaselineResult
)


#  BASELINE AGENT
#  Uses rule-based heuristics — no GPT-4
#  Must complete within 60 seconds
#  OPENAI_API_KEY must come from environment

def _check_api_key():
    """Edge case: OPENAI_API_KEY not set → raise clear error."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running baseline: "
            "set OPENAI_API_KEY=your-key-here"
        )
    return key


def _rule_based_agent(env: SQLDebuggerEnvironment, task: dict) -> tuple[float, int, str]:
    """
    Rule-based baseline agent that analyzes the buggy query
    and attempts a fix using heuristics.
    Fast — no API calls needed for baseline scoring.
    """
    context     = task.get("current_context", {})
    buggy_query = context.get("buggy_query", "")
    error_msg   = context.get("error_message", "")
    error_type  = context.get("error_type_hint", "syntax")
    category    = context.get("category", "syntax")

    steps_taken = 0
    total_reward = 0.0

    # ── Step 1: Identify the error ────────────────────────────────
    identify_payload = {
        "error_location": _guess_error_location(buggy_query, error_msg, category),
        "error_type":     error_type,
        "explanation":    f"Detected {category} issue in query: {error_msg[:100]}"
    }
    action1 = Action(
        action_type=ActionType.IDENTIFY_ERROR,
        payload=identify_payload
    )
    resp1 = env.step(action1)
    total_reward += resp1.reward.score
    steps_taken  += 1

    if resp1.done:
        return total_reward, steps_taken, resp1.reward.feedback

    # ── Step 2: Submit answer based on heuristic fix ──────────────
    fixed_query  = _apply_heuristic_fix(buggy_query, category, error_msg)
    explanation  = _generate_explanation(buggy_query, fixed_query, category)

    if category == "performance":
        action2 = Action(
            action_type=ActionType.OPTIMIZE_QUERY,
            payload={
                "optimized_query":     fixed_query,
                "optimization_type":   f"Fix {category} issue: {error_type}",
                "explanation":         explanation,
                "root_cause":          f"Performance issue detected: {error_msg[:100]}",
                "expected_improvement":"Significant reduction in query execution time",
                "confidence":          0.6
            }
        )
    else:
        action2 = Action(
            action_type=ActionType.SUBMIT_ANSWER,
            payload={
                "fixed_query":   fixed_query,
                "explanation":   explanation,
                "error_type":    error_type,
                "error_location": identify_payload["error_location"],
                "confidence":    0.6
            }
        )

    resp2 = env.step(action2)
    total_reward += resp2.reward.score
    steps_taken  += 1

    return total_reward, steps_taken, resp2.reward.feedback


def _guess_error_location(query: str, error_msg: str, category: str) -> str:
    """Heuristic: guess where the error is based on keywords."""
    q = query.upper()
    e = error_msg.upper()

    if "SELECT" in e or "COLUMN" in e:
        return "SELECT clause"
    if "WHERE" in e or "FILTER" in e:
        return "WHERE clause"
    if "JOIN" in e or "ON" in e:
        return "JOIN condition"
    if "GROUP" in e or "HAVING" in e:
        return "GROUP BY / HAVING clause"
    if "ORDER" in e:
        return "ORDER BY clause"
    if category == "performance":
        return "Query structure — performance bottleneck"
    return "Unknown location"


def _apply_heuristic_fix(query: str, category: str, error_msg: str) -> str:
    """
    Apply simple heuristic fixes based on category.
    Not perfect — baseline is meant to score low-medium,
    showing the environment has room for agent improvement.
    """
    q = query.strip()

    if category == "syntax":
        # Fix missing commas in SELECT
        if "syntax error" in error_msg.lower() and "name" in error_msg.lower():
            import re
            q = re.sub(r"SELECT\s+(\w+)\s+(\w+)", r"SELECT \1, \2", q, flags=re.IGNORECASE)

        # Fix missing WHERE
        if "WHERE" not in q.upper() and "=" in q:
            q = q.replace(" id =", " WHERE id =")
            q = q.replace(" name =", " WHERE name =")

        # Fix unclosed string
        if q.count("'") % 2 != 0:
            q = q + "'"

        # Fix ORDER → ORDER BY
        import re
        q = re.sub(r"\bORDER\s+(?!BY)(\w)", r"ORDER BY \1", q, flags=re.IGNORECASE)

        # Fix GROUP → GROUP BY
        q = re.sub(r"\bGROUP\s+(?!BY)(\w)", r"GROUP BY \1", q, flags=re.IGNORECASE)

    elif category == "logic":
        # Fix INNER JOIN → LEFT JOIN for inclusion
        if "INNER JOIN" in q.upper():
            q = q.replace("INNER JOIN", "LEFT JOIN").replace("inner join", "LEFT JOIN")

        # Fix WHERE aggregate → HAVING
        import re
        having_pattern = re.compile(
            r"WHERE\s+(AVG|SUM|COUNT|MAX|MIN)\s*\(", re.IGNORECASE
        )
        if having_pattern.search(q):
            # Move aggregate condition to HAVING
            q = having_pattern.sub("HAVING \\1(", q)

    elif category == "performance":
        # For performance issues, suggest JOIN-based rewrite
        if "SELECT *" in q.upper():
            q = q.replace("SELECT *", "SELECT id, name, status, created_at")

    return q


def _generate_explanation(buggy: str, fixed: str, category: str) -> str:
    """Generate a human-readable explanation of the fix."""
    if buggy.strip() == fixed.strip():
        return f"Analyzed the {category} issue. The query may require deeper inspection."

    explanations = {
        "syntax":      "Fixed syntax error in the SQL query by correcting the query structure.",
        "logic":       "Fixed logic error by correcting the JOIN type and query conditions.",
        "performance": "Optimized query performance by restructuring to avoid expensive operations.",
    }
    base = explanations.get(category, "Applied heuristic fix to the SQL query.")
    return f"{base} Original: '{buggy[:60]}...' Fixed: '{fixed[:60]}...'"


# ─────────────────────────────────────────────
#  MAIN BASELINE RUNNER
# ─────────────────────────────────────────────

def run_baseline() -> BaselineResponse:
    """
    Runs baseline agent against one task of each difficulty.
    Returns BaselineResponse with scores for all 3 tasks.
    Must complete within 60 seconds.
    """
    # Check API key exists (even if rule-based agent doesn't use it,
    # the spec requires it to be validated)
    try:
        _check_api_key()
    except ValueError as e:
        print(f"Warning: {e}")
        # Continue with rule-based agent anyway for demo

    results     = []
    difficulties = [
        (DifficultyLevel.EASY,   "easy_001"),
        (DifficultyLevel.MEDIUM, "medium_001"),
        (DifficultyLevel.HARD,   "hard_001"),
    ]

    for difficulty, task_id in difficulties:
        env = SQLDebuggerEnvironment()
        try:
            obs          = env.reset(difficulty=difficulty.value, task_id=task_id)
            task_context = {"current_context": obs.current_context}

            start        = time.time()
            score, steps, feedback = _rule_based_agent(env, task_context)
            elapsed      = time.time() - start

            results.append(BaselineResult(
                task_id    = task_id,
                difficulty = difficulty,
                score      = round(score, 4),
                steps      = steps,
                feedback   = f"{feedback} (elapsed: {elapsed:.2f}s)"
            ))
            print(f"Baseline {difficulty.value}: score={round(score,4)}, steps={steps}")

        except Exception as e:
            results.append(BaselineResult(
                task_id    = task_id,
                difficulty = difficulty,
                score      = 0.0,
                steps      = 0,
                feedback   = f"Error: {str(e)}"
            ))

    avg = round(sum(r.score for r in results) / len(results), 4) if results else 0.0
    print(f"Baseline average score: {avg}")

    return BaselineResponse(results=results, average_score=avg)


# ─────────────────────────────────────────────
#  DIRECT RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running baseline agent...")
    response = run_baseline()
    print(f"\nFinal Results:")
    for r in response.results:
        print(f"  {r.difficulty.value:8} | {r.task_id:12} | score={r.score} | steps={r.steps}")
    print(f"\nAverage Score: {response.average_score}")