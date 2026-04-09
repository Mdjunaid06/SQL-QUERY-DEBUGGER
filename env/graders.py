import re
import math
from env.models import Action, DifficultyLevel
from env.tasks import task_manager

# ─────────────────────────────────────────────
#  SCORE BOUNDS  (strictly between 0 and 1)
# ─────────────────────────────────────────────
SCORE_MIN = 0.001   # 0 < SCORE_MIN < 1
SCORE_MAX = 0.999   # 0 < SCORE_MAX < 1


def _clamp(value) -> float:
    """
    Guarantee the returned float is strictly inside (0, 1).
    Handles NaN, Inf, None, strings, and any numeric type safely.
    The round() call is applied AFTER the clamp, never before.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SCORE_MIN

    # Guard against NaN and ±Inf before any comparison
    if not math.isfinite(v):
        return SCORE_MIN

    clamped = max(min(v, SCORE_MAX), SCORE_MIN)
    return round(clamped, 4)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _safe_get(payload: dict, key: str, default=None):
    if not isinstance(payload, dict):
        return default
    return payload.get(key, default)


def _score_explanation(explanation: str) -> float:
    if not explanation or not isinstance(explanation, str):
        return SCORE_MIN
    explanation = explanation.strip()
    if len(explanation) < 10:
        return SCORE_MIN
    if len(explanation) < 30:
        return 0.05
    if len(explanation) < 80:
        return 0.10
    return 0.15


def _score_confidence(confidence) -> float:
    try:
        c = float(confidence)
        if math.isfinite(c) and 0.0 <= c <= 1.0:
            return 0.05
    except (TypeError, ValueError):
        pass
    return SCORE_MIN


def _query_similarity(submitted: str, expected: str) -> float:
    s = _normalize(submitted)
    e = _normalize(expected)

    if s == e:
        # Exact match — return SCORE_MAX, NOT 1.0
        return SCORE_MAX

    s_tokens = set(s.split())
    e_tokens = set(e.split())

    if not e_tokens:
        return SCORE_MIN

    overlap = len(s_tokens & e_tokens) / len(e_tokens)

    critical_keywords = _extract_critical_keywords(e)
    critical_found = sum(1 for kw in critical_keywords if kw in s)
    critical_score = (critical_found / len(critical_keywords)
                      if critical_keywords else 0.0)

    raw = (overlap * 0.4) + (critical_score * 0.6)
    return _clamp(raw)


def _extract_critical_keywords(query: str) -> list:
    keywords = [
        "left join", "inner join", "right join",
        "group by", "order by", "having",
        "partition by", "coalesce", "distinct",
        "where", "on", "and", "or", "not",
        "count", "sum", "avg", "max", "min",
        "select", "from", "join",
    ]
    q = query.lower()
    return [kw for kw in keywords if kw in q]


def _score_error_type(submitted_type: str, expected_type: str) -> float:
    if not submitted_type:
        return SCORE_MIN
    s = submitted_type.strip().lower()
    e = expected_type.strip().lower()
    if s == e:
        return 0.10
    related = {
        "performance": ["optimization", "slow", "index", "scan"],
        "logic":       ["semantic", "incorrect", "wrong"],
        "syntax":      ["parse", "grammar", "token"],
    }
    for canonical, aliases in related.items():
        if e == canonical and any(alias in s for alias in aliases):
            return 0.05
    return SCORE_MIN


def _score_error_location(submitted_location: str,
                          expected_location: str) -> float:
    if not submitted_location or not expected_location:
        return SCORE_MIN
    s = submitted_location.strip().lower()
    e = expected_location.strip().lower()
    if s == e:
        return 0.15
    e_words = set(e.split())
    s_words = set(s.split())
    if not e_words:
        return SCORE_MIN
    overlap = len(e_words & s_words) / len(e_words)
    return _clamp(overlap * 0.10)


# ─────────────────────────────────────────────
#  GRADERS
# ─────────────────────────────────────────────

def grade_easy(action: Action, ground_truth: dict) -> tuple:
    """
    Easy — syntax errors.
    Scoring budget: fix(0.50) + loc(0.15) + type(0.10) + expl(0.15) + conf(0.05) = 0.95
    """
    if action is None or action.payload is None:
        return SCORE_MIN, {"error": "null_action"}, "No action provided."

    payload        = action.payload
    score          = 0.0
    breakdown      = {}
    feedback_parts = []

    # 1. Fix correctness (0.50)
    submitted_query = (_safe_get(payload, "fixed_query", "")
                       or _safe_get(payload, "optimized_query", "") or "")
    expected_query  = ground_truth.get("fixed_query", "")
    similarity      = _query_similarity(submitted_query, expected_query)

    if similarity >= SCORE_MAX:
        fix_score = 0.50
        feedback_parts.append("Correct fix applied.")
    elif similarity >= 0.75:
        fix_score = 0.30
        feedback_parts.append("Fix is mostly correct but has minor differences.")
    elif similarity >= 0.50:
        fix_score = 0.15
        feedback_parts.append("Fix is partially correct.")
    else:
        fix_score = 0.0
        feedback_parts.append("Fix is incorrect or not provided.")

    breakdown["fix_correctness"] = _clamp(fix_score)
    score += fix_score

    # 2. Error location (0.15)
    loc_score = _score_error_location(
        str(_safe_get(payload, "error_location", "") or ""),
        ground_truth.get("error_location", ""),
    )
    breakdown["error_location"] = _clamp(loc_score)
    score += loc_score
    if loc_score > SCORE_MIN:
        feedback_parts.append("Correctly identified error location.")

    # 3. Error type (0.10)
    type_score = _score_error_type(
        str(_safe_get(payload, "error_type", "") or ""),
        ground_truth.get("error_type", "syntax"),
    )
    breakdown["error_type"] = _clamp(type_score)
    score += type_score
    if type_score > SCORE_MIN:
        feedback_parts.append("Correctly identified error type.")

    # 4. Explanation quality (0.15)
    explanation = (_safe_get(payload, "explanation", "")
                   or _safe_get(payload, "change_made", "") or "")
    expl_score  = _score_explanation(str(explanation))
    breakdown["explanation"] = _clamp(expl_score)
    score += expl_score
    if expl_score > SCORE_MIN:
        feedback_parts.append("Explanation provided.")

    # 5. Confidence (0.05)
    conf_score = _score_confidence(_safe_get(payload, "confidence", None))
    breakdown["confidence"] = _clamp(conf_score)
    score += conf_score

    final_score = _clamp(score)
    feedback    = " ".join(feedback_parts) or "No valid response provided."
    return final_score, breakdown, feedback


def grade_medium(action: Action, ground_truth: dict) -> tuple:
    """
    Medium — logic errors.
    Scoring budget: fix(0.40) + logic(0.20) + loc(0.15) + expl(0.15)
                    + conf(0.05) + impact(0.05) = 1.00  -> clamped to SCORE_MAX
    """
    if action is None or action.payload is None:
        return SCORE_MIN, {"error": "null_action"}, "No action provided."

    payload        = action.payload
    score          = 0.0
    breakdown      = {}
    feedback_parts = []

    # 1. Fix correctness (0.40)
    submitted_query = (_safe_get(payload, "fixed_query", "")
                       or _safe_get(payload, "optimized_query", "") or "")
    expected_query  = ground_truth.get("fixed_query", "")
    similarity      = _query_similarity(submitted_query, expected_query)

    if similarity >= SCORE_MAX:
        fix_score = 0.40
        feedback_parts.append("Correct fix applied.")
    elif similarity >= 0.80:
        fix_score = 0.28
        feedback_parts.append("Fix is mostly correct.")
    elif similarity >= 0.60:
        fix_score = 0.16
        feedback_parts.append("Fix is partially correct.")
    elif similarity >= 0.40:
        fix_score = 0.08
        feedback_parts.append("Fix shows some understanding.")
    else:
        fix_score = 0.0
        feedback_parts.append("Fix is incorrect or missing.")

    breakdown["fix_correctness"] = _clamp(fix_score)
    score += fix_score

    # 2. Logic flaw identification (0.20)
    explanation = str(_safe_get(payload, "explanation", "")
                      or _safe_get(payload, "change_made", "") or "")
    error_type  = ground_truth.get("error_type", "logic")

    logic_keywords = {
        "logic": ["join", "left join", "inner join", "having", "where",
                  "group by", "aggregate", "subquery", "correlation",
                  "distinct", "count"],
        "performance": ["index", "scan", "n+1", "correlated",
                        "cartesian", "window"],
    }
    keywords_to_check = logic_keywords.get(error_type, logic_keywords["logic"])
    expl_lower        = explanation.lower()
    keyword_hits      = sum(1 for kw in keywords_to_check if kw in expl_lower)
    logic_score       = _clamp(min(keyword_hits * 0.05, 0.20))
    breakdown["logic_flaw_identification"] = _clamp(logic_score)
    score += logic_score
    if logic_score > SCORE_MIN:
        feedback_parts.append("Shows understanding of the logic flaw.")

    # 3. Error location (0.15)
    loc_score = _score_error_location(
        str(_safe_get(payload, "error_location", "") or ""),
        ground_truth.get("error_location", ""),
    )
    breakdown["error_location"] = _clamp(loc_score)
    score += loc_score

    # 4. Explanation quality (0.15)
    expl_score = _score_explanation(explanation)
    breakdown["explanation"] = _clamp(expl_score)
    score += expl_score

    # 5. Confidence (0.05)
    conf_score = _score_confidence(_safe_get(payload, "confidence", None))
    breakdown["confidence"] = _clamp(conf_score)
    score += conf_score

    # 6. Impact analysis bonus (0.05)
    impact = str(_safe_get(payload, "impact", "") or "")
    impact_score = 0.05 if len(impact.strip()) > 20 else 0.0
    breakdown["impact_analysis"] = _clamp(impact_score)
    score += impact_score
    if impact_score > 0:
        feedback_parts.append("Impact analysis provided.")

    final_score = _clamp(score)
    feedback    = " ".join(feedback_parts) or "No valid response provided."
    return final_score, breakdown, feedback


def grade_hard(action: Action, ground_truth: dict) -> tuple:
    """
    Hard — performance issues (N+1, missing index, cartesian, etc).
    Scoring budget: query(0.30) + concept(0.30) + expl(0.15)
                    + root(0.10) + improvement(0.10) + conf(0.05) = 1.00 -> clamped
    """
    if action is None or action.payload is None:
        return SCORE_MIN, {"error": "null_action"}, "No action provided."

    # All variables initialised before first use
    payload        = action.payload
    score          = 0.0
    breakdown      = {}
    feedback_parts = []
    _rubric        = ground_truth.get("scoring_rubric", {})  # reserved for future use

    # 1. Query correctness (0.30)
    submitted_query = (_safe_get(payload, "optimized_query", "")
                       or _safe_get(payload, "fixed_query", "") or "")
    expected_query  = ground_truth.get("fixed_query", "")
    similarity      = _query_similarity(submitted_query, expected_query)

    if similarity >= SCORE_MAX:
        fix_score = 0.30
        feedback_parts.append("Perfectly optimized query.")
    elif similarity >= 0.85:
        fix_score = 0.22
        feedback_parts.append("Query is mostly correct.")
    elif similarity >= 0.65:
        fix_score = 0.14
        feedback_parts.append("Query shows correct approach but incomplete.")
    elif similarity >= 0.40:
        fix_score = 0.07
        feedback_parts.append("Query partially addresses the issue.")
    else:
        fix_score = 0.0
        feedback_parts.append("Query does not address the performance issue.")

    breakdown["query_correctness"] = _clamp(fix_score)
    score += fix_score

    # 2. Performance concept identification (0.30)
    explanation   = str(_safe_get(payload, "explanation", "")
                        or _safe_get(payload, "change_made", "") or "")
    optimization  = str(_safe_get(payload, "optimization_type", "") or "")
    combined_text = (explanation + " " + optimization).lower()
    perf_issue    = ground_truth.get("performance_issue", {})
    issue_type    = (perf_issue.get("type", "").lower()
                     if isinstance(perf_issue, dict) else "")

    performance_concept_map = {
        "n+1":               ["n+1", "correlated subquery", "subquery per row",
                               "multiple queries", "join instead"],
        "full table scan":   ["full table scan", "index not used",
                               "function on column", "sargable",
                               "range scan", "seek"],
        "cartesian product": ["cartesian", "cross join",
                               "missing join condition",
                               "implicit join", "comma join"],
        "select *":          ["select *", "over-fetch", "covering index",
                               "column projection", "unnecessary columns"],
        "window function":   ["window function", "partition by", "row_number",
                               "subquery filter", "where clause window"],
    }

    concept_score = 0.0
    for concept, keywords in performance_concept_map.items():
        if any(part in issue_type for part in concept.split()):
            hits = sum(1 for kw in keywords if kw in combined_text)
            concept_score = min(hits * 0.06, 0.30)
            break

    breakdown["performance_concept"] = _clamp(concept_score)
    score += concept_score
    if concept_score > 0:
        feedback_parts.append("Demonstrates understanding of the performance issue.")

    # 3. Explanation depth (0.15)
    expl_score = _score_explanation(explanation)
    if len(explanation.strip()) > 150:
        expl_score = min(expl_score + 0.05, 0.15)
    breakdown["explanation_depth"] = _clamp(expl_score)
    score += expl_score

    # 4. Root cause analysis (0.10)
    root_cause = str(_safe_get(payload, "root_cause", "") or "")
    root_score = 0.10 if len(root_cause.strip()) > 30 else 0.0
    breakdown["root_cause_analysis"] = _clamp(root_score)
    score += root_score
    if root_score > 0:
        feedback_parts.append("Root cause analysis provided.")

    # 5. Expected improvement (0.10)
    improvement = str(_safe_get(payload, "expected_improvement", "") or "")
    imp_score = 0.10 if len(improvement.strip()) > 20 else 0.0
    breakdown["expected_improvement"] = _clamp(imp_score)
    score += imp_score
    if imp_score > 0:
        feedback_parts.append("Performance improvement estimate provided.")

    # 6. Confidence (0.05)
    conf_score = _score_confidence(_safe_get(payload, "confidence", None))
    breakdown["confidence"] = _clamp(conf_score)
    score += conf_score

    final_score = _clamp(score)
    feedback    = " ".join(feedback_parts) or "Performance issue not identified."
    return final_score, breakdown, feedback


# ─────────────────────────────────────────────
#  MAIN GRADER DISPATCHER
# ─────────────────────────────────────────────

def grade(action: Action, task_id: str) -> tuple:
    """
    Main entry point.  Always returns (float, dict, str) — never crashes.
    The returned float is always strictly inside (0, 1).
    """
    if action is None:
        return SCORE_MIN, {"error": "null_action"}, "No action provided."

    ground_truth = task_manager.get_ground_truth(task_id)
    if ground_truth is None:
        return SCORE_MIN, {"error": "unknown_task"}, f"Task '{task_id}' not found."

    difficulty = ground_truth.get("id", "").split("_")[0]

    try:
        if difficulty == "easy":
            result = grade_easy(action, ground_truth)
        elif difficulty == "medium":
            result = grade_medium(action, ground_truth)
        elif difficulty == "hard":
            result = grade_hard(action, ground_truth)
        else:
            return (SCORE_MIN,
                    {"error": "unknown_difficulty"},
                    f"Unknown difficulty: {difficulty}")

        # Final safety net: re-clamp the returned score and every breakdown value
        final_score, breakdown, feedback = result
        safe_score     = _clamp(final_score)
        safe_breakdown = {
            k: _clamp(v) if isinstance(v, (int, float)) else v
            for k, v in breakdown.items()
        }
        return safe_score, safe_breakdown, feedback

    except Exception as e:
        return SCORE_MIN, {"error": str(e)}, f"Grader error: {str(e)}"
