import re
from env.models import Action, DifficultyLevel
from env.tasks import task_manager

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalize SQL for comparison — lowercase, strip whitespace, collapse spaces."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def _safe_get(payload: dict, key: str, default=None):
    """Safe dict access — never KeyError."""
    if not isinstance(payload, dict):
        return default
    return payload.get(key, default)

def _score_explanation(explanation: str) -> float:
    """Score explanation quality by length and keyword richness."""
    if not explanation or not isinstance(explanation, str):
        return 0.0
    explanation = explanation.strip()
    if len(explanation) < 10:
        return 0.0
    if len(explanation) < 30:
        return 0.05
    if len(explanation) < 80:
        return 0.10
    return 0.15

def _score_confidence(confidence) -> float:
    """Give partial credit for providing a valid confidence score."""
    try:
        c = float(confidence)
        if 0.0 <= c <= 1.0:
            return 0.05
    except (TypeError, ValueError):
        pass
    return 0.0

def _query_similarity(submitted: str, expected: str) -> float:
    """
    Multi-level SQL similarity check.
    Returns 0.0 - 1.0 based on how close the submitted query is to expected.
    """
    s = _normalize(submitted)
    e = _normalize(expected)

    if s == e:
        return 1.0

    s_tokens = set(s.split())
    e_tokens = set(e.split())

    if not e_tokens:
        return 0.0

    overlap = len(s_tokens & e_tokens) / len(e_tokens)

    critical_keywords = _extract_critical_keywords(e)
    critical_found = sum(1 for kw in critical_keywords if kw in s)
    critical_score = critical_found / len(critical_keywords) if critical_keywords else 0.0

    return round((overlap * 0.4) + (critical_score * 0.6), 4)

def _extract_critical_keywords(query: str) -> list[str]:
    """Extract SQL keywords that are critical to correctness."""
    keywords = [
        "left join", "inner join", "right join",
        "group by", "order by", "having",
        "partition by", "coalesce", "distinct",
        "where", "on", "and", "or", "not",
        "count", "sum", "avg", "max", "min",
        "select", "from", "join"
    ]
    found = []
    q = query.lower()
    for kw in keywords:
        if kw in q:
            found.append(kw)
    return found

def _score_error_type(submitted_type: str, expected_type: str) -> float:
    """Score for correctly identifying the error type."""
    if not submitted_type:
        return 0.0
    s = submitted_type.strip().lower()
    e = expected_type.strip().lower()
    if s == e:
        return 0.10
    related = {
        "performance": ["optimization", "slow", "index", "scan"],
        "logic":       ["semantic", "incorrect", "wrong"],
        "syntax":      ["parse", "grammar", "token"]
    }
    for canonical, aliases in related.items():
        if e == canonical and any(alias in s for alias in aliases):
            return 0.05
    return 0.0

def _score_error_location(submitted_location: str, expected_location: str) -> float:
    """Score for correctly identifying WHERE in the query the error is."""
    if not submitted_location or not expected_location:
        return 0.0
    s = submitted_location.strip().lower()
    e = expected_location.strip().lower()
    if s == e:
        return 0.15
    e_words = set(e.split())
    s_words = set(s.split())
    overlap = len(e_words & s_words) / len(e_words) if e_words else 0.0
    return round(overlap * 0.10, 4)


# ─────────────────────────────────────────────
#  GRADERS PER DIFFICULTY
# ─────────────────────────────────────────────

def grade_easy(action: Action, ground_truth: dict) -> tuple[float, dict, str]:
    """
    Easy task grader — syntax errors.
    Max score: 0.999 (strictly less than 1.0)
    DETERMINISTIC: same input always returns same score.
    """
    if action is None or action.payload is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    payload   = action.payload
    score     = 0.0
    breakdown = {}
    feedback_parts = []

    # ── 1. Query fix correctness (0.50) ──────────────────────────
    submitted_query = _safe_get(payload, "fixed_query", "") or _safe_get(payload, "optimized_query", "")
    expected_query  = ground_truth.get("fixed_query", "")
    similarity      = _query_similarity(submitted_query, expected_query)

    if similarity >= 1.0:
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

    score += fix_score
    breakdown["fix_correctness"] = round(fix_score, 4)

    # ── 2. Error location (0.15) ─────────────────────────────────
    submitted_location = _safe_get(payload, "error_location", "")
    expected_location  = ground_truth.get("error_location", "")
    loc_score          = _score_error_location(str(submitted_location), expected_location)
    score             += loc_score
    breakdown["error_location"] = round(loc_score, 4)
    if loc_score > 0:
        feedback_parts.append("Correctly identified error location.")

    # ── 3. Error type (0.10) ─────────────────────────────────────
    submitted_type = _safe_get(payload, "error_type", "")
    expected_type  = ground_truth.get("error_type", "syntax")
    type_score     = _score_error_type(str(submitted_type), expected_type)
    score         += type_score
    breakdown["error_type"] = round(type_score, 4)
    if type_score > 0:
        feedback_parts.append("Correctly identified error type.")

    # ── 4. Explanation quality (0.15) ────────────────────────────
    explanation = _safe_get(payload, "explanation", "") or _safe_get(payload, "change_made", "")
    expl_score  = _score_explanation(str(explanation) if explanation else "")
    score      += expl_score
    breakdown["explanation"] = round(expl_score, 4)
    if expl_score > 0:
        feedback_parts.append("Explanation provided.")

    # ── 5. Confidence (0.05) ─────────────────────────────────────
    confidence = _safe_get(payload, "confidence", None)
    conf_score = _score_confidence(confidence)
    score     += conf_score
    breakdown["confidence"] = round(conf_score, 4)

    # Clamp strictly between 0 and 1 exclusive
    final_score = round(max(0.001, min(0.999, score)), 4)
    feedback    = " ".join(feedback_parts) if feedback_parts else "No valid response provided."
    return final_score, breakdown, feedback


def grade_medium(action: Action, ground_truth: dict) -> tuple[float, dict, str]:
    """
    Medium task grader — logic errors.
    Max score: 0.999 (strictly less than 1.0)
    DETERMINISTIC: same input always returns same score.
    """
    if action is None or action.payload is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    payload        = action.payload
    score          = 0.0
    breakdown      = {}
    feedback_parts = []

    # ── 1. Query fix correctness (0.40) ──────────────────────────
    submitted_query = _safe_get(payload, "fixed_query", "") or _safe_get(payload, "optimized_query", "")
    expected_query  = ground_truth.get("fixed_query", "")
    similarity      = _query_similarity(submitted_query, expected_query)

    if similarity >= 1.0:
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

    score += fix_score
    breakdown["fix_correctness"] = round(fix_score, 4)

    # ── 2. Logic flaw identification (0.20) ──────────────────────
    explanation = str(_safe_get(payload, "explanation", "") or _safe_get(payload, "change_made", "") or "")
    error_type  = ground_truth.get("error_type", "logic")

    logic_keywords = {
        "logic":       ["join", "left join", "inner join", "having", "where", "group by",
                        "aggregate", "subquery", "correlation", "distinct", "count"],
        "performance": ["index", "scan", "n+1", "correlated", "cartesian", "window"]
    }

    keywords_to_check = logic_keywords.get(error_type, logic_keywords["logic"])
    expl_lower        = explanation.lower()
    keyword_hits      = sum(1 for kw in keywords_to_check if kw in expl_lower)
    logic_score       = min(keyword_hits * 0.05, 0.20)
    score            += logic_score
    breakdown["logic_flaw_identification"] = round(logic_score, 4)
    if logic_score > 0:
        feedback_parts.append("Shows understanding of the logic flaw.")

    # ── 3. Error location (0.15) ─────────────────────────────────
    submitted_location = _safe_get(payload, "error_location", "")
    expected_location  = ground_truth.get("error_location", "")
    loc_score          = _score_error_location(str(submitted_location), expected_location)
    score             += loc_score
    breakdown["error_location"] = round(loc_score, 4)

    # ── 4. Explanation quality (0.15) ────────────────────────────
    expl_score = _score_explanation(explanation)
    score     += expl_score
    breakdown["explanation"] = round(expl_score, 4)

    # ── 5. Confidence (0.05) ─────────────────────────────────────
    confidence = _safe_get(payload, "confidence", None)
    conf_score = _score_confidence(confidence)
    score     += conf_score
    breakdown["confidence"] = round(conf_score, 4)

    # ── 6. Impact analysis bonus (0.05) ──────────────────────────
    impact = str(_safe_get(payload, "impact", "") or "")
    if len(impact.strip()) > 20:
        score += 0.05
        breakdown["impact_analysis"] = 0.05
        feedback_parts.append("Impact analysis provided.")
    else:
        breakdown["impact_analysis"] = 0.0

    # Clamp strictly between 0 and 1 exclusive
    final_score = round(max(0.001, min(0.999, score)), 4)
    feedback    = " ".join(feedback_parts) if feedback_parts else "No valid response provided."
    return final_score, breakdown, feedback


def grade_hard(action: Action, ground_truth: dict) -> tuple[float, dict, str]:
    """
    Hard task grader — performance issues.
    Max score: 0.999 (strictly less than 1.0)
    Frontier models expected ~0.10-0.20.
    DETERMINISTIC: same input always returns same score.
    """
    if action is None or action.payload is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    payload        = action.payload
    score          = 0.0
    breakdown      = {}
    feedback_parts = []

    # ── 1. Query correctness (0.30) ──────────────────────────────
    submitted_query = (
        _safe_get(payload, "optimized_query", "")
        or _safe_get(payload, "fixed_query", "")
        or ""
    )
    expected_query = ground_truth.get("fixed_query", "")
    similarity     = _query_similarity(submitted_query, expected_query)

    if similarity >= 1.0:
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

    score += fix_score
    breakdown["query_correctness"] = round(fix_score, 4)

    # ── 2. Performance concept identification (0.30) ──────────────
    explanation   = str(_safe_get(payload, "explanation", "") or _safe_get(payload, "change_made", "") or "")
    optimization  = str(_safe_get(payload, "optimization_type", "") or "")
    combined_text = (explanation + " " + optimization).lower()
    perf_issue    = ground_truth.get("performance_issue", {})
    issue_type    = perf_issue.get("type", "").lower()

    performance_concept_map = {
        "n+1":               ["n+1", "correlated subquery", "subquery per row", "multiple queries", "join instead"],
        "full table scan":   ["full table scan", "index not used", "function on column", "sargable", "range scan", "seek"],
        "cartesian product": ["cartesian", "cross join", "missing join condition", "implicit join", "comma join"],
        "select *":          ["select *", "over-fetch", "covering index", "column projection", "unnecessary columns"],
        "window function":   ["window function", "partition by", "row_number", "subquery filter", "where clause window"]
    }

    concept_score = 0.0
    for concept, keywords in performance_concept_map.items():
        if any(concept_part in issue_type for concept_part in concept.split()):
            hits = sum(1 for kw in keywords if kw in combined_text)
            concept_score = min(hits * 0.06, 0.30)
            break

    score += concept_score
    breakdown["performance_concept"] = round(concept_score, 4)
    if concept_score > 0:
        feedback_parts.append("Demonstrates understanding of the performance issue.")

    # ── 3. Explanation depth (0.15) ───────────────────────────────
    expl_score = _score_explanation(explanation)
    if len(explanation.strip()) > 150:
        expl_score = min(expl_score + 0.05, 0.15)
    score += expl_score
    breakdown["explanation_depth"] = round(expl_score, 4)

    # ── 4. Root cause analysis (0.10) ─────────────────────────────
    root_cause = str(_safe_get(payload, "root_cause", "") or "")
    if len(root_cause.strip()) > 30:
        score += 0.10
        breakdown["root_cause_analysis"] = 0.10
        feedback_parts.append("Root cause analysis provided.")
    else:
        breakdown["root_cause_analysis"] = 0.0

    # ── 5. Expected improvement (0.10) ────────────────────────────
    improvement = str(_safe_get(payload, "expected_improvement", "") or "")
    if len(improvement.strip()) > 20:
        score += 0.10
        breakdown["expected_improvement"] = 0.10
        feedback_parts.append("Performance improvement estimate provided.")
    else:
        breakdown["expected_improvement"] = 0.0

    # ── 6. Confidence (0.05) ──────────────────────────────────────
    confidence = _safe_get(payload, "confidence", None)
    conf_score = _score_confidence(confidence)
    score     += conf_score
    breakdown["confidence"] = round(conf_score, 4)

    # Clamp strictly between 0 and 1 exclusive
    final_score = round(max(0.001, min(0.999, score)), 4)
    feedback    = " ".join(feedback_parts) if feedback_parts else "Performance issue not identified."
    return final_score, breakdown, feedback


# ─────────────────────────────────────────────
#  MAIN GRADER DISPATCHER
# ─────────────────────────────────────────────

def grade(action: Action, task_id: str) -> tuple[float, dict, str]:
    """
    Main grader entry point.
    Looks up ground truth, dispatches to correct grader by difficulty.
    ALWAYS returns (float, dict, str) — never crashes.
    Score is always strictly between 0.001 and 0.999.
    """
    if action is None:
        return 0.001, {"error": "null_action"}, "No action provided."

    ground_truth = task_manager.get_ground_truth(task_id)
    if ground_truth is None:
        return 0.001, {"error": "unknown_task"}, f"Task '{task_id}' not found."

    difficulty = ground_truth.get("id", "").split("_")[0]

    try:
        if difficulty == "easy":
            return grade_easy(action, ground_truth)
        elif difficulty == "medium":
            return grade_medium(action, ground_truth)
        elif difficulty == "hard":
            return grade_hard(action, ground_truth)
        else:
            return 0.001, {"error": "unknown_difficulty"}, f"Unknown difficulty: {difficulty}"
    except Exception as e:
        return 0.001, {"error": str(e)}, f"Grader error: {str(e)}"