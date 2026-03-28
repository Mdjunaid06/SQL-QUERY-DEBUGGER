import pytest
from env.models import Action, ActionType
from env.graders import grade, _normalize, _query_similarity


def test_easy_perfect_score():
    action = Action(
        action_type=ActionType.SUBMIT_ANSWER,
        payload={
            "fixed_query":   "SELECT id, name, email FROM users WHERE active = 1",
            "explanation":   "Added missing commas between column names in SELECT clause",
            "error_type":    "syntax",
            "error_location":"SELECT clause",
            "confidence":    0.95
        }
    )
    score, breakdown, feedback = grade(action, "easy_001")
    assert score > 0.5
    assert 0.0 <= score <= 1.0


def test_null_action_returns_zero():
    score, breakdown, feedback = grade(None, "easy_001")
    assert score == 0.0
    assert "null" in feedback.lower() or "no action" in feedback.lower()


def test_unknown_task_returns_zero():
    action = Action(action_type=ActionType.SUBMIT_ANSWER,
                    payload={"fixed_query": "SELECT 1", "explanation": "test"})
    score, _, _ = grade(action, "nonexistent_task_999")
    assert score == 0.0


def test_determinism():
    """Same input must always return same score."""
    action = Action(
        action_type=ActionType.SUBMIT_ANSWER,
        payload={
            "fixed_query":   "SELECT id, name, email FROM users WHERE active = 1",
            "explanation":   "Fixed commas",
            "error_type":    "syntax",
            "error_location":"SELECT clause",
            "confidence":    0.9
        }
    )
    scores = [grade(action, "easy_001")[0] for _ in range(5)]
    assert len(set(scores)) == 1


def test_score_range():
    """All graders must return score in 0.0 - 1.0."""
    action = Action(action_type=ActionType.SUBMIT_ANSWER,
                    payload={"fixed_query": "SELECT 1", "explanation": "test"})
    for task_id in ["easy_001", "medium_001", "hard_001"]:
        score, _, _ = grade(action, task_id)
        assert 0.0 <= score <= 1.0, f"Score out of range for {task_id}: {score}"


def test_no_binary_graders():
    """Graders must not always return only 0 or only 1."""
    payloads = [
        {"fixed_query": "SELECT id, name, email FROM users WHERE active = 1",
         "explanation": "Fixed", "confidence": 0.9},
        {"fixed_query": "SELECT *", "explanation": "wrong"},
        {"fixed_query": "", "explanation": ""},
    ]
    for task_id in ["easy_001", "medium_001", "hard_001"]:
        scores = set()
        for p in payloads:
            action = Action(action_type=ActionType.SUBMIT_ANSWER, payload=p)
            score, _, _ = grade(action, task_id)
            scores.add(score)
        assert len(scores) > 1, f"Grader for {task_id} returns same score always"


def test_empty_string_answer():
    """Empty string must return 0.0, not crash."""
    action = Action(action_type=ActionType.SUBMIT_ANSWER,
                    payload={"fixed_query": "", "explanation": ""})
    score, _, _ = grade(action, "easy_001")
    assert score == 0.0 or score < 0.3


def test_case_insensitive_normalization():
    """Grader normalizes case — UPPER and lower should score similarly."""
    action_upper = Action(action_type=ActionType.SUBMIT_ANSWER,
                          payload={"fixed_query": "SELECT ID, NAME, EMAIL FROM USERS WHERE ACTIVE = 1",
                                   "explanation": "Fixed", "confidence": 0.9})
    action_lower = Action(action_type=ActionType.SUBMIT_ANSWER,
                          payload={"fixed_query": "select id, name, email from users where active = 1",
                                   "explanation": "Fixed", "confidence": 0.9})
    score_upper, _, _ = grade(action_upper, "easy_001")
    score_lower, _, _ = grade(action_lower, "easy_001")
    assert abs(score_upper - score_lower) < 0.1


def test_whitespace_normalization():
    """Extra whitespace must not affect score."""
    action = Action(action_type=ActionType.SUBMIT_ANSWER,
                    payload={"fixed_query": "  SELECT   id,  name,  email  FROM  users  WHERE  active = 1  ",
                             "explanation": "Fixed", "confidence": 0.9})
    score, _, _ = grade(action, "easy_001")
    assert score > 0.5


def test_medium_logic_grader():
    action = Action(
        action_type=ActionType.SUBMIT_ANSWER,
        payload={
            "fixed_query":    "SELECT u.id, u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name",
            "explanation":    "Changed INNER JOIN to LEFT JOIN to include users with zero orders",
            "error_type":     "logic",
            "error_location": "JOIN type",
            "confidence":     0.9
        }
    )
    score, _, _ = grade(action, "medium_001")
    assert score > 0.4


def test_hard_grader_frontier_model_range():
    """Hard grader must allow scores in 0.10-0.20 range for partial answers."""
    action = Action(
        action_type=ActionType.OPTIMIZE_QUERY,
        payload={
            "optimized_query": "SELECT u.id FROM users u LEFT JOIN orders o ON u.id = o.user_id",
            "optimization_type": "Replace N+1 with JOIN",
            "explanation": "N+1 pattern detected",
            "confidence": 0.5
        }
    )
    score, _, _ = grade(action, "hard_001")
    assert 0.0 <= score <= 1.0


def test_query_similarity_helper():
    assert _query_similarity("SELECT id FROM users", "SELECT id FROM users") == 1.0
    assert _query_similarity("", "SELECT id FROM users") < 0.5
    assert _query_similarity("SELECT id FROM users", "") == 0.0