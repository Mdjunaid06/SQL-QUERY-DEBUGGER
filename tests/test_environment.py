import pytest
from env.environment import SQLDebuggerEnvironment
from env.models import Action, ActionType, DifficultyLevel


@pytest.fixture
def env():
    e = SQLDebuggerEnvironment()
    return e


def test_state_before_reset(env):
    """state() before reset must not crash — returns default state."""
    s = env.state()
    assert s.initialized == False
    assert s.step_count == 0


def test_reset_easy(env):
    obs = env.reset(difficulty="easy")
    assert obs.task_id.startswith("easy")
    assert obs.step_count == 0
    assert obs.difficulty == DifficultyLevel.EASY
    assert "fixed_query" not in obs.current_context
    assert "buggy_query" in obs.current_context


def test_reset_medium(env):
    obs = env.reset(difficulty="medium")
    assert obs.task_id.startswith("medium")


def test_reset_hard(env):
    obs = env.reset(difficulty="hard")
    assert obs.task_id.startswith("hard")


def test_reset_clears_state(env):
    """Reset mid-episode must clear all state — no leakage."""
    env.reset(difficulty="easy")
    action = Action(action_type=ActionType.IDENTIFY_ERROR,
                    payload={"error_location": "SELECT", "error_type": "syntax"})
    env.step(action)
    assert env.state().step_count == 1

    # Reset mid-episode
    env.reset(difficulty="medium")
    assert env.state().step_count == 0
    assert env.state().total_reward == 0.0
    assert env.state().previous_actions == []


def test_step_identify_error(env):
    env.reset(difficulty="easy")
    action = Action(action_type=ActionType.IDENTIFY_ERROR,
                    payload={"error_location": "SELECT clause", "error_type": "syntax",
                             "explanation": "Missing commas"})
    resp = env.step(action)
    assert resp.reward.score > 0
    assert resp.done == False
    assert resp.observation.step_count == 1


def test_step_null_action(env):
    """Null action must return -0.1, never crash."""
    env.reset(difficulty="easy")
    resp = env.step(None)
    assert resp.reward.score == -0.1
    assert resp.done == False


def test_step_after_done(env):
    """Step after done must not crash."""
    env.reset(difficulty="easy", task_id="easy_001")
    action = Action(action_type=ActionType.SUBMIT_ANSWER,
                    payload={"fixed_query": "SELECT id, name, email FROM users WHERE active = 1",
                             "explanation": "Fixed", "confidence": 0.9})
    env.step(action)
    assert env.state().done == True

    # Step again after done
    resp = env.step(action)
    assert resp.done == True
    assert "Call reset()" in resp.reward.feedback


def test_dense_reward(env):
    """Reward must vary at each step — not only at end."""
    env.reset(difficulty="easy")
    rewards = []
    actions = [
        Action(action_type=ActionType.IDENTIFY_ERROR,
               payload={"error_location": "SELECT", "error_type": "syntax"}),
        Action(action_type=ActionType.EXPLAIN_ISSUE,
               payload={"explanation": "Missing commas between column names in SELECT"}),
    ]
    for a in actions:
        r = env.step(a)
        rewards.append(r.reward.score)
        if r.done:
            break

    # Rewards must not all be zero
    assert any(r != 0.0 for r in rewards)


def test_max_steps(env):
    """Episode must terminate at max_steps."""
    env.reset(difficulty="easy")
    action = Action(action_type=ActionType.IDENTIFY_ERROR,
                    payload={"error_location": "x", "error_type": "syntax"})
    done = False
    for _ in range(25):
        resp = env.step(action)
        if resp.done:
            done = True
            break
    assert done == True


def test_hint_injected_in_context(env):
    """Hint must appear in next observation after request_hint."""
    env.reset(difficulty="easy")
    action = Action(action_type=ActionType.REQUEST_HINT,
                    payload={"hint_type": "location"})
    resp = env.step(action)
    assert "last_hint" in resp.observation.current_context


def test_state_reflects_latest_step(env):
    """state() must always reflect the latest step accurately."""
    env.reset(difficulty="easy")
    action = Action(action_type=ActionType.IDENTIFY_ERROR,
                    payload={"error_location": "SELECT", "error_type": "syntax"})
    env.step(action)
    s = env.state()
    assert s.step_count == 1
    assert s.initialized == True
    assert "identify_error" in s.previous_actions