from env.environment import SQLDebuggerEnvironment
from env.models import Action, ActionType

env = SQLDebuggerEnvironment()

# Test 1: state() before reset — must not crash
s = env.state()
print(f"State before reset: initialized={s.initialized}")

# Test 2: reset()
obs = env.reset(difficulty="easy")
print(f"Reset OK: task_id={obs.task_id}, difficulty={obs.difficulty}")
print(f"Context keys: {list(obs.current_context.keys())}")
print(f"Ground truth NOT in context: {'fixed_query' not in obs.current_context}")

# Test 3: step() identify_error
action1 = Action(
    action_type=ActionType.IDENTIFY_ERROR,
    payload={"error_location": "SELECT clause", "error_type": "syntax", "explanation": "Missing commas"}
)
resp1 = env.step(action1)
print(f"Step 1: reward={resp1.reward.score}, done={resp1.done}, step={resp1.observation.step_count}")

# Test 4: step() request_hint
action2 = Action(action_type=ActionType.REQUEST_HINT, payload={"hint_type": "location"})
resp2 = env.step(action2)
print(f"Step 2 hint: reward={resp2.reward.score}, hints_used={resp2.observation.hints_used}")
print(f"Hint in context: {'last_hint' in resp2.observation.current_context}")

# Test 5: step() submit_answer
obs = env.reset(difficulty="easy", task_id="easy_001")
action3 = Action(
    action_type=ActionType.SUBMIT_ANSWER,
    payload={
        "fixed_query": "SELECT id, name, email FROM users WHERE active = 1",
        "explanation": "Added missing commas between column names in SELECT clause",
        "error_type": "syntax",
        "error_location": "SELECT clause",
        "confidence": 0.95
    }
)
resp3 = env.step(action3)
print(f"Submit answer: reward={resp3.reward.score}, done={resp3.done}")

# Test 6: step after done — must not crash
resp4 = env.step(action3)
print(f"Step after done: done={resp4.done}, feedback='{resp4.reward.feedback}'")

# Test 7: null action
obs = env.reset(difficulty="easy")
resp5 = env.step(None)
print(f"Null action: reward={resp5.reward.score}, done={resp5.done}")

# Test 8: reset mid-episode clears state
obs = env.reset(difficulty="medium")
print(f"Mid-episode reset: new task={obs.task_id}, step_count={obs.step_count}")

# Test 9: full episode 10 steps
obs = env.reset(difficulty="hard")
print(f"Hard episode started: {obs.task_id}")
actions = [
    Action(action_type=ActionType.IDENTIFY_ERROR, payload={"error_location": "SELECT clause", "error_type": "performance"}),
    Action(action_type=ActionType.EXPLAIN_ISSUE, payload={"explanation": "N+1 correlated subqueries cause multiple DB hits per row", "impact": "O(n) queries", "root_cause": "Subquery per user"}),
    Action(action_type=ActionType.OPTIMIZE_QUERY, payload={
        "optimized_query": "SELECT u.id, u.name, COUNT(o.id) as order_count, COALESCE(SUM(o.total), 0) as total_spent FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name",
        "optimization_type": "Replace N+1 correlated subqueries with LEFT JOIN aggregation",
        "explanation": "Single query replaces N+1 pattern",
        "root_cause": "Correlated subqueries in SELECT",
        "expected_improvement": "99% reduction in DB round trips",
        "confidence": 0.9
    }),
]
total = 0.0
for i, a in enumerate(actions):
    r = env.step(a)
    total += r.reward.score
    print(f"  Hard step {i+1}: reward={r.reward.score}, done={r.done}")
print(f"Hard episode total reward: {round(total,4)}")

print("environment.py OK")