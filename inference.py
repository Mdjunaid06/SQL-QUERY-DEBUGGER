"""
inference.py — SQL Query Debugger OpenEnv
Follows the mandatory [START]/[STEP]/[END] stdout format.
Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN.
"""

import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() 

from env.environment import SQLDebuggerEnvironment
# ─────────────────────────────────────────────
#  ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_KEY = HF_TOKEN
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL debugger. You will be given a buggy SQL query and must fix it.

    You must respond with a JSON object only — no explanation outside the JSON.

    For syntax/logic errors, respond with:
    {
        "action_type": "submit_answer",
        "fixed_query": "<your fixed SQL query here>",
        "explanation": "<brief explanation of what was wrong>",
        "error_type": "<syntax|logic|performance>",
        "error_location": "<where in the query the error is>",
        "confidence": 0.9
    }

    For performance issues, respond with:
    {
        "action_type": "optimize_query",
        "optimized_query": "<your optimized SQL query here>",
        "optimization_type": "<what optimization was applied>",
        "explanation": "<why this optimization works>",
        "root_cause": "<what caused the performance issue>",
        "expected_improvement": "<expected performance gain>",
        "confidence": 0.85
    }

    Always provide valid JSON. Never include markdown code blocks.
""").strip()


def build_user_prompt(obs) -> str:
    ctx = obs.current_context
    return textwrap.dedent(f"""
        Task: {obs.task_description}
        Difficulty: {obs.difficulty}

        Buggy Query:
        {ctx.get('buggy_query', 'N/A')}

        Error Message:
        {ctx.get('error_message', 'N/A')}

        Database Schema:
        {json.dumps(ctx.get('database_schema', {}), indent=2)}

        Error Type Hint: {ctx.get('error_type_hint', 'unknown')}
        Category: {ctx.get('category', 'unknown')}
        Steps Remaining: {ctx.get('steps_remaining', 20)}

        Analyze the buggy query and provide your fix as a JSON object.
    """).strip()


# ─────────────────────────────────────────────
#  LLM CALL
# ─────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs, step: int) -> Action:
    """Call the LLM and parse its response into an Action."""
    user_prompt = build_user_prompt(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse JSON response
        # Remove markdown code blocks if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data        = json.loads(text)
        action_type = data.get("action_type", "submit_answer")

        if action_type == "optimize_query":
            return Action(
                action_type=ActionType.OPTIMIZE_QUERY,
                payload={
                    "optimized_query":      data.get("optimized_query", "SELECT 1"),
                    "optimization_type":    data.get("optimization_type", "Performance fix"),
                    "explanation":          data.get("explanation", ""),
                    "root_cause":           data.get("root_cause", ""),
                    "expected_improvement": data.get("expected_improvement", ""),
                    "confidence":           float(data.get("confidence", 0.7)),
                }
            )
        else:
            return Action(
                action_type=ActionType.SUBMIT_ANSWER,
                payload={
                    "fixed_query":    data.get("fixed_query", "SELECT 1"),
                    "explanation":    data.get("explanation", ""),
                    "error_type":     data.get("error_type", "syntax"),
                    "error_location": data.get("error_location", "unknown"),
                    "confidence":     float(data.get("confidence", 0.7)),
                }
            )

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Fallback to identify_error action
        return Action(
            action_type=ActionType.IDENTIFY_ERROR,
            payload={
                "error_location": "unknown",
                "error_type":     "syntax",
                "explanation":    "LLM call failed, using fallback"
            }
        )


# ─────────────────────────────────────────────
#  MAIN INFERENCE LOOP
# ─────────────────────────────────────────────

def run_episode(client: OpenAI, difficulty: str, task_id: str) -> dict:
    """Run one full episode and return results."""
    env      = SQLDebuggerEnvironment()
    obs      = env.reset(difficulty=difficulty, task_id=task_id)
    rewards  = []
    steps    = 0
    success  = False
    score    = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if env.state().done:
                break

            # Get action from LLM
            action       = get_llm_action(client, obs, step)
            action_str   = f"{action.action_type.value}"
            error_str    = None

            try:
                resp   = env.step(action)
                reward = resp.reward.score
                done   = resp.done
                obs    = resp.observation
            except Exception as e:
                reward   = -0.1
                done     = False
                error_str = str(e)[:100]

            rewards.append(reward)
            steps = step

            log_step(
                step   = step,
                action = action_str,
                reward = reward,
                done   = done,
                error  = error_str
            )

            if done:
                break

        # Calculate score
        total_reward = sum(rewards)
        score        = min(max(total_reward / MAX_STEPS, 0.0), 1.0)
        success      = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        error_str = str(e)[:100]

    finally:
        log_end(
            success = success,
            steps   = steps,
            score   = score,
            rewards = rewards
        )

    return {
        "task_id":    task_id,
        "difficulty": difficulty,
        "score":      score,
        "steps":      steps,
        "success":    success,
    }


def main():
    """Main entry point — runs inference on all 3 difficulty levels."""
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        ("easy",   "easy_001"),
        ("medium", "medium_001"),
        ("hard",   "hard_001"),
    ]

    results = []
    for difficulty, task_id in tasks:
        result = run_episode(client, difficulty, task_id)
        results.append(result)

    # Final summary
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n[DEBUG] Average Score: {avg_score:.3f}", flush=True)
    for r in results:
        print(f"[DEBUG] {r['difficulty']:8} | {r['task_id']:12} | score={r['score']:.3f} | steps={r['steps']}", flush=True)


if __name__ == "__main__":
    main()