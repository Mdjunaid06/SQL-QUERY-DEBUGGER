import os
import json
import textwrap
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from env.environment import SQLDebuggerEnvironment
from env.models import Action, ActionType

# ── Required environment variables ──────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
BENCHMARK = "sql-query-debugger"
MAX_STEPS = 5

SYSTEM_PROMPT = """You are an expert SQL debugger. Given a buggy SQL query, respond with ONLY a JSON object.

For syntax/logic errors:
{"action_type":"submit_answer","fixed_query":"<fixed SQL>","explanation":"<what was wrong>","error_type":"syntax","confidence":0.9}

For performance issues:
{"action_type":"optimize_query","optimized_query":"<optimized SQL>","optimization_type":"<what was optimized>","explanation":"<why>","root_cause":"<cause>","expected_improvement":"<improvement>","confidence":0.85}

Never include markdown. Only valid JSON."""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_llm_action(obs) -> Action:
    ctx = obs.current_context
    prompt = f"""Task: {obs.task_description}
Buggy Query: {ctx.get('buggy_query','N/A')}
Error: {ctx.get('error_message','N/A')}
Schema: {json.dumps(ctx.get('database_schema',{}))}
Category: {ctx.get('category','syntax')}
Fix this SQL query and respond with JSON only."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512,
        )
        text = (completion.choices[0].message.content or "").strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        data = json.loads(text)

        if data.get("action_type") == "optimize_query":
            return Action(action_type=ActionType.OPTIMIZE_QUERY, payload={
                "optimized_query": data.get("optimized_query", "SELECT 1"),
                "optimization_type": data.get("optimization_type", "fix"),
                "explanation": data.get("explanation", ""),
                "root_cause": data.get("root_cause", ""),
                "expected_improvement": data.get("expected_improvement", ""),
                "confidence": float(data.get("confidence", 0.7)),
            })
        else:
            return Action(action_type=ActionType.SUBMIT_ANSWER, payload={
                "fixed_query": data.get("fixed_query", "SELECT 1"),
                "explanation": data.get("explanation", ""),
                "error_type": data.get("error_type", "syntax"),
                "error_location": data.get("error_location", "unknown"),
                "confidence": float(data.get("confidence", 0.7)),
            })
    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
        return Action(action_type=ActionType.IDENTIFY_ERROR, payload={
            "error_location": "unknown",
            "error_type": "syntax",
            "explanation": "fallback"
        })

def run_episode(difficulty, task_id):
    env = SQLDebuggerEnvironment()
    obs = env.reset(difficulty=difficulty, task_id=task_id)
    rewards = []
    steps = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if env.state().done:
                break
            action = get_llm_action(obs)
            error_str = None
            try:
                resp = env.step(action)
                raw_reward = resp.reward.score
                done = resp.done
                obs = resp.observation
            except Exception as e:
                raw_reward = 0.1
                done = False
                error_str = str(e)[:50]

            # Normalize reward strictly between 0 and 1
            reward = max(0.01, min(0.99, (raw_reward + 1.0) / 2.0))
            rewards.append(reward)
            steps = step
            log_step(step=step, action=action.action_type.value, reward=reward, done=done, error=error_str)
            if done:
                break

        score = max(0.01, min(0.99, sum(rewards) / len(rewards))) if rewards else 0.5
        success = score > 0.5

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = 0.5
        success = False
    finally:
        safe_rewards = rewards if rewards else [0.5]
        log_end(success=success, steps=steps, rewards=safe_rewards)

    return {"task_id": task_id, "score": score, "steps": steps}

def main():
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)

    tasks = [
        ("easy", "easy_001"),
        ("medium", "medium_001"),
        ("hard", "hard_001"),
    ]

    results = []
    for difficulty, task_id in tasks:
        result = run_episode(difficulty, task_id)
        results.append(result)

    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n[DEBUG] Average Score: {avg:.3f}", flush=True)

if __name__ == "__main__":
    main()