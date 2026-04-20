import os
import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ValidationError

from env.environment import environment
from env.models import (
    Action, Observation, EpisodeState,
    DifficultyLevel, ActionType,
    StepResponse, ResetResponse, TaskListResponse,
    BaselineResponse, BaselineResult,
    GraderRequest, GraderResponse,
    HealthResponse, TaskInfo, ProgressResponse
)
from env.tasks import task_manager, ACTION_SCHEMA
from env.graders import grade


# ─────────────────────────────────────────────
#  STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

_startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    environment.reset(difficulty="easy")
    yield


# ─────────────────────────────────────────────
#  APP DEFINITION
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "SQL Database Engineer Agent — OpenEnv Environment",
    description = (
        "An OpenEnv-compliant reinforcement learning environment where AI agents "
        "learn to act like senior database engineers. "
        "The agent manages a simulated production database over 50+ steps: "
        "inspecting slow queries, creating indexes, rewriting queries, partitioning tables. "
        "Built for the META x PyTorch x SST OpenEnv Hackathon Finals — April 25-26, Bangalore."
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────
#  GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = 500,
        content     = {"error": str(exc), "type": type(exc).__name__}
    )


# ─────────────────────────────────────────────
#  FAVICON
# ─────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ─────────────────────────────────────────────
#  1. /health — GET
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check. Always returns 200."""
    return HealthResponse(
        status  = "ok",
        version = "2.0.0",
        uptime  = round(time.time() - _startup_time, 2)
    )


# ─────────────────────────────────────────────
#  2. /reset — POST
# ─────────────────────────────────────────────

class ResetBody(BaseModel):
    difficulty: Optional[str] = None
    task_id:    Optional[str] = None

@app.post("/reset", response_model=Observation, tags=["Environment"])
async def reset(body: ResetBody = ResetBody()):
    """
    Starts a fresh episode. Initializes DatabaseSimulator.
    Returns the initial Observation with DB state and slow queries.
    """
    try:
        obs = environment.reset(
            difficulty = body.difficulty,
            task_id    = body.task_id
        )
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# ─────────────────────────────────────────────
#  3. /step — POST
# ─────────────────────────────────────────────

@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(action: Action):
    """
    Submits an action to the environment.
    Round 2 actions: inspect_query, create_index, rewrite_query,
    partition_table, analyze_statistics, analyze_indexes, submit_report.
    Returns (observation, reward, done, info) with DB performance delta.
    """
    try:
        response = environment.step(action)
        return response
    except ValidationError as e:
        from env.models import Reward
        return StepResponse(
            observation = environment._build_observation(),
            reward      = Reward(
                score     = 0.001,
                breakdown = {"validation_error": 0.001},
                feedback  = f"Malformed action: {str(e)}"
            ),
            done = False,
            info = {"error": "validation_error", "detail": str(e)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


# ─────────────────────────────────────────────
#  4. /state — GET
# ─────────────────────────────────────────────

@app.get("/state", response_model=EpisodeState, tags=["Environment"])
async def state():
    """Returns full current environment state including performance history."""
    return environment.state()


# ─────────────────────────────────────────────
#  5. /tasks — GET
# ─────────────────────────────────────────────

@app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def tasks():
    """
    Lists all 30 tasks (15 Round 2 scenarios + 15 Round 1 cases).
    Includes complete action schema for all 15 action types.
    """
    all_tasks = task_manager.list_all_tasks()
    return TaskListResponse(
        tasks        = all_tasks,
        total        = len(all_tasks),
        action_types = [a.value for a in ActionType]
    )


# ─────────────────────────────────────────────
#  6. /grader — POST
# ─────────────────────────────────────────────

@app.post("/grader", response_model=GraderResponse, tags=["Grading"])
async def grader(request: GraderRequest):
    """
    Grades a completed episode action.
    For Round 2 submit_report: computes score from DB performance improvement.
    Returns float score strictly between 0.0 and 1.0 exclusive.
    """
    try:
        if request.action is None:
            return GraderResponse(
                score     = 0.001,
                feedback  = "No action provided for grading.",
                breakdown = {"error": "null_action"}
            )

        # Round 2: submit_report grading uses DB state
        if request.action.action_type == ActionType.SUBMIT_REPORT:
            ep_state    = environment.state()
            perf_history = ep_state.action_counts.get("_perf_history", [0.0])
            baseline     = ep_state.action_counts.get("_baseline_score", 0.0)
            best_score   = ep_state.action_counts.get("_best_score", 0.0)
            current      = perf_history[-1] if perf_history else 0.0
            max_possible = max(1.0, 100.0 - baseline)

            perf_improvement = (current - baseline) / max_possible
            step_efficiency  = 1.0 - (ep_state.step_count / max(1, 50))
            score = round(
                (perf_improvement * 0.60) + (step_efficiency * 0.20) + 0.10, 4
            )
            score = max(0.001, min(0.999, score))

            return GraderResponse(
                score    = score,
                feedback = (
                    f"DB performance: {baseline:.1f} → {current:.1f} "
                    f"(best: {best_score:.1f}). "
                    f"Steps used: {ep_state.step_count}/50."
                ),
                breakdown = {
                    "perf_improvement": round(perf_improvement, 4),
                    "step_efficiency":  round(step_efficiency, 4),
                    "base_score":       0.10,
                }
            )

        # Round 1 grading
        score, breakdown, feedback = grade(request.action, request.task_id)
        score = max(0.001, min(0.999, score))
        return GraderResponse(score=score, feedback=feedback, breakdown=breakdown)

    except Exception as e:
        return GraderResponse(
            score     = 0.001,
            feedback  = f"Grader error: {str(e)}",
            breakdown = {"error": str(e)}
        )


# ─────────────────────────────────────────────
#  7. /baseline — POST
# ─────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResponse, tags=["Baseline"])
async def baseline():
    """
    Runs the baseline agent against all difficulty levels.
    Must complete within 60 seconds.
    """
    try:
        import baseline as baseline_module
        results = await asyncio.wait_for(
            asyncio.to_thread(baseline_module.run_baseline),
            timeout=55.0
        )
        return results
    except asyncio.TimeoutError:
        return BaselineResponse(
            results=[BaselineResult(
                task_id="timeout", difficulty=DifficultyLevel.EASY,
                score=0.0, steps=0, feedback="Baseline timed out."
            )],
            average_score=0.0
        )
    except Exception as e:
        return BaselineResponse(
            results=[BaselineResult(
                task_id="error", difficulty=DifficultyLevel.EASY,
                score=0.0, steps=0, feedback=f"Baseline error: {str(e)}"
            )],
            average_score=0.0
        )


# ─────────────────────────────────────────────
#  8. /progress — GET  (Round 2 NEW)
# ─────────────────────────────────────────────

@app.get("/progress", response_model=ProgressResponse, tags=["Training"])
async def progress():
    """
    Returns DB performance history for training visualization.
    Used by evaluate_agent.py to generate reward curves.
    Shows improvement from baseline to current score.
    """
    ep_state     = environment.state()
    ac           = ep_state.action_counts
    perf_history = ac.get("_perf_history", [])
    milestones   = ac.get("_milestones", [])
    baseline     = ac.get("_baseline_score", 0.0)
    target       = ac.get("_target_score", 85.0)
    best         = ac.get("_best_score", 0.0)
    current      = perf_history[-1] if perf_history else 0.0

    return ProgressResponse(
        scenario_id         = ep_state.task_id,
        performance_score   = current,
        baseline_score      = baseline,
        target_score        = target,
        improvement_history = perf_history,
        milestones_earned   = milestones,
        best_score          = best,
        steps_used          = ep_state.step_count,
        budget_remaining    = max(0, 50 - ep_state.step_count),
        total_reward        = ep_state.total_reward,
    )


# ─────────────────────────────────────────────
#  ROOT
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name":        "SQL Database Engineer Agent — OpenEnv Environment",
        "version":     "2.0.0",
        "tagline":     "Training LLMs to act like senior database engineers",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints":   ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/progress", "/health"],
        "hackathon":   "META x PyTorch x SST OpenEnv Hackathon — Finals April 25-26 Bangalore",
        "domain":      "Long-Horizon Database Engineering",
        "tasks_count": 30,
        "max_steps":   50,
        "themes":      ["Long-Horizon Planning", "World Modeling", "Self-Improvement", "Wildcard"],
    }
