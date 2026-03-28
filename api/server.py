import os
import time
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from env.environment import environment
from env.models import (
    Action, Observation, EpisodeState,
    DifficultyLevel, ActionType,
    StepResponse, ResetResponse, TaskListResponse,
    BaselineResponse, BaselineResult,
    GraderRequest, GraderResponse,
    HealthResponse, TaskInfo
)
from env.tasks import task_manager, ACTION_SCHEMA
from env.graders import grade


# ─────────────────────────────────────────────
#  STARTUP / SHUTDOWN
# ─────────────────────────────────────────────

_startup_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up — pre-load datasets and reset environment
    environment.reset(difficulty="easy")
    yield
    # Shutdown — nothing to clean up

# ─────────────────────────────────────────────
#  APP DEFINITION
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "SQL Query Debugger — OpenEnv Environment",
    description = (
        "An OpenEnv-compliant reinforcement learning environment where AI agents "
        "learn to debug SQL queries across syntax errors, logic bugs, and performance issues. "
        "Built for the META × PyTorch × SST OpenEnv Hackathon."
    ),
    version     = "1.0.0",
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
#  1. /health — GET
#  Must always return 200 even if env not initialized
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Liveness check. Always returns 200.
    Used by HF Space health monitoring.
    """
    return HealthResponse(
        status  = "ok",
        version = "1.0.0",
        uptime  = round(time.time() - _startup_time, 2)
    )


# ─────────────────────────────────────────────
#  2. /reset — POST
#  Starts new episode, returns Observation
# ─────────────────────────────────────────────

class ResetRequest(Action.__class__):
    pass

from pydantic import BaseModel

class ResetBody(BaseModel):
    difficulty: Optional[str] = None
    task_id:    Optional[str] = None

@app.post("/reset", response_model=Observation, tags=["Environment"])
async def reset(body: ResetBody = ResetBody()):
    """
    Starts a fresh episode.
    Returns the initial Observation the agent sees.

    Edge case: always returns valid Observation even if dataset issues occur.
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
#  Accepts Action, returns StepResponse
# ─────────────────────────────────────────────

@app.post("/step", response_model=StepResponse, tags=["Environment"])
async def step(action: Action):
    """
    Submits an action to the environment.
    Returns (observation, reward, done, info).

    Edge cases:
    - Invalid/malformed action → reward=-0.1, done=False
    - Episode already done → returns terminal state
    - Null payload → graceful penalty
    """
    try:
        response = environment.step(action)
        return response
    except ValidationError as e:
        # Malformed action — return penalty reward, never crash
        obs = environment.state()
        return StepResponse(
            observation = environment._build_observation(),
            reward      = __import__("env.models", fromlist=["Reward"]).Reward(
                score     = -0.1,
                breakdown = {"validation_error": -0.1},
                feedback  = f"Malformed action: {str(e)}"
            ),
            done = False,
            info = {"error": "validation_error", "detail": str(e)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


# ─────────────────────────────────────────────
#  4. /state — GET
#  Returns current environment state
# ─────────────────────────────────────────────

@app.get("/state", response_model=EpisodeState, tags=["Environment"])
async def state():
    """
    Returns full current environment state.
    Works before reset() is called — returns default empty state.
    Must always be JSON-serializable.
    """
    return environment.state()


# ─────────────────────────────────────────────
#  5. /tasks — GET
#  Lists all tasks + action schema
# ─────────────────────────────────────────────

@app.get("/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def tasks():
    """
    Lists all 15 tasks with full action schema definitions.
    Validator checks for action field definitions, not just task names.
    """
    all_tasks = task_manager.list_all_tasks()
    return TaskListResponse(
        tasks        = all_tasks,
        total        = len(all_tasks),
        action_types = [a.value for a in ActionType]
    )


# ─────────────────────────────────────────────
#  6. /grader — POST
#  Grades a completed episode
# ─────────────────────────────────────────────

@app.post("/grader", response_model=GraderResponse, tags=["Grading"])
async def grader(request: GraderRequest):
    """
    Grades a completed episode.
    Returns float score between 0.0 and 1.0.

    Edge cases:
    - Null/empty episode → returns 0.0, never crashes
    - Unknown task_id → returns 0.0 with explanation
    """
    try:
        # Edge case: null action in request
        if request.action is None:
            return GraderResponse(
                score     = 0.0,
                feedback  = "No action provided for grading.",
                breakdown = {"error": "null_action"}
            )

        score, breakdown, feedback = grade(request.action, request.task_id)
        return GraderResponse(
            score     = score,
            feedback  = feedback,
            breakdown = breakdown
        )
    except Exception as e:
        # Never crash — return 0.0
        return GraderResponse(
            score     = 0.0,
            feedback  = f"Grader error: {str(e)}",
            breakdown = {"error": str(e)}
        )


# ─────────────────────────────────────────────
#  7. /baseline — POST
#  Runs baseline inference, returns scores
#  Must complete within 60 seconds
# ─────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResponse, tags=["Baseline"])
async def baseline():
    """
    Runs the baseline agent against all 3 difficulty levels.
    Returns scores JSON. Must complete within 60 seconds.

    Edge case: OPENAI_API_KEY not set → returns error scores without crashing.
    """
    try:
        # Import here to avoid circular imports
        import baseline as baseline_module
        results = await asyncio.wait_for(
            asyncio.to_thread(baseline_module.run_baseline),
            timeout=55.0  # 5s buffer before 60s limit
        )
        return results
    except asyncio.TimeoutError:
        # Return partial results on timeout
        return BaselineResponse(
            results=[
                BaselineResult(task_id="timeout", difficulty=DifficultyLevel.EASY,
                               score=0.0, steps=0, feedback="Baseline timed out after 55 seconds.")
            ],
            average_score=0.0
        )
    except Exception as e:
        return BaselineResponse(
            results=[
                BaselineResult(task_id="error", difficulty=DifficultyLevel.EASY,
                               score=0.0, steps=0, feedback=f"Baseline error: {str(e)}")
            ],
            average_score=0.0
        )


# ─────────────────────────────────────────────
#  ROOT — redirect to docs
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name":        "SQL Query Debugger — OpenEnv Environment",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/health",
        "endpoints":   ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/health"],
        "hackathon":   "META × PyTorch × SST OpenEnv Hackathon",
        "domain":      "SQL Query Debugging",
        "tasks_count": 15,
    }