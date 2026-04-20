from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
from enum import Enum
import time


# ─────────────────────────────────────────────
#  ENUMS
# ─────────────────────────────────────────────

class DifficultyLevel(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class ActionType(str, Enum):
    # ── Round 1 actions (keep — backward compatible) ──
    IDENTIFY_ERROR  = "identify_error"
    PROPOSE_FIX     = "propose_fix"
    SUBMIT_ANSWER   = "submit_answer"
    REQUEST_HINT    = "request_hint"
    EXPLAIN_ISSUE   = "explain_issue"
    OPTIMIZE_QUERY  = "optimize_query"

    # ── Round 2 new actions ──
    INSPECT_QUERY    = "inspect_query"
    ANALYZE_INDEXES  = "analyze_indexes"
    CREATE_INDEX     = "create_index"
    REWRITE_QUERY    = "rewrite_query"
    ADD_COLUMN       = "add_column"
    DROP_INDEX       = "drop_index"
    PARTITION_TABLE  = "partition_table"
    ANALYZE_STATS    = "analyze_statistics"
    SUBMIT_REPORT    = "submit_report"


# ─────────────────────────────────────────────
#  CORE MODELS
# ─────────────────────────────────────────────

class Observation(BaseModel):
    task_id:          str             = Field(..., description="Unique task identifier")
    task_description: str             = Field(..., description="What the agent must do")
    current_context:  dict            = Field(..., description="What the agent currently sees")
    step_count:       int             = Field(default=0, ge=0, description="Steps taken so far")
    difficulty:       DifficultyLevel = Field(..., description="Task difficulty level")
    max_steps:        int             = Field(default=50, description="Maximum steps allowed")
    hints_used:       int             = Field(default=0, description="Number of hints used")
    previous_actions: list[str]       = Field(default_factory=list, description="History of action types taken")
    metadata:         dict            = Field(default_factory=dict, description="Extra task metadata")

    model_config = {"json_schema_extra": {
        "example": {
            "task_id": "easy_s001",
            "task_description": "Optimize a slow user lookup query on 10K users table.",
            "current_context": {
                "tables": [{"name": "users", "rows": 10000, "indexes": ["PRIMARY"]}],
                "slow_queries": [{"id": "q1", "sql": "SELECT * FROM users WHERE email=?", "avg_ms": 2000}],
                "performance_score": 8.0,
                "target_score": 80.0
            },
            "step_count": 0,
            "difficulty": "easy",
            "max_steps": 50,
            "hints_used": 0,
            "previous_actions": [],
            "metadata": {"scenario_id": "easy_s001", "baseline_score": 8.0}
        }
    }}


class Action(BaseModel):
    action_type: ActionType = Field(..., description="Type of action the agent is taking")
    payload:     dict       = Field(..., description="Action-specific data")

    @field_validator("payload")
    @classmethod
    def payload_must_not_be_empty(cls, v):
        if v is None:
            raise ValueError("Payload cannot be None")
        return v

    @field_validator("payload")
    @classmethod
    def truncate_long_strings(cls, v):
        def truncate(obj, max_len=5000):
            if isinstance(obj, str) and len(obj) > max_len:
                return obj[:max_len] + "...[truncated]"
            if isinstance(obj, dict):
                return {k: truncate(val, max_len) for k, val in obj.items()}
            return obj
        return truncate(v)

    model_config = {"json_schema_extra": {
        "example": {
            "action_type": "create_index",
            "payload": {
                "table":   "users",
                "columns": ["email"]
            }
        }
    }}


class Reward(BaseModel):
    score:     float = Field(..., ge=-1.0, le=1.0, description="Reward score between -1.0 and 1.0")
    breakdown: dict  = Field(..., description="Partial credit details per dimension")
    feedback:  str   = Field(..., description="Human-readable explanation of the reward")

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        return max(0.001, min(0.999, round(v, 4)))

    model_config = {"json_schema_extra": {
        "example": {
            "score": 0.75,
            "breakdown": {
                "step_reward":    0.05,
                "delta_reward":   0.40,
                "milestone_bonus": 0.15,
                "total":          0.60
            },
            "feedback": "Index created. Performance improved 55%. Milestone bonus earned!"
        }
    }}


# ─────────────────────────────────────────────
#  EPISODE STATE (used by state() endpoint)
# ─────────────────────────────────────────────

class EpisodeState(BaseModel):
    task_id:          Optional[str]             = Field(default=None)
    difficulty:       Optional[DifficultyLevel] = Field(default=None)
    step_count:       int                       = Field(default=0)
    total_reward:     float                     = Field(default=0.0)
    done:             bool                      = Field(default=False)
    hints_used:       int                       = Field(default=0)
    previous_actions: list[str]                 = Field(default_factory=list)
    action_counts:    dict[str, Any]            = Field(default_factory=dict)
    started_at:       Optional[float]           = Field(default=None)
    last_reward:      float                     = Field(default=0.0)
    initialized:      bool                      = Field(default=False)

    model_config = {"json_schema_extra": {
        "example": {
            "task_id":          "easy_s001",
            "difficulty":       "easy",
            "step_count":       3,
            "total_reward":     0.65,
            "done":             False,
            "hints_used":       0,
            "previous_actions": ["inspect_query", "analyze_indexes", "create_index"],
            "action_counts":    {"inspect_query": 1, "analyze_indexes": 1, "create_index": 1},
            "started_at":       1700000000.0,
            "last_reward":      0.45,
            "initialized":      True
        }
    }}


# ─────────────────────────────────────────────
#  API REQUEST / RESPONSE WRAPPERS
# ─────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: Observation
    reward:      Reward
    done:        bool
    info:        dict

class ResetResponse(BaseModel):
    observation: Observation

class TaskInfo(BaseModel):
    id:            str
    difficulty:    DifficultyLevel
    description:   str
    action_schema: dict

class TaskListResponse(BaseModel):
    tasks:        list[TaskInfo]
    total:        int
    action_types: list[str]

class BaselineResult(BaseModel):
    task_id:    str
    difficulty: DifficultyLevel
    score:      float
    steps:      int
    feedback:   str

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        return max(0.001, min(0.999, round(float(v), 4)))

class BaselineResponse(BaseModel):
    results:       list[BaselineResult]
    average_score: float
    completed_at:  float = Field(default_factory=time.time)

class GraderRequest(BaseModel):
    task_id:  str
    action:   Optional[Action] = None
    episode:  Optional[dict]   = None

class GraderResponse(BaseModel):
    score:     float = Field(..., description="Score strictly between 0 and 1 exclusive")
    feedback:  str
    breakdown: dict

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        return max(0.001, min(0.999, round(float(v), 4)))

    model_config = {"json_schema_extra": {
        "example": {
            "score":    0.82,
            "feedback": "Performance improved from 12.5 to 85.0. Excellent optimization!",
            "breakdown": {"perf_improvement": 0.60, "step_efficiency": 0.12, "index_quality": 0.10}
        }
    }}

class HealthResponse(BaseModel):
    status:  str   = "ok"
    version: str   = "2.0.0"
    uptime:  float = Field(default_factory=time.time)


# ─────────────────────────────────────────────
#  ROUND 2 — PROGRESS RESPONSE
# ─────────────────────────────────────────────

class ProgressResponse(BaseModel):
    scenario_id:         Optional[str]  = Field(default=None)
    performance_score:   float          = Field(default=0.0, description="Current DB performance score 0-100")
    baseline_score:      float          = Field(default=0.0, description="Starting score this episode")
    target_score:        float          = Field(default=85.0, description="Score needed to succeed")
    improvement_history: list[float]    = Field(default_factory=list)
    milestones_earned:   list[float]    = Field(default_factory=list)
    best_score:          float          = Field(default=0.0)
    steps_used:          int            = Field(default=0)
    budget_remaining:    int            = Field(default=50)
    total_reward:        float          = Field(default=0.0)
