from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
from enum import Enum
import time


#  ENUMS

class DifficultyLevel(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class ActionType(str, Enum):
    IDENTIFY_ERROR    = "identify_error"
    PROPOSE_FIX       = "propose_fix"
    SUBMIT_ANSWER     = "submit_answer"
    REQUEST_HINT      = "request_hint"
    EXPLAIN_ISSUE     = "explain_issue"
    OPTIMIZE_QUERY    = "optimize_query"

# CORE MODELS


class Observation(BaseModel):
    task_id:          str            = Field(..., description="Unique task identifier")
    task_description: str            = Field(..., description="What the agent must do")
    current_context:  dict           = Field(..., description="What the agent currently sees")
    step_count:       int            = Field(default=0, ge=0, description="Steps taken so far")
    difficulty:       DifficultyLevel = Field(..., description="Task difficulty level")
    max_steps:        int            = Field(default=20, description="Maximum steps allowed")
    hints_used:       int            = Field(default=0, description="Number of hints used")
    previous_actions: list[str]      = Field(default_factory=list, description="History of action types taken")
    metadata:         dict           = Field(default_factory=dict, description="Extra task metadata")

    model_config = {"json_schema_extra": {
        "example": {
            "task_id": "easy_001",
            "task_description": "Fix the SQL syntax error in the query below.",
            "current_context": {
                "buggy_query": "SELECT id, name FROM users WHERE id = 1 AND",
                "error_message": "SyntaxError: unexpected end of input",
                "database_schema": "users(id INT, name VARCHAR, email VARCHAR)"
            },
            "step_count": 0,
            "difficulty": "easy",
            "max_steps": 20,
            "hints_used": 0,
            "previous_actions": [],
            "metadata": {"category": "syntax", "estimated_fix_steps": 2}
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
        # Edge case: extremely long agent output — truncate gracefully
        def truncate(obj, max_len=5000):
            if isinstance(obj, str) and len(obj) > max_len:
                return obj[:max_len] + "...[truncated]"
            if isinstance(obj, dict):
                return {k: truncate(val, max_len) for k, val in obj.items()}
            return obj
        return truncate(v)

    model_config = {"json_schema_extra": {
        "example": {
            "action_type": "submit_answer",
            "payload": {
                "fixed_query":   "SELECT id, name FROM users WHERE id = 1",
                "explanation":   "Removed the trailing AND which caused a syntax error",
                "error_type":    "syntax",
                "confidence":    0.95
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
                "correct_answer":  0.5,
                "explanation":     0.2,
                "confidence":      0.05,
                "step_efficiency": 0.0
            },
            "feedback": "Correct fix applied. Good explanation provided. Minor efficiency penalty."
        }
    }}


#  EPISODE STATE (used by state() endpoint)

class EpisodeState(BaseModel):
    task_id:          Optional[str]            = Field(default=None)
    difficulty:       Optional[DifficultyLevel] = Field(default=None)
    step_count:       int                       = Field(default=0)
    total_reward:     float                     = Field(default=0.0)
    done:             bool                      = Field(default=False)
    hints_used:       int                       = Field(default=0)
    previous_actions: list[str]                 = Field(default_factory=list)
    action_counts:    dict[str, int]            = Field(default_factory=dict)
    started_at:       Optional[float]           = Field(default=None)
    last_reward:      float                     = Field(default=0.0)
    initialized:      bool                      = Field(default=False)

    model_config = {"json_schema_extra": {
        "example": {
            "task_id":          "medium_002",
            "difficulty":       "medium",
            "step_count":       3,
            "total_reward":     0.45,
            "done":             False,
            "hints_used":       1,
            "previous_actions": ["identify_error", "request_hint", "propose_fix"],
            "action_counts":    {"identify_error": 1, "request_hint": 1, "propose_fix": 1},
            "started_at":       1700000000.0,
            "last_reward":      0.25,
            "initialized":      True
        }
    }}


#  API REQUEST / RESPONSE WRAPPERS

class StepResponse(BaseModel):
    observation: Observation
    reward:      Reward
    done:        bool
    info:        dict

class ResetResponse(BaseModel):
    observation: Observation

class TaskInfo(BaseModel):
    id:           str
    difficulty:   DifficultyLevel
    description:  str
    action_schema: dict   # REQUIRED by validator — field definitions not just names

class TaskListResponse(BaseModel):
    tasks:         list[TaskInfo]
    total:         int
    action_types:  list[str]

class BaselineResult(BaseModel):
    task_id:    str
    difficulty: DifficultyLevel
    score:      float
    steps:      int
    feedback:   str

class BaselineResponse(BaseModel):
    results:      list[BaselineResult]
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
            "score": 0.75,
            "feedback": "Correct fix applied.",
            "breakdown": {"fix_correctness": 0.5, "explanation": 0.15, "confidence": 0.05}
        }
    }}

class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"
    uptime:  float = Field(default_factory=time.time)