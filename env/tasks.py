import json
import random
from pathlib import Path
from env.models import DifficultyLevel, TaskInfo

#  LOAD DATASETS

BASE_DIR = Path(__file__).parent.parent / "dataset"

def _load(filename: str) -> list[dict]:
    path = BASE_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

EASY_CASES   = _load("easy_cases.json")
MEDIUM_CASES = _load("medium_cases.json")
HARD_CASES   = _load("hard_cases.json")

ALL_CASES: dict[str, list[dict]] = {
    DifficultyLevel.EASY:   EASY_CASES,
    DifficultyLevel.MEDIUM: MEDIUM_CASES,
    DifficultyLevel.HARD:   HARD_CASES,
}

#  ACTION SCHEMA (required by /tasks validator)

ACTION_SCHEMA = {
    "identify_error": {
        "description": "Identify where and what the error is without fixing it yet",
        "payload_fields": {
            "error_location": {"type": "string", "required": True,  "description": "Where in the query the error occurs"},
            "error_type":     {"type": "string", "required": True,  "description": "Type: syntax | logic | performance"},
            "explanation":    {"type": "string", "required": False, "description": "Brief explanation of the error"}
        }
    },
    "propose_fix": {
        "description": "Propose a fix without submitting as final answer",
        "payload_fields": {
            "fixed_query":  {"type": "string", "required": True,  "description": "The proposed corrected SQL query"},
            "change_made":  {"type": "string", "required": True,  "description": "What specifically was changed"},
            "confidence":   {"type": "float",  "required": False, "description": "Confidence score 0.0-1.0"}
        }
    },
    "submit_answer": {
        "description": "Submit the final fixed query as the definitive answer",
        "payload_fields": {
            "fixed_query":   {"type": "string", "required": True,  "description": "Final corrected SQL query"},
            "explanation":   {"type": "string", "required": True,  "description": "Full explanation of what was wrong and how it was fixed"},
            "error_type":    {"type": "string", "required": False, "description": "Type: syntax | logic | performance"},
            "confidence":    {"type": "float",  "required": False, "description": "Confidence score 0.0-1.0"}
        }
    },
    "request_hint": {
        "description": "Request a hint — costs 0.05 reward penalty per hint",
        "payload_fields": {
            "hint_type": {"type": "string", "required": False, "description": "Type of hint wanted: location | error_type | fix_direction"}
        }
    },
    "explain_issue": {
        "description": "Explain the issue in detail — earns partial credit even without fixing",
        "payload_fields": {
            "explanation":    {"type": "string", "required": True,  "description": "Detailed explanation of the SQL problem"},
            "impact":         {"type": "string", "required": False, "description": "What impact the bug has on query results or performance"},
            "root_cause":     {"type": "string", "required": False, "description": "Root cause analysis"}
        }
    },
    "optimize_query": {
        "description": "Submit an optimized version of the query (used for hard/performance tasks)",
        "payload_fields": {
            "optimized_query":    {"type": "string", "required": True,  "description": "The performance-optimized SQL query"},
            "optimization_type":  {"type": "string", "required": True,  "description": "What optimization was applied"},
            "expected_improvement":{"type": "string", "required": False, "description": "Expected performance gain description"},
            "explanation":        {"type": "string", "required": False, "description": "Why this optimization works"},
            "confidence":         {"type": "float",  "required": False, "description": "Confidence 0.0-1.0"}
        }
    }
}
#  TASK MANAGER


class TaskManager:
    """
    Manages task selection, hint generation, and task metadata.
    All tasks are loaded from JSON datasets — no hardcoded tasks.
    """

    def __init__(self):
        self._used_ids: set[str] = set()

    def get_task(self, difficulty: DifficultyLevel, task_id: str | None = None) -> dict:
        """
        Returns a task dict for the given difficulty.
        If task_id is provided, returns that specific task.
        Otherwise picks randomly, avoiding recently used tasks.
        """
        pool = ALL_CASES[difficulty]

        if task_id:
            for case in pool:
                if case["id"] == task_id:
                    return case
            raise ValueError(f"Task '{task_id}' not found in {difficulty} pool")

        # Avoid repeating recently used tasks
        available = [c for c in pool if c["id"] not in self._used_ids]
        if not available:
            self._used_ids.clear()
            available = pool

        task = random.choice(available)
        self._used_ids.add(task["id"])
        return task

    def get_random_task(self) -> dict:
        """Pick a random task from any difficulty."""
        difficulty = random.choice(list(DifficultyLevel))
        return self.get_task(difficulty)

    def build_observation_context(self, task: dict) -> dict:
        """
        Build the current_context dict for the Observation.
        CRITICAL: Must NOT leak the fixed_query (ground truth) to the agent.
        """
        context = {
            "buggy_query":       task["buggy_query"],
            "error_message":     task["error_message"],
            "database_schema":   task["database_schema"],
            "error_type_hint":   task["error_type"],
            "category":          task["category"],
            "estimated_steps":   task["estimated_fix_steps"],
        }

        # For performance tasks include extra context
        if task.get("performance_issue"):
            context["performance_issue"] = {
                "type":   task["performance_issue"]["type"],
                "impact": task["performance_issue"]["impact"],
                # Do NOT include timing numbers — agent must figure it out
            }

        # Include expected output shape (but not the fixed query!)
        if task.get("expected_output") and isinstance(task["expected_output"], list):
            context["expected_output_sample"] = task["expected_output"][:1]

        return context

    def get_hint(self, task: dict, hint_number: int) -> str:
        """
        Returns progressive hints. Each hint gives more info.
        Hints cost -0.05 reward each (handled in reward.py).
        """
        hints = [
            f"Hint 1: The error is in the {task.get('error_location', 'query')}.",
            f"Hint 2: This is a {task.get('error_type', 'unknown')} type error. Category: {task.get('category')}.",
            f"Hint 3: Fix description — {task.get('fix_description', 'Review the query carefully.')}",
        ]
        idx = min(hint_number - 1, len(hints) - 1)
        return hints[idx]

    def list_all_tasks(self) -> list[TaskInfo]:
        """Returns TaskInfo list for the /tasks endpoint."""
        result = []
        for difficulty, cases in ALL_CASES.items():
            for case in cases:
                result.append(TaskInfo(
                    id=case["id"],
                    difficulty=difficulty,
                    description=case["description"],
                    action_schema=ACTION_SCHEMA
                ))
        return result

    def get_ground_truth(self, task_id: str) -> dict | None:
        """Returns the full ground truth for a task (used by grader only)."""
        for cases in ALL_CASES.values():
            for case in cases:
                if case["id"] == task_id:
                    return case
        return None


# Singleton instance
task_manager = TaskManager()