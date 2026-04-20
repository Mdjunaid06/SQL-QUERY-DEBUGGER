import json
import random
from pathlib import Path
from env.models import DifficultyLevel, TaskInfo

# ─────────────────────────────────────────────
#  LOAD DATASETS — Round 1 + Round 2
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent / "dataset"


def _load(filename: str) -> list[dict]:
    path = BASE_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Round 1 cases (keep for backward compatibility)
EASY_CASES   = _load("easy_cases.json")
MEDIUM_CASES = _load("medium_cases.json")
HARD_CASES   = _load("hard_cases.json")

# Round 2 scenarios (new long-horizon DB engineering tasks)
EASY_SCENARIOS   = _load("easy_scenarios.json")
MEDIUM_SCENARIOS = _load("medium_scenarios.json")
HARD_SCENARIOS   = _load("hard_scenarios.json")

# Combined pools — Round 2 scenarios take priority (listed first)
ALL_CASES: dict[str, list[dict]] = {
    DifficultyLevel.EASY:   EASY_SCENARIOS   + EASY_CASES,
    DifficultyLevel.MEDIUM: MEDIUM_SCENARIOS + MEDIUM_CASES,
    DifficultyLevel.HARD:   HARD_SCENARIOS   + HARD_CASES,
}

# Round 2 only (for training pipeline)
SCENARIO_ONLY: dict[str, list[dict]] = {
    DifficultyLevel.EASY:   EASY_SCENARIOS,
    DifficultyLevel.MEDIUM: MEDIUM_SCENARIOS,
    DifficultyLevel.HARD:   HARD_SCENARIOS,
}


# ─────────────────────────────────────────────
#  ACTION SCHEMA (required by /tasks validator)
# ─────────────────────────────────────────────

ACTION_SCHEMA = {
    # ── Round 1 actions ──────────────────────────────────────────
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
            "fixed_query": {"type": "string", "required": True,  "description": "The proposed corrected SQL query"},
            "change_made": {"type": "string", "required": True,  "description": "What specifically was changed"},
            "confidence":  {"type": "float",  "required": False, "description": "Confidence score 0.0-1.0"}
        }
    },
    "submit_answer": {
        "description": "Submit the final fixed query as the definitive answer",
        "payload_fields": {
            "fixed_query": {"type": "string", "required": True,  "description": "Final corrected SQL query"},
            "explanation": {"type": "string", "required": True,  "description": "Full explanation of fix"},
            "error_type":  {"type": "string", "required": False, "description": "syntax | logic | performance"},
            "confidence":  {"type": "float",  "required": False, "description": "Confidence 0.0-1.0"}
        }
    },
    "request_hint": {
        "description": "Request a hint — costs 0.10 reward penalty per hint",
        "payload_fields": {
            "hint_type": {"type": "string", "required": False, "description": "location | error_type | fix_direction"}
        }
    },
    "explain_issue": {
        "description": "Explain the issue in detail",
        "payload_fields": {
            "explanation": {"type": "string", "required": True,  "description": "Detailed explanation"},
            "impact":      {"type": "string", "required": False, "description": "Impact on query performance"},
            "root_cause":  {"type": "string", "required": False, "description": "Root cause analysis"}
        }
    },
    "optimize_query": {
        "description": "Submit an optimized version of the query",
        "payload_fields": {
            "optimized_query":     {"type": "string", "required": True,  "description": "Optimized SQL"},
            "optimization_type":   {"type": "string", "required": True,  "description": "What optimization was applied"},
            "expected_improvement":{"type": "string", "required": False, "description": "Expected performance gain"},
            "explanation":         {"type": "string", "required": False, "description": "Why this optimization works"},
            "confidence":          {"type": "float",  "required": False, "description": "Confidence 0.0-1.0"}
        }
    },
    # ── Round 2 actions ──────────────────────────────────────────
    "inspect_query": {
        "description": "EXPLAIN a slow query — reveals scan type, rows examined, index usage",
        "payload_fields": {
            "query_id": {"type": "string", "required": True, "description": "ID of slow query to inspect (e.g. 'q1')"}
        }
    },
    "analyze_indexes": {
        "description": "Show all indexes on a table + usage frequency + missing index hints",
        "payload_fields": {
            "table": {"type": "string", "required": True, "description": "Table name to analyze"}
        }
    },
    "create_index": {
        "description": "Add a composite index on specified columns — core optimization action",
        "payload_fields": {
            "table":   {"type": "string",      "required": True, "description": "Table to index"},
            "columns": {"type": "list|string", "required": True, "description": "Columns to index (list or comma-separated string)"}
        }
    },
    "rewrite_query": {
        "description": "Submit a rewritten SQL query — system evaluates execution time improvement",
        "payload_fields": {
            "query_id": {"type": "string", "required": True, "description": "ID of query to rewrite"},
            "new_sql":  {"type": "string", "required": True, "description": "Rewritten SQL query"}
        }
    },
    "add_column": {
        "description": "Add a denormalization column to reduce expensive JOINs",
        "payload_fields": {
            "table":   {"type": "string", "required": True,  "description": "Table to modify"},
            "column":  {"type": "string", "required": True,  "description": "New column name"},
            "purpose": {"type": "string", "required": False, "description": "Why this column helps"}
        }
    },
    "drop_index": {
        "description": "Remove an unused index to reduce write overhead",
        "payload_fields": {
            "table":      {"type": "string", "required": True, "description": "Table name"},
            "index_name": {"type": "string", "required": True, "description": "Index name to drop (cannot drop PRIMARY)"}
        }
    },
    "partition_table": {
        "description": "Partition a large table by date or ID range for range query efficiency",
        "payload_fields": {
            "table":          {"type": "string", "required": True,  "description": "Table to partition"},
            "partition_by":   {"type": "string", "required": False, "description": "Column to partition on (e.g. 'created_at')"},
            "partition_type": {"type": "string", "required": False, "description": "RANGE | LIST | HASH"}
        }
    },
    "analyze_statistics": {
        "description": "Update table statistics for query planner accuracy",
        "payload_fields": {
            "table": {"type": "string", "required": True, "description": "Table to analyze"}
        }
    },
    "submit_report": {
        "description": "TERMINAL: Submit final optimization report — ends episode, computes full score",
        "payload_fields": {
            "summary":       {"type": "string", "required": True,  "description": "Summary of optimizations applied"},
            "actions_taken": {"type": "list",   "required": False, "description": "List of key actions taken"},
            "expected_gain": {"type": "string", "required": False, "description": "Expected performance improvement"}
        }
    },
}


# ─────────────────────────────────────────────
#  TASK MANAGER
# ─────────────────────────────────────────────

class TaskManager:
    """
    Manages task selection for both Round 1 and Round 2 scenarios.
    Round 2 scenarios have tables/slow_queries structure.
    Round 1 cases have buggy_query structure.
    """

    def __init__(self):
        self._used_ids: set[str] = set()

    def get_task(self, difficulty: DifficultyLevel, task_id: str | None = None) -> dict:
        """
        Returns a task for the given difficulty.
        Prefers Round 2 scenarios, falls back to Round 1 cases.
        """
        pool = ALL_CASES[difficulty]

        if task_id:
            for case in pool:
                if case["id"] == task_id:
                    return case
            raise ValueError(f"Task '{task_id}' not found in {difficulty} pool")

        # Avoid recently used tasks
        available = [c for c in pool if c["id"] not in self._used_ids]
        if not available:
            self._used_ids.clear()
            available = pool

        task = random.choice(available)
        self._used_ids.add(task["id"])
        return task

    def get_random_task(self) -> dict:
        difficulty = random.choice(list(DifficultyLevel))
        return self.get_task(difficulty)

    def get_scenario(self, difficulty: DifficultyLevel, scenario_id: str | None = None) -> dict:
        """Get Round 2 scenario specifically."""
        pool = SCENARIO_ONLY[difficulty]
        if scenario_id:
            for s in pool:
                if s["id"] == scenario_id:
                    return s
            raise ValueError(f"Scenario '{scenario_id}' not found")
        return random.choice(pool)

    def build_observation_context(self, task: dict) -> dict:
        """
        Builds current_context for the Observation.
        Handles both Round 2 scenario format and Round 1 case format.
        CRITICAL: Never leaks ground truth (fixed_query / optimal_actions).
        """
        # ── Round 2 scenario format ───────────────────────────────
        if "slow_queries" in task:
            return {
                "scenario_id":          task["id"],
                "description":          task.get("description", ""),
                "tables":               task.get("tables", []),
                "slow_queries":         task.get("slow_queries", []),
                "performance_score_baseline": task.get("performance_score_baseline", 0.0),
                "target_score":         task.get("target_score", 85.0),
                "max_steps":            task.get("max_steps", 50),
                "category":             task.get("category", ""),
                # Do NOT include missing_index_hints (that's the answer)
                # Do NOT include optimal_actions (that's the answer)
            }

        # ── Round 1 case format (backward compatible) ────────────
        context = {
            "buggy_query":     task.get("buggy_query", ""),
            "error_message":   task.get("error_message", ""),
            "database_schema": task.get("database_schema", ""),
            "error_type_hint": task.get("error_type", ""),
            "category":        task.get("category", ""),
            "estimated_steps": task.get("estimated_fix_steps", 5),
        }
        if task.get("performance_issue"):
            context["performance_issue"] = {
                "type":   task["performance_issue"]["type"],
                "impact": task["performance_issue"]["impact"],
            }
        if task.get("expected_output") and isinstance(task["expected_output"], list):
            context["expected_output_sample"] = task["expected_output"][:1]
        return context

    def get_hint(self, task: dict, hint_number: int) -> str:
        """Progressive hints. Each hint reveals more info. Costs -0.10 each."""
        # Round 2 scenario hints
        if "slow_queries" in task:
            hints = [
                f"Hint 1: Start by inspecting your slow queries with inspect_query action.",
                f"Hint 2: Use analyze_indexes on tables appearing in slow queries.",
                f"Hint 3: Category is '{task.get('category', 'indexing')}'. Target score: {task.get('target_score', 85.0)}.",
            ]
        else:
            # Round 1 hints
            hints = [
                f"Hint 1: The error is in the {task.get('error_location', 'query')}.",
                f"Hint 2: This is a {task.get('error_type', 'unknown')} error. Category: {task.get('category')}.",
                f"Hint 3: Fix: {task.get('fix_description', 'Review the query carefully.')}",
            ]
        idx = min(hint_number - 1, len(hints) - 1)
        return hints[max(0, idx)]

    def list_all_tasks(self) -> list[TaskInfo]:
        """Returns TaskInfo list for the /tasks endpoint — all 30 tasks."""
        result = []
        for difficulty, cases in ALL_CASES.items():
            for case in cases:
                result.append(TaskInfo(
                    id            = case["id"],
                    difficulty    = difficulty,
                    description   = case.get("description", ""),
                    action_schema = ACTION_SCHEMA
                ))
        return result

    def get_ground_truth(self, task_id: str) -> dict | None:
        """Returns full task including ground truth (used by grader only)."""
        for cases in ALL_CASES.values():
            for case in cases:
                if case["id"] == task_id:
                    return case
        return None


# Singleton instance
task_manager = TaskManager()
