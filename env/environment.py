"""
env/environment.py — SQL Database Engineer Agent (SDEA)
Round 2: Long-horizon DB optimization environment.
Agent manages a simulated production database over 50 steps.
"""

import time
import random
from typing import Optional
from pydantic import ValidationError

from env.models import (
    Action, Observation, Reward, EpisodeState,
    DifficultyLevel, ActionType, StepResponse
)
from env.tasks import task_manager
from env.reward import compute_reward, is_done, MAX_STEPS
from env.db_simulator import DatabaseSimulator


class SQLDebuggerEnvironment:
    """
    OpenEnv-compliant SQL Database Engineer Agent Environment.

    Round 2 evolution:
    - 50-step long-horizon episodes (up from 20)
    - 10 action types including DB-specific actions
    - DatabaseSimulator tracks real performance score 0-100
    - Milestone bonuses at 25%/50%/75% improvement
    - Backward compatible with Round 1 actions
    """

    def __init__(self):
        self._state             = EpisodeState()
        self._current_task      = None
        self._started_at        = None
        self._db_sim: Optional[DatabaseSimulator] = None
        self._milestones_earned: set  = set()
        self._baseline_score:   float = 0.0

    # ─────────────────────────────────────────────
    #  reset() → Observation
    # ─────────────────────────────────────────────

    def reset(self, difficulty: Optional[str] = None, task_id: Optional[str] = None) -> Observation:
        """
        Starts a fresh episode. Clears ALL state.
        Loads scenario and initializes DatabaseSimulator.
        """

        # ── Resolve difficulty ────────────────────────────────────
        if difficulty is not None:
            try:
                diff_enum = DifficultyLevel(difficulty.lower())
            except ValueError:
                diff_enum = random.choice(list(DifficultyLevel))
        else:
            diff_enum = random.choice(list(DifficultyLevel))

        # ── Load task ─────────────────────────────────────────────
        try:
            task = task_manager.get_task(diff_enum, task_id=task_id)
        except Exception as e:
            raise ValueError(f"Failed to load task: {str(e)}")

        # ── Initialize DatabaseSimulator ──────────────────────────
        # Only initialize for Round 2 scenarios (have 'tables' key)
        if "tables" in task and "slow_queries" in task:
            self._db_sim         = DatabaseSimulator(task)
            self._baseline_score = self._db_sim.get_performance_score()
        else:
                # Round 1 task — no DB simulator needed
            self._db_sim         = None
            self._baseline_score = 0.0
            self._milestones_earned = set()

        # ── Reset episode state ───────────────────────────────────
        self._current_task = task
        self._started_at   = time.time()
        self._state        = EpisodeState(
            task_id          = task["id"],
            difficulty       = diff_enum,
            step_count       = 0,
            total_reward     = 0.0,
            done             = False,
            hints_used       = 0,
            previous_actions = [],
            action_counts    = {
                "_baseline_score": self._baseline_score,
                "_target_score":   task.get("target_score", 85.0),
                "_milestones":     [],
                "_perf_history":   [self._baseline_score],
                "_best_score":     self._baseline_score,
            },
            started_at       = self._started_at,
            last_reward      = 0.0,
            initialized      = True,
        )

        return self._build_observation()

    # ─────────────────────────────────────────────
    #  step() → StepResponse
    # ─────────────────────────────────────────────

    def step(self, action: Optional[Action]) -> StepResponse:
        """
        Processes an action, updates DB simulator, computes reward.
        Handles all Round 2 DB engineering actions.
        """

        # ── Auto-reset if not initialized ────────────────────────
        if not self._state.initialized or self._current_task is None:
            obs = self.reset()
            return StepResponse(
                observation = obs,
                reward      = Reward(score=0.5, breakdown={"auto_reset": True}, feedback="Environment auto-reset."),
                done        = False,
                info        = {"auto_reset": True}
            )

        # ── Episode already done ──────────────────────────────────
        if self._state.done:
            obs = self._build_observation()
            return StepResponse(
                observation = obs,
                reward      = Reward(score=0.5, breakdown={"episode_done": True}, feedback="Episode finished. Call reset()."),
                done        = True,
                info        = {"episode_done": True, "total_reward": self._state.total_reward}
            )

        # ── Handle null action ────────────────────────────────────
        if action is None or action.payload is None:
            self._state.step_count += 1
            obs    = self._build_observation()
            reward = Reward(score=0.001, breakdown={"invalid_action": 0.001}, feedback="Null action.")
            done   = self._state.step_count >= MAX_STEPS
            self._state.done = done
            return StepResponse(observation=obs, reward=reward, done=done, info={"error": "null_action"})

        action_type_val  = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        action_type_enum = action.action_type

        # ── Update step count ─────────────────────────────────────
        self._state.step_count += 1
        self._state.previous_actions.append(action_type_val)
        self._state.action_counts[action_type_val] = \
            self._state.action_counts.get(action_type_val, 0) + 1

        # ── Handle hints ──────────────────────────────────────────
        if action_type_enum == ActionType.REQUEST_HINT:
            self._state.hints_used += 1
            hint_text = task_manager.get_hint(self._current_task, self._state.hints_used)
            self._current_task["_last_hint"] = hint_text

        # ── Apply DB action and get delta ─────────────────────────
        db_delta      = 0.0
        current_score = self._baseline_score
        action_info   = {}

        if self._db_sim is not None:
            payload = action.payload or {}

            if action_type_enum == ActionType.INSPECT_QUERY:
                qid         = payload.get("query_id", "q1")
                action_info = self._db_sim.inspect_query(qid)
                self._current_task["_last_inspect"] = action_info
                # No score change — investigation action

            elif action_type_enum == ActionType.ANALYZE_INDEXES:
                table       = payload.get("table", "")
                action_info = self._db_sim.analyze_indexes(table)
                self._current_task["_last_analysis"] = action_info

            elif action_type_enum == ActionType.CREATE_INDEX:
                result      = self._db_sim.apply_action("create_index", payload)
                db_delta    = result["delta"]
                action_info = result

            elif action_type_enum == ActionType.REWRITE_QUERY:
                result      = self._db_sim.apply_action("rewrite_query", payload)
                db_delta    = result["delta"]
                action_info = result

            elif action_type_enum == ActionType.ADD_COLUMN:
                result      = self._db_sim.apply_action("add_column", payload)
                db_delta    = result["delta"]
                action_info = result

            elif action_type_enum == ActionType.DROP_INDEX:
                result      = self._db_sim.apply_action("drop_index", payload)
                db_delta    = result["delta"]
                action_info = result

            elif action_type_enum == ActionType.PARTITION_TABLE:
                result      = self._db_sim.apply_action("partition_table", payload)
                db_delta    = result["delta"]
                action_info = result

            elif action_type_enum == ActionType.ANALYZE_STATS:
                result      = self._db_sim.apply_action("analyze_statistics", payload)
                db_delta    = result["delta"]
                action_info = result

            current_score = self._db_sim.get_performance_score()

            # Update tracking in action_counts dict (used by /progress)
            perf_history = self._state.action_counts.get("_perf_history", [])
            perf_history.append(current_score)
            self._state.action_counts["_perf_history"] = perf_history
            self._state.action_counts["_best_score"]   = self._db_sim.best_score

        # ── Compute reward ────────────────────────────────────────
        reward = compute_reward(
            action            = action,
            task_id           = self._state.task_id,
            difficulty        = self._state.difficulty,
            step_count        = self._state.step_count,
            previous_actions  = self._state.previous_actions[:-1],
            hints_used        = self._state.hints_used,
            estimated_steps   = self._current_task.get("estimated_fix_steps", MAX_STEPS),
            action_counts     = self._state.action_counts,
            db_delta          = db_delta,
            baseline_score    = self._baseline_score,
            current_score     = current_score,
            milestones_earned = self._milestones_earned,
        )

        # Update milestone tracking
        self._state.action_counts["_milestones"] = list(self._milestones_earned)

        # ── Update cumulative reward ──────────────────────────────
        self._state.last_reward  = reward.score
        self._state.total_reward = round(self._state.total_reward + reward.score, 4)

        # ── Check done ────────────────────────────────────────────
        target_reached = (
            self._db_sim.is_target_reached() if self._db_sim else False
        )
        done = is_done(
            action_type     = action_type_enum,
            step_count      = self._state.step_count,
            grader_score    = reward.breakdown.get("grader_score", 0.0),
            target_reached  = target_reached,
        )
        self._state.done = done

        # ── Build observation ─────────────────────────────────────
        obs = self._build_observation()

        # ── Info dict ─────────────────────────────────────────────
        info = {
            "step_count":       self._state.step_count,
            "total_reward":     self._state.total_reward,
            "hints_used":       self._state.hints_used,
            "task_id":          self._state.task_id,
            "difficulty":       self._state.difficulty.value if self._state.difficulty else None,
            "performance_score": current_score,
            "db_delta":         db_delta,
            "milestones":       list(self._milestones_earned),
            "action_result":    action_info,
        }
        if done:
            info["episode_summary"] = {
                "total_steps":       self._state.step_count,
                "total_reward":      self._state.total_reward,
                "hints_used":        self._state.hints_used,
                "duration_sec":      round(time.time() - (self._started_at or time.time()), 2),
                "final_score":       current_score,
                "baseline_score":    self._baseline_score,
                "improvement":       round(current_score - self._baseline_score, 2),
                "milestones_earned": list(self._milestones_earned),
            }

        # Normalize reward for validator compliance
        normalized_score = max(0.001, min(0.999, (reward.score + 1.0) / 2.0))
        reward = Reward(
            score=normalized_score,
            breakdown=reward.breakdown,
            feedback=reward.feedback
        )

        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    # ─────────────────────────────────────────────
    #  state() → EpisodeState
    # ─────────────────────────────────────────────

    def state(self) -> EpisodeState:
        return self._state

    # ─────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        """Builds Observation from current state + DB simulator state."""

        if self._current_task is None:
            return Observation(
                task_id          = "none",
                task_description = "No task loaded. Call reset() first.",
                current_context  = {},
                step_count       = self._state.step_count,
                difficulty       = DifficultyLevel.EASY,
                max_steps        = MAX_STEPS,
                hints_used       = self._state.hints_used,
                previous_actions = self._state.previous_actions,
                metadata         = {}
            )

        # Base context from task
        context = task_manager.build_observation_context(self._current_task)

        # Inject DB simulator state
        if self._db_sim is not None:
            db_state = self._db_sim.get_current_state()
            context.update({
                "performance_score":   db_state["performance_score"],
                "target_score":        db_state["target_score"],
                "baseline_score":      db_state["baseline_score"],
                "tables":              db_state["tables"],
                "slow_queries":        db_state["slow_queries"],
                "indexes":             db_state["indexes"],
                "improvement_history": db_state["history"],
                "best_score":          db_state["best_score"],
                "milestones_earned":   list(self._milestones_earned),
            })

        # Inject last action result if available
        if "_last_inspect" in self._current_task:
            context["last_inspect_result"] = self._current_task["_last_inspect"]
        if "_last_analysis" in self._current_task:
            context["last_analysis_result"] = self._current_task["_last_analysis"]
        if "_last_hint" in self._current_task:
            context["last_hint"] = self._current_task["_last_hint"]

        context["steps_remaining"]    = MAX_STEPS - self._state.step_count
        context["total_reward_so_far"] = self._state.total_reward

        return Observation(
            task_id          = self._state.task_id or "none",
            task_description = self._current_task.get("description", ""),
            current_context  = context,
            step_count       = self._state.step_count,
            difficulty       = self._state.difficulty or DifficultyLevel.EASY,
            max_steps        = MAX_STEPS,
            hints_used       = self._state.hints_used,
            previous_actions = self._state.previous_actions.copy(),
            metadata         = {
                "category":         self._current_task.get("category", ""),
                "baseline_score":   self._baseline_score,
                "target_score":     self._current_task.get("target_score", 85.0),
                "total_reward":     self._state.total_reward,
                "milestones":       list(self._milestones_earned),
            }
        )


# ─────────────────────────────────────────────
#  SINGLETON INSTANCE (used by FastAPI)
# ─────────────────────────────────────────────

environment = SQLDebuggerEnvironment()
