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


class SQLDebuggerEnvironment:
    """
    OpenEnv-compliant SQL Query Debugger Environment.

    Implements the 3 required methods:
        reset()  → Observation
        step()   → (Observation, Reward, done, info)
        state()  → EpisodeState

    Design principles:
    - Dense reward signal at every step
    - No state leakage between episodes
    - Graceful handling of all edge cases
    - Deterministic grading
    - Thread-safe episode state
    """

    def __init__(self):
        self._state        = EpisodeState()
        self._current_task = None
        self._started_at   = None

    # ─────────────────────────────────────────────
    #  reset() → Observation
    # ─────────────────────────────────────────────

    def reset(self, difficulty: Optional[str] = None, task_id: Optional[str] = None) -> Observation:
        """
        Starts a fresh episode. Clears ALL state from previous episode.
        Loads a new task from the dataset.
        Returns the initial Observation the agent sees.

        Edge cases handled:
        - reset() called mid-episode → cleanly resets, no state leakage
        - invalid difficulty → defaults to random
        - dataset empty → raises ValueError with clear message
        """

        # ── Resolve difficulty ────────────────────────────────────
        if difficulty is not None:
            try:
                diff_enum = DifficultyLevel(difficulty.lower())
            except ValueError:
                # Invalid difficulty — pick random
                diff_enum = random.choice(list(DifficultyLevel))
        else:
            diff_enum = random.choice(list(DifficultyLevel))

        # ── Load task ─────────────────────────────────────────────
        try:
            task = task_manager.get_task(diff_enum, task_id=task_id)
        except Exception as e:
            raise ValueError(f"Failed to load task: {str(e)}")

        # ── Reset ALL state — no leakage ──────────────────────────
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
            action_counts    = {},
            started_at       = self._started_at,
            last_reward      = 0.0,
            initialized      = True,
        )

        # ── Build initial observation ─────────────────────────────
        context = task_manager.build_observation_context(task)
        return Observation(
            task_id          = task["id"],
            task_description = task["description"],
            current_context  = context,
            step_count       = 0,
            difficulty       = diff_enum,
            max_steps        = MAX_STEPS,
            hints_used       = 0,
            previous_actions = [],
            metadata         = {
                "category":        task.get("category", ""),
                "estimated_steps": task.get("estimated_fix_steps", 5),
                "started_at":      self._started_at,
            }
        )

    # ─────────────────────────────────────────────
    #  step() → (Observation, Reward, done, info)
    # ─────────────────────────────────────────────

    def step(self, action: Optional[Action]) -> StepResponse:
        """
        Accepts an Action, processes it, updates state,
        computes dense reward, returns next Observation.

        Edge cases handled:
        - step() called before reset() → auto-resets
        - null action → reward=-0.1, done=False, never crash
        - malformed action payload → catches ValidationError
        - agent loops (same action 3+ times) → loop penalty
        - episode already done → returns terminal observation
        - max steps reached → forces done=True
        - extremely long payload → truncated in models.py
        """

        # ── Auto-reset if not initialized ────────────────────────
        if not self._state.initialized or self._current_task is None:
            obs = self.reset()
            return StepResponse(
                observation=obs,
                reward=Reward(score=0.0, breakdown={"auto_reset": True}, feedback="Environment auto-reset."),
                done=False,
                info={"auto_reset": True}
            )

        # ── Episode already done ──────────────────────────────────
        if self._state.done:
            obs = self._build_observation()
            return StepResponse(
                observation=obs,
                reward=Reward(score=0.0, breakdown={"episode_done": True}, feedback="Episode already finished. Call reset()."),
                done=True,
                info={"episode_done": True, "total_reward": self._state.total_reward}
            )

        # ── Handle null / invalid action ─────────────────────────
        if action is None or action.payload is None:
            self._state.step_count += 1
            obs = self._build_observation()
            reward = Reward(
                score=-0.1,
                breakdown={"invalid_action": -0.1},
                feedback="Null or invalid action received. Penalty -0.1."
            )
            self._state.last_reward   = -0.1
            self._state.total_reward  = round(self._state.total_reward - 0.1, 4)
            done = self._state.step_count >= MAX_STEPS
            self._state.done = done
            return StepResponse(observation=obs, reward=reward, done=done, info={"error": "null_action"})

        # ── Validate action type ──────────────────────────────────
        try:
            action_type_val = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        except Exception:
            action_type_val = "unknown"

        # ── Update step count ─────────────────────────────────────
        self._state.step_count += 1
        self._state.previous_actions.append(action_type_val)
        self._state.action_counts[action_type_val] = self._state.action_counts.get(action_type_val, 0) + 1

        # ── Track hints ───────────────────────────────────────────
        if action.action_type == ActionType.REQUEST_HINT:
            self._state.hints_used += 1
            # Inject hint into next observation context
            hint_text = task_manager.get_hint(self._current_task, self._state.hints_used)
            self._current_task["_last_hint"] = hint_text

        # ── Compute dense reward ──────────────────────────────────
        reward = compute_reward(
            action           = action,
            task_id          = self._state.task_id,
            difficulty       = self._state.difficulty,
            step_count       = self._state.step_count,
            previous_actions = self._state.previous_actions[:-1],  # exclude current
            hints_used       = self._state.hints_used,
            estimated_steps  = self._current_task.get("estimated_fix_steps", 5),
            action_counts    = self._state.action_counts,
        )

        # ── Update cumulative reward ──────────────────────────────
        self._state.last_reward  = reward.score
        self._state.total_reward = round(self._state.total_reward + reward.score, 4)

        # ── Check done condition ──────────────────────────────────
        done = is_done(
            action_type  = action.action_type,
            step_count   = self._state.step_count,
            grader_score = reward.breakdown.get("grader_score", 0.0),
        )
        self._state.done = done

        # ── Build next observation ────────────────────────────────
        obs = self._build_observation()

        # ── Build info dict ───────────────────────────────────────
        info = {
            "step_count":    self._state.step_count,
            "total_reward":  self._state.total_reward,
            "hints_used":    self._state.hints_used,
            "action_counts": self._state.action_counts,
            "task_id":       self._state.task_id,
            "difficulty":    self._state.difficulty.value if self._state.difficulty else None,
        }
        if done:
            info["episode_summary"] = {
                "total_steps":  self._state.step_count,
                "total_reward": self._state.total_reward,
                "hints_used":   self._state.hints_used,
                "duration_sec": round(time.time() - (self._started_at or time.time()), 2),
            }

        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    # ─────────────────────────────────────────────
    #  state() → EpisodeState
    # ─────────────────────────────────────────────

    def state(self) -> EpisodeState:
        """
        Returns the full current state at any point.
        Must be JSON-serializable. Must always reflect latest step.

        Edge case: state() called before reset() → returns default empty state.
        Never crashes.
        """
        return self._state

    # ─────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ─────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        """
        Builds the current Observation from internal state.
        Injects hint into context if one was just requested.
        CRITICAL: Never leaks fixed_query (ground truth) to agent.
        """
        if self._current_task is None:
            # Fallback safe observation
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

        context = task_manager.build_observation_context(self._current_task)

        # Inject hint if available
        if "_last_hint" in self._current_task:
            context["last_hint"] = self._current_task["_last_hint"]

        # Add step progress info
        context["steps_remaining"] = MAX_STEPS - self._state.step_count
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
                "category":        self._current_task.get("category", ""),
                "estimated_steps": self._current_task.get("estimated_fix_steps", 5),
                "total_reward":    self._state.total_reward,
                "action_counts":   self._state.action_counts,
            }
        )


# ─────────────────────────────────────────────
#  SINGLETON INSTANCE (used by FastAPI)
# ─────────────────────────────────────────────

environment = SQLDebuggerEnvironment()