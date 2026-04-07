"""
CodeReviewEnvironment — OpenEnv-compliant RL environment for code review.

Inherits from openenv.core.env_server.Environment and implements the
standard reset() / step() / state API. Runs as a FastAPI server inside
Docker; agents interact via HTTP/WebSocket through a typed client.

This is a Semantic Markov Decision Process (S-MDP) where:
  - States are PR diffs + review context (text)
  - Actions are review decisions (labels, orderings, comments)
  - Transitions are deterministic (next PR in queue)
  - Rewards are computed by deterministic graders
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from env.data_generator import DataGenerator, PR_TEMPLATES, get_ground_truth, _build_observation
from graders.grader_easy import EasyGrader
from graders.grader_medium import MediumGrader
from graders.grader_hard import HardGrader
from tasks.task_easy import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask


class CodeReviewEnvironment(
    Environment[CodeReviewAction, CodeReviewObservation, CodeReviewState]
):
    """OpenEnv-compliant code review RL environment.

    Three difficulty levels — easy (severity labeling), medium (queue
    prioritization), hard (feedback generation) — each with deterministic
    graders and exploit-prevention penalties.

    Usage via OpenEnv client:
        async with CodeReviewEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(seed=42)
            result = await env.step(CodeReviewAction(action_type="label_severity", severity="high"))
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task: str = "easy", seed: int = 42):
        super().__init__()
        self.task_name = task
        self.seed = seed
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._trajectory: List[Dict[str, Any]] = []
        self._reviewed_prs: List[str] = []
        self._current_obs: Optional[CodeReviewObservation] = None

        # Initialize task + grader and auto-reset to valid state
        self._init_task(task, seed)
        self._auto_reset(task, seed)

    def _init_task(self, task: str, seed: int) -> None:
        """Initialize the task and grader for the given difficulty."""
        if task == "easy":
            self.task = EasyTask(seed=seed)
            self.grader = EasyGrader()
        elif task == "medium":
            self.task = MediumTask(seed=seed)
            self.grader = MediumGrader()
        elif task == "hard":
            self.task = HardTask(seed=seed)
            self.grader = HardGrader()
        else:
            raise ValueError(f"Unknown task: {task}. Must be easy|medium|hard")

    def _auto_reset(self, task: str, seed: int) -> None:
        """Auto-reset to ensure environment starts in a valid state.

        Called from __init__ so that even without an explicit reset(),
        the environment has episode data loaded for step().
        """
        self._step_count = 0
        self._total_reward = 0.0
        self._trajectory = []
        self._reviewed_prs = []
        self.grader.reset()
        internal_obs = self.task.reset()
        self._current_obs = self._convert_observation(internal_obs, done=False, reward=0.0)

    # ─── OpenEnv API ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        """Reset the environment and return the initial observation.

        Args:
            seed: Random seed for reproducible episodes
            episode_id: Custom episode identifier
            **kwargs: May include 'task' to change difficulty

        Returns:
            CodeReviewObservation with the first PR to review
        """
        # Allow changing task on reset
        task = kwargs.get("task", self.task_name)
        actual_seed = seed if seed is not None else self.seed

        self.task_name = task
        self.seed = actual_seed
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._trajectory = []
        self._reviewed_prs = []

        self._init_task(task, actual_seed)
        self.grader.reset()

        # Get initial observation from task
        internal_obs = self.task.reset()
        self._current_obs = self._convert_observation(internal_obs, done=False, reward=0.0)
        return self._current_obs

    def step(
        self,
        action: CodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        """Execute one step in the environment.

        Args:
            action: CodeReviewAction with the agent's decision
            timeout_s: Optional timeout (unused)

        Returns:
            CodeReviewObservation with next PR, reward, and done flag
        """
        from env.models import Action as InternalAction

        # Convert OpenEnv action to internal action
        internal_action = InternalAction(
            action_type=action.action_type,
            severity=action.severity,
            priority_order=action.priority_order,
            comment=action.comment,
            target_file=action.target_file,
            target_line=action.target_line,
        )

        # Grade the action
        reward_value, reward_breakdown, info, done = self._grade_action(
            internal_action, self._step_count
        )

        # Record trajectory
        prev_obs = self._current_obs
        self._trajectory.append({
            "step": self._step_count,
            "observation": prev_obs.model_dump() if prev_obs else {},
            "action": action.model_dump(),
            "reward": reward_value,
            "info": info,
        })

        self._total_reward += reward_value
        self._step_count += 1

        # Determine done: for hard task, delegate to task's is_done()
        if self.task_name == "hard":
            done = done or self.task.is_done()
        else:
            done = done or self._step_count >= self._get_episode_length()

        # Get next observation
        if not done:
            next_internal_obs = self.task.get_observation(self._step_count)
            self._current_obs = self._convert_observation(
                next_internal_obs,
                done=False,
                reward=reward_value,
                reward_breakdown=reward_breakdown,
                info=info,
            )
        else:
            done = True
            # Return final observation with done=True
            try:
                final_obs = self.task.get_observation(self._step_count)
            except Exception:
                final_obs = self.task.get_observation(
                    max(0, self._step_count - 1)
                )
            self._current_obs = self._convert_observation(
                final_obs,
                done=True,
                reward=reward_value,
                reward_breakdown=reward_breakdown,
                info=info,
            )

        # Track reviewed PRs
        if hasattr(self.task, 'get_current_pr_id'):
            try:
                pr_id = self.task.get_current_pr_id(
                    self._step_count - 1 if self.task_name != "hard" else None
                )
            except TypeError:
                pr_id = self.task.get_current_pr_id()
            if pr_id not in self._reviewed_prs:
                self._reviewed_prs.append(pr_id)

        return self._current_obs

    @property
    def state(self) -> CodeReviewState:
        """Get the current environment state."""
        return CodeReviewState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task=self.task_name,
            seed=self.seed,
            reviewed_prs=list(self._reviewed_prs),
            pending_prs=[],
            total_reward=self._total_reward,
            trajectory=list(self._trajectory),
        )

    def get_metadata(self) -> EnvironmentMetadata:
        """Return environment metadata for the OpenEnv framework."""
        return EnvironmentMetadata(
            name="CodeReviewEnv",
            description=(
                "A Semantic MDP environment for code review. "
                "Agents review pull requests across three difficulty levels: "
                "severity labeling (easy), queue prioritization (medium), "
                "and feedback generation (hard)."
            ),
            version="1.0.0",
            author="CodeReviewEnv Team",
        )

    # ─── Internal helpers ────────────────────────────────────────────────

    def _get_episode_length(self) -> int:
        """Get the episode length for the current task."""
        if self.task_name == "easy":
            return 5
        elif self.task_name == "medium":
            return 3
        elif self.task_name == "hard":
            return 18  # Max steps (3 PRs × 6 actions max: 5 comments + 1 decision)
        return 5

    def _grade_action(self, action, step: int):
        """Grade an action using the appropriate grader."""
        if self.task_name == "easy":
            return self._grade_easy(action, step)
        elif self.task_name == "medium":
            return self._grade_medium(action, step)
        elif self.task_name == "hard":
            return self._grade_hard(action, step)
        return 0.0, {}, {}, True

    def _grade_easy(self, action, step: int):
        """Grade severity labeling action."""
        pr_id = self.task.get_current_pr_id(step)
        reward_obj, info = self.grader.grade(action, pr_id)
        info["ground_truth"] = self.task.get_ground_truth(step)
        done = (step + 1) >= self.task.EPISODE_LENGTH
        return reward_obj.value, reward_obj.breakdown, info, done

    def _grade_medium(self, action, step: int):
        """Grade queue prioritization action."""
        queue_templates = self.task.get_queue_templates(step)
        gt_order = self.task.get_ground_truth_order(step)
        reward_obj, info = self.grader.grade(action, queue_templates, gt_order)
        done = (step + 1) >= self.task.EPISODE_LENGTH
        return reward_obj.value, reward_obj.breakdown, info, done

    def _grade_hard(self, action, step: int):
        """Grade feedback generation action.

        Hard task has per-PR multi-step grading:
        - add_comment: accumulates comments, returns decaying ack reward
        - approve/request_changes: triggers full PR grading via grade_pr()

        Anti-exploit: consecutive comments get decaying reward (0.05 → 0.03 → 0.01)
        to prevent the spam-then-decide loop.
        """
        pr_id = self.task.get_current_pr_id()

        if action.action_type == "add_comment":
            # Accumulate comment for later scoring
            self.grader.add_comment(pr_id, action)
            self.grader.consecutive_comments += 1

            # Process action — updates task state (comment count increments)
            self.task.process_action(action.action_type)

            # Decaying ack reward: penalize consecutive comments without decision
            base_ack = 0.05
            spam_penalty = 0.02 * max(0, self.grader.consecutive_comments - 1)
            ack_reward = max(0.01, base_ack - spam_penalty)

            info = {
                "comment_added": True,
                "pr_id": pr_id,
                "comments_so_far": self.task.comments_on_current_pr,
                "consecutive_comments": self.grader.consecutive_comments,
            }
            done = self.task.is_done()

            # IMPORTANT: Rebuild observation so comment count updates in state
            # (fixes frozen observation bug — Problem 3)
            if not done:
                next_obs = self.task.get_observation(self._step_count)
                self._current_obs = self._convert_observation(
                    next_obs, done=False, reward=ack_reward,
                    reward_breakdown={"comment_ack": ack_reward},
                    info=info,
                )

            return ack_reward, {"comment_ack": ack_reward}, info, done

        elif action.action_type in ("approve", "request_changes"):
            # Reset consecutive comment counter on decision
            self.grader.consecutive_comments = 0

            # Score all accumulated comments + decision
            reward_obj, info = self.grader.grade_pr(pr_id, action.action_type)
            # Advance to next PR
            self.task.process_action(action.action_type)
            done = self.task.is_done()
            return reward_obj.value, reward_obj.breakdown, info, done

        else:
            # Invalid action for hard task — penalize
            info = {"error": f"Invalid action type: {action.action_type}"}
            return 0.01, {"invalid_action": -0.01}, info, False

    def _convert_observation(
        self,
        internal_obs,
        done: bool,
        reward: float,
        reward_breakdown: Optional[Dict] = None,
        info: Optional[Dict] = None,
    ) -> CodeReviewObservation:
        """Convert an internal Observation to a CodeReviewObservation."""
        return CodeReviewObservation(
            done=done,
            reward=reward,
            metadata={
                "task": self.task_name,
                "episode_id": self._episode_id,
                "step": self._step_count,
            },
            pr_id=internal_obs.pr_id,
            title=internal_obs.title,
            description=internal_obs.description,
            author_experience=internal_obs.author_experience,
            files=[f.model_dump() if hasattr(f, 'model_dump') else f for f in internal_obs.files],
            existing_comments=internal_obs.existing_comments,
            review_queue=internal_obs.review_queue,
            step_number=internal_obs.step_number,
            episode_budget=internal_obs.episode_budget,
            reward_breakdown=reward_breakdown,
            info=info,
        )

    # ─── Compatibility methods ───────────────────────────────────────────

    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM agents."""
        return self.task.get_system_prompt()

    def export_trajectory(self) -> List[Dict]:
        """Export full trajectory for MBRL research."""
        return list(self._trajectory)
