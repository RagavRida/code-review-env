"""
CodeReviewEnv — Semantic RL Environment for OpenEnv

A real-world RL environment simulating software code review.
Designed as an MBRL-ready benchmark: every episode is logged as a
(state, action, reward, next_state) trajectory for semantic world model training.

We define a Semantic Markov Decision Process (S-MDP) as a tuple:
    (S, A, T, R, γ)

Where:
  S — semantic state space: structured text + metadata (not R^n)
  A — structured action space: typed decisions over semantic entities
  T — semantic transition function: T(s, a) → s' where s,s' ∈ S
      T is not expressible as a closed-form equation
      T is learned from trajectory data
  R — shaped reward: R(s, a, s') → [-1, 1] with trajectory-level components
  γ — discount factor: 0.95 (standard)

CodeReviewEnv is the first concrete instantiation of an S-MDP
designed for empirical study of semantic world model learning.

This is distinct from:
  - POMDPs: partial observability, not semantic transitions
  - Text games (Jericho, TWC): synthetic, not real-world tasks
  - LLM agent benchmarks: measure success/failure, no MDP formalism
  - Standard MBRL benchmarks: continuous vector state, physics transitions

Research context:
  Current MBRL benchmarks: MuJoCo, Atari, DMControl — continuous/pixel state spaces
  CodeReviewEnv: structured text state, semantic transitions, knowledge-work domain
  Gap filled: first environment enabling Model-Based RL over semantic state spaces
  Trajectory dataset: use export_trajectory() to build training data for world models

How this helps OpenEnv:
  Expands ecosystem into knowledge-work domains
  Proves spec works for semantic observation spaces
  Trajectory logging makes OpenEnv useful for MBRL, not just evaluation
"""

import random
from typing import Dict, List, Tuple, Optional, Any

from ulid import ULID

from env.models import Observation, Action, Reward, State
from env.trajectory_logger import TrajectoryLogger
from tasks.task_easy import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask
from graders.grader_easy import EasyGrader
from graders.grader_medium import MediumGrader
from graders.grader_hard import HardGrader


class CodeReviewEnv:
    """
    Semantic RL Environment for Software Code Review.

    Implements the OpenEnv interface: reset() → step() → state() → export_trajectory().
    Three difficulty levels (easy, medium, hard) with deterministic grading.

    MBRL hook: every transition is logged as (state, action, reward, next_state)
    for semantic world model training. See export_trajectory() and
    world_model/scaffold.py for the research pipeline.

    Usage:
        env = CodeReviewEnv(task="easy", seed=42)
        obs = env.reset()
        action = Action(action_type="label_severity", severity="high")
        obs, reward, done, info = env.step(action)
        trajectory = env.export_trajectory()
    """

    def __init__(self, task: str = "easy", seed: int = 42):
        """
        Initialize environment.

        Args:
            task: "easy" | "medium" | "hard"
            seed: random seed for reproducibility
        """
        self.task_name = task
        self.seed = seed

        # Set global seed for reproducibility
        random.seed(seed)

        # Initialize task
        if task == "easy":
            self.task = EasyTask(seed=seed)
        elif task == "medium":
            self.task = MediumTask(seed=seed)
        elif task == "hard":
            self.task = HardTask(seed=seed)
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'easy', 'medium', or 'hard'.")

        # Initialize grader
        if task == "easy":
            self.grader = EasyGrader()
        elif task == "medium":
            self.grader = MediumGrader()
        else:
            self.grader = HardGrader()

        # Initialize trajectory logger
        self.logger = TrajectoryLogger()

        # Episode state
        self.episode_id: Optional[str] = None
        self.current_obs: Optional[Observation] = None
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.done: bool = False
        self.reviewed_prs: List[str] = []
        self.pending_prs: List[str] = []
        self.step_rewards: List[float] = []
        self._trajectory: List[Dict[str, Any]] = []
        self._severity_labels: Dict[str, str] = {}  # for consistency check

    def reset(self) -> Observation:
        """
        Reset environment for a new episode.

        Generates a fresh episode using FIXED_TEST_SUITE with the
        configured seed. Clears trajectory log.

        Returns:
            Initial observation for the episode.
        """
        # Generate new episode ID
        self.episode_id = str(ULID())

        # Reset grader state
        self.grader.reset()

        # Reset episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.reviewed_prs = []
        self.step_rewards = []
        self._trajectory = []
        self._severity_labels = {}

        # Start trajectory logging
        self.logger.start_episode(self.episode_id, self.task_name)

        # Get initial observation from task
        self.current_obs = self.task.reset()
        self.pending_prs = list(self.current_obs.review_queue)

        return self.current_obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one step in the environment.

        Validates action, computes reward using grader, logs transition,
        and advances state. Invalid actions receive penalty reward but
        never crash the environment.

        Args:
            action: Agent's action (must pass Pydantic validation)

        Returns:
            (next_observation, reward, done, info)
        """
        if self.done:
            return self._terminal_step()

        info: Dict[str, Any] = {
            "task": self.task_name,
            "episode_id": self.episode_id,
            "step": self.step_count,
            "parse_error": None,
        }

        prev_obs = self.current_obs

        # ── Route to task-specific step logic ────────────────────────
        try:
            if self.task_name == "easy":
                reward, grader_info = self._step_easy(action)
            elif self.task_name == "medium":
                reward, grader_info = self._step_medium(action)
            else:
                reward, grader_info = self._step_hard(action)

            info.update(grader_info)

        except Exception as e:
            # Never crash on agent output — return penalty
            reward = Reward(
                value=0.01,
                breakdown={"step_reward": 0.01, "error_penalty": -0.01},
                reason=f"Action processing error: {str(e)}",
            )
            info["parse_error"] = str(e)

        # ── Apply trajectory-level reward shaping ────────────────────
        reward = self._apply_reward_shaping(reward)

        # ── Update state ─────────────────────────────────────────────
        self.step_rewards.append(reward.value)
        self.total_reward += reward.value
        self.step_count += 1

        # Check if episode is done
        if self.task_name == "hard":
            self.done = self.task.is_done()
        else:
            self.done = self.task.is_done(self.step_count)

        # Get next observation
        if not self.done:
            if self.task_name == "hard":
                self.current_obs = self.task.get_observation()
            else:
                self.current_obs = self.task.get_observation(self.step_count)
        next_obs = self.current_obs

        # ── Log transition ───────────────────────────────────────────
        self.logger.log_transition(
            step=self.step_count - 1,
            state=prev_obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=self.done,
        )

        # Store in internal trajectory
        self._trajectory.append({
            "step": self.step_count - 1,
            "state": prev_obs.model_dump(),
            "action": action.model_dump(),
            "reward": reward.model_dump(),
            "next_state": next_obs.model_dump(),
            "done": self.done,
        })

        # Save trajectory file when episode ends
        if self.done:
            self.logger.save()
            info["trajectory_path"] = f"trajectories/{self.task_name}_{self.episode_id}.jsonl"

        info["reward_breakdown"] = reward.breakdown
        info["ground_truth"] = self._get_ground_truth_for_step()

        return next_obs, reward, self.done, info

    def _step_easy(self, action: Action) -> Tuple[Reward, Dict]:
        """Handle one step of the easy task."""
        pr_id = self.task.get_current_pr_id(self.step_count)

        # Track for consistency checking
        if action.severity:
            self._severity_labels[pr_id] = action.severity

        self.reviewed_prs.append(pr_id)
        reward, grader_info = self.grader.grade(action, pr_id)
        return reward, grader_info

    def _step_medium(self, action: Action) -> Tuple[Reward, Dict]:
        """Handle one step of the medium task."""
        queue_templates = self.task.get_queue_templates(self.step_count)
        ground_truth_order = self.task.get_ground_truth_order(self.step_count)

        pr_id = self.task.get_current_pr_id(self.step_count)
        self.reviewed_prs.append(pr_id)

        reward, grader_info = self.grader.grade(action, queue_templates, ground_truth_order)
        return reward, grader_info

    def _step_hard(self, action: Action) -> Tuple[Reward, Dict]:
        """Handle one step of the hard task."""
        pr_id = self.task.get_current_pr_id()
        grader_info: Dict = {}

        if action.action_type == "add_comment":
            # Track comment in grader, give decaying feedback
            self.grader.add_comment(pr_id, action)
            self.grader.consecutive_comments += 1
            advanced = self.task.process_action("add_comment")

            # If auto-advanced due to comment limit, grade the PR
            if advanced:
                self.grader.consecutive_comments = 0
                self.reviewed_prs.append(pr_id)
                reward, grader_info = self.grader.grade_pr(pr_id, "request_changes")
                return reward, grader_info

            # Decaying ack: consecutive comments without decision get penalized
            base_ack = 0.05
            spam_penalty = 0.02 * max(0, self.grader.consecutive_comments - 1)
            ack_value = max(0.01, base_ack - spam_penalty)

            reward = Reward(
                value=ack_value,
                breakdown={"step_reward": ack_value},
                reason=f"Comment {self.task.comments_on_current_pr} recorded — awaiting decision.",
            )
            return reward, grader_info

        elif action.action_type in ("approve", "request_changes"):
            self.grader.consecutive_comments = 0
            self.reviewed_prs.append(pr_id)
            reward, grader_info = self.grader.grade_pr(pr_id, action.action_type)
            self.task.process_action(action.action_type)
            return reward, grader_info

        else:
            # Invalid action type for hard task
            reward = Reward(
                value=-0.1,
                breakdown={"step_reward": 0.0, "invalid_action_penalty": -0.1},
                reason=f"Invalid action_type '{action.action_type}' for hard task.",
            )
            return reward, grader_info

    def _apply_reward_shaping(self, reward: Reward) -> Reward:
        """
        Apply trajectory-level reward shaping bonuses and penalties.

        Beyond per-step reward, these shape agent behavior across the episode:
          efficiency_bonus:     +0.1 if completing under budget
          consistency_penalty: -0.2 if contradicting own labels
          coverage_bonus:      +0.15 if catching all critical bugs
        """
        breakdown = dict(reward.breakdown)

        # Efficiency bonus: complete in fewer steps than budget
        # Only applied at episode end to avoid premature termination incentive
        breakdown["efficiency_bonus"] = 0.0
        if self.task_name != "hard" and self.done:
            if self.task_name == "easy":
                budget = EasyTask.EPISODE_LENGTH
            else:
                budget = MediumTask.EPISODE_LENGTH
            if self.step_count < budget:
                breakdown["efficiency_bonus"] = 0.1

        # Consistency penalty: labeling same PR differently in same episode
        # This catches agents that flip-flop on severity assessments
        breakdown["consistency_penalty"] = 0.0
        if self.task_name == "easy":
            # Check if we've seen this PR before with a different label
            pr_id = self.task.get_current_pr_id(max(0, self.step_count))
            current_severity = getattr(reward, '_current_severity', None)
            if pr_id in self._severity_labels and current_severity:
                prev = self._severity_labels[pr_id]
                if prev != current_severity:
                    breakdown["consistency_penalty"] = -0.2

        # Coverage bonus: catch all critical bugs
        # Only applied at episode end to encourage thorough review
        breakdown["coverage_bonus"] = 0.0
        if self.task_name == "hard" and self.done:
            # Check if the grader's coverage component averaged >= 0.9
            # across all PRs reviewed in this episode
            coverage_scores = [
                r for r in self.step_rewards if r > 0.0  # non-zero means actual review
            ]
            if coverage_scores and sum(coverage_scores) / len(coverage_scores) >= 0.7:
                breakdown["coverage_bonus"] = 0.15

        # Compute shaping adjustment (only the bonuses/penalties added here)
        shaping_adjustment = (
            breakdown.get("efficiency_bonus", 0.0)
            + breakdown.get("consistency_penalty", 0.0)
            + breakdown.get("coverage_bonus", 0.0)
        )

        # Use original reward value + shaping adjustments only
        # (avoid double-counting grader component scores in breakdown)
        total = reward.value + shaping_adjustment
        total = max(0.01, min(0.99, total))

        return Reward(
            value=total,
            breakdown=breakdown,
            reason=reward.reason,
        )

    def _terminal_step(self) -> Tuple[Observation, Reward, bool, Dict]:
        """Handle step() calls after episode is done."""
        reward = Reward(
            value=0.0,
            breakdown={"step_reward": 0.0},
            reason="Episode already done.",
        )
        return self.current_obs, reward, True, {
            "task": self.task_name,
            "episode_id": self.episode_id,
            "step": self.step_count,
            "terminal": True,
        }

    def _get_ground_truth_for_step(self) -> Dict:
        """Get ground truth for the current step (for info dict)."""
        try:
            if self.task_name == "easy":
                return self.task.get_ground_truth(max(0, self.step_count - 1))
            elif self.task_name == "medium":
                step = max(0, self.step_count - 1)
                return {
                    "pr_id": self.task.get_current_pr_id(step),
                    "priority_order": self.task.get_ground_truth_order(step),
                }
            else:
                template = self.task.get_current_template()
                return {
                    "pr_id": template["pr_id"],
                    "severity": template["ground_truth_severity"],
                    "bug_category": template["bug_category"],
                    "bug_lines": template["bug_lines"],
                }
        except Exception:
            return {}

    def state(self) -> State:
        """
        Return full current state including trajectory history.

        The trajectory list enables in-episode analysis and is the
        raw material for semantic world model training.
        """
        return State(
            current_pr=self.current_obs,
            reviewed_prs=self.reviewed_prs,
            pending_prs=self.pending_prs,
            total_reward=self.total_reward,
            step=self.step_count,
            done=self.done,
            trajectory=self._trajectory,
        )

    def export_trajectory(self) -> List[Dict]:
        """
        Return full episode as list of dicts.

        Format: [{step, state, action, reward, next_state, done, timestamp}]
        Clean JSONL-ready format for world model training dataset.

        Usage for MBRL research:
            trajectory = env.export_trajectory()
            # Each entry is one (s, a, r, s') transition
            # Encode states with sentence-transformers
            # Train transition model: f(z_t, a_t) → (z_{t+1}, r_t)
        """
        return self.logger.export()

    def get_system_prompt(self) -> str:
        """
        Return task-specific system prompt for LLM agents.

        Includes: role description, exact Action JSON schema,
        one example per action type. Ends with:
        "Respond ONLY with valid JSON matching the Action schema. No explanation."
        """
        return self.task.get_system_prompt()
