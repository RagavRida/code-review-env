"""
Easy Task — Severity Labeling

Objective: Agent receives one PR per step. Must label bug severity.
Episode length: 5 PRs
Required action: action_type="label_severity", severity=<label>

This is the foundational task — can the agent distinguish between
critical security bugs and style-only changes? Success requires
understanding code semantics, not just surface patterns.
"""

from typing import Dict, List
from env.data_generator import DataGenerator, _build_observation
from env.models import Observation


class EasyTask:
    """
    Task configuration for severity labeling.

    Generates episodes of 5 individual PRs from FIXED_TEST_SUITE.
    Each step presents one PR; the agent must label its severity.
    """

    TASK_NAME = "easy"
    EPISODE_LENGTH = 5
    REQUIRED_ACTION = "label_severity"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = DataGenerator(seed=seed)
        self.episode_prs: List[Dict] = []
        self.current_step: int = 0

    def reset(self) -> Observation:
        """Generate a new episode and return first observation."""
        self.episode_prs = self.generator.generate_easy_episode(self.EPISODE_LENGTH)
        self.current_step = 0
        return self._get_observation(0)

    def get_observation(self, step: int) -> Observation:
        """Get observation for a specific step."""
        return self._get_observation(step)

    def _get_observation(self, step: int) -> Observation:
        """Build observation from template at given step."""
        if step >= len(self.episode_prs):
            # Return last PR if we've gone past — shouldn't happen if done is tracked
            step = len(self.episode_prs) - 1

        template = self.episode_prs[step]
        remaining_ids = [t["pr_id"] for t in self.episode_prs[step + 1:]]

        return _build_observation(
            template=template,
            step_number=step,
            episode_budget=self.EPISODE_LENGTH - step,
            review_queue=remaining_ids,
        )

    def get_current_pr_id(self, step: int) -> str:
        """Get the PR ID for the current step."""
        if step < len(self.episode_prs):
            return self.episode_prs[step]["pr_id"]
        return self.episode_prs[-1]["pr_id"]

    def is_done(self, step: int) -> bool:
        """Check if episode is complete."""
        return step >= self.EPISODE_LENGTH

    def get_ground_truth(self, step: int) -> Dict:
        """Get ground truth for grading at given step."""
        template = self.episode_prs[min(step, len(self.episode_prs) - 1)]
        return {
            "pr_id": template["pr_id"],
            "severity": template["ground_truth_severity"],
            "bug_category": template["bug_category"],
        }

    def get_system_prompt(self) -> str:
        """Return system prompt for LLM agents on this task."""
        return (
            "You are a senior software engineer. You will receive a pull request.\n"
            'Respond ONLY with this JSON:\n'
            '{"action_type": "label_severity", "severity": "<critical|high|medium|low|none>"}\n'
            'Example: {"action_type": "label_severity", "severity": "high"}\n'
            "No explanation. JSON only."
        )
