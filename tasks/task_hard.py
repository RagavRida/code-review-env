"""
Hard Task — Actionable Feedback Generation

Objective: Agent reviews PRs, adds comments, then approves or requests changes.
Episode length: 3 PRs
Agent may make up to 5 add_comment actions per PR before approve/request_changes.
Required actions: add_comment (multiple), then approve or request_changes.

This is the most challenging task — requires understanding code semantics,
identifying bug locations, generating specific feedback, and making
appropriate review decisions. The five-component grader ensures agents
can't game the score with superficial comments.
"""

from typing import Dict, List, Optional
from env.data_generator import DataGenerator, _build_observation
from env.models import Observation


class HardTask:
    """
    Task configuration for feedback generation.

    Generates episodes of 3 PRs requiring detailed review.
    Each PR allows up to 5 add_comment actions before a final
    approve/request_changes decision.
    """

    TASK_NAME = "hard"
    EPISODE_LENGTH = 3  # number of PRs per episode
    MAX_COMMENTS_PER_PR = 5
    REQUIRED_ACTIONS = {"add_comment", "approve", "request_changes"}

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = DataGenerator(seed=seed)
        self.episode_prs: List[Dict] = []
        self.current_pr_index: int = 0
        self.comments_on_current_pr: int = 0

    def reset(self) -> Observation:
        """Generate a new episode and return first observation."""
        self.episode_prs = self.generator.generate_hard_episode(
            num_prs=self.EPISODE_LENGTH,
        )
        self.current_pr_index = 0
        self.comments_on_current_pr = 0
        return self._get_observation()

    def get_observation(self, step: int = -1) -> Observation:
        """Get observation for the current PR being reviewed."""
        return self._get_observation()

    def _get_observation(self) -> Observation:
        """Build observation from current PR template."""
        if self.current_pr_index >= len(self.episode_prs):
            idx = len(self.episode_prs) - 1
        else:
            idx = self.current_pr_index

        template = self.episode_prs[idx]
        remaining_ids = [t["pr_id"] for t in self.episode_prs[idx + 1:]]

        return _build_observation(
            template=template,
            step_number=self.current_pr_index,
            episode_budget=self.EPISODE_LENGTH - self.current_pr_index,
            review_queue=remaining_ids,
            existing_comments=[
                f"Comment {i+1} on this PR" for i in range(self.comments_on_current_pr)
            ] if self.comments_on_current_pr > 0 else [],
        )

    def process_action(self, action_type: str) -> bool:
        """
        Process an action and return whether we advance to next PR.

        add_comment: increments counter, stays on current PR
        approve/request_changes: advances to next PR

        Returns True if we moved to the next PR.
        """
        if action_type == "add_comment":
            self.comments_on_current_pr += 1
            # Auto-advance if hit comment limit
            if self.comments_on_current_pr >= self.MAX_COMMENTS_PER_PR:
                return self._advance_pr()
            return False
        elif action_type in ("approve", "request_changes"):
            return self._advance_pr()
        return False

    def _advance_pr(self) -> bool:
        """Move to next PR in the episode."""
        self.current_pr_index += 1
        self.comments_on_current_pr = 0
        return True

    def get_current_pr_id(self) -> str:
        """Get the PR ID currently being reviewed."""
        idx = min(self.current_pr_index, len(self.episode_prs) - 1)
        return self.episode_prs[idx]["pr_id"]

    def get_current_template(self) -> Dict:
        """Get full template for current PR."""
        idx = min(self.current_pr_index, len(self.episode_prs) - 1)
        return self.episode_prs[idx]

    def is_done(self) -> bool:
        """Check if episode is complete (all PRs reviewed)."""
        return self.current_pr_index >= self.EPISODE_LENGTH

    def get_total_steps(self) -> int:
        """
        Get total steps in this episode.

        Hard task is variable-length: each PR can have 1-6 actions
        (up to 5 comments + 1 decision). Max steps = 3 * 6 = 18.
        """
        return self.EPISODE_LENGTH * (self.MAX_COMMENTS_PER_PR + 1)

    def get_system_prompt(self) -> str:
        """Return system prompt for LLM agents on this task."""
        return (
            "You are a senior software engineer performing code review.\n"
            "Add review comments, then approve or request changes.\n"
            "For comments respond with:\n"
            '{"action_type": "add_comment", "comment": "<your comment>", '
            '"target_file": "<filename>", "target_line": <line_number>}\n'
            "To finish respond with:\n"
            '{"action_type": "request_changes"} or {"action_type": "approve"}\n'
            "No explanation. JSON only."
        )
