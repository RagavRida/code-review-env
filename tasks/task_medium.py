"""
Medium Task — Review Queue Prioritization

Objective: Agent receives a queue of 5 PRs. Must order by review priority.
Episode length: 3 queue orderings
Required action: action_type="prioritize", priority_order=[list of pr_ids]

Priority rules (ground truth ordering):
  1. Security PRs (sql_injection, security_vulnerability) always first
  2. By severity: critical > high > medium > low > none
  3. Within same severity: junior authors first (urgency heuristic)
"""

from typing import Dict, List
from env.data_generator import DataGenerator, _build_observation
from env.models import Observation


class MediumTask:
    """
    Task configuration for queue prioritization.

    Generates episodes of 3 queue orderings from FIXED_TEST_SUITE.
    Each step presents a queue of 5 PRs; the agent must order them.
    """

    TASK_NAME = "medium"
    EPISODE_LENGTH = 3
    QUEUE_SIZE = 5
    REQUIRED_ACTION = "prioritize"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = DataGenerator(seed=seed)
        self.episode_queues: List[List[Dict]] = []
        self.current_step: int = 0

    def reset(self) -> Observation:
        """Generate a new episode and return first observation."""
        self.episode_queues = self.generator.generate_medium_episode(
            num_queues=self.EPISODE_LENGTH,
            queue_size=self.QUEUE_SIZE,
        )
        self.current_step = 0
        return self._get_observation(0)

    def get_observation(self, step: int) -> Observation:
        """Get observation for a specific step."""
        return self._get_observation(step)

    def _get_observation(self, step: int) -> Observation:
        """Build observation from queue at given step."""
        if step >= len(self.episode_queues):
            step = len(self.episode_queues) - 1

        queue = self.episode_queues[step]
        # Use first PR in queue as the main observation, with queue IDs
        template = queue[0]
        queue_ids = [t["pr_id"] for t in queue]

        return _build_observation(
            template=template,
            step_number=step,
            episode_budget=self.EPISODE_LENGTH - step,
            review_queue=queue_ids,
        )

    def get_queue_templates(self, step: int) -> List[Dict]:
        """Get full template dicts for the queue at given step."""
        if step < len(self.episode_queues):
            return self.episode_queues[step]
        return self.episode_queues[-1]

    def get_ground_truth_order(self, step: int) -> List[str]:
        """Get ground truth priority ordering for the queue at given step."""
        queue = self.get_queue_templates(step)
        return self.generator.compute_priority_order(queue)

    def get_current_pr_id(self, step: int) -> str:
        """Get the representative PR ID for the current step."""
        if step < len(self.episode_queues):
            return self.episode_queues[step][0]["pr_id"]
        return self.episode_queues[-1][0]["pr_id"]

    def is_done(self, step: int) -> bool:
        """Check if episode is complete."""
        return step >= self.EPISODE_LENGTH

    def get_system_prompt(self) -> str:
        """Return system prompt for LLM agents on this task."""
        return (
            "You are a senior software engineer. You will receive a queue of PRs.\n"
            "Order them by review priority, most urgent first.\n"
            'Respond ONLY with this JSON:\n'
            '{"action_type": "prioritize", "priority_order": ["pr_id_1", "pr_id_2", ...]}\n'
            "No explanation. JSON only."
        )
