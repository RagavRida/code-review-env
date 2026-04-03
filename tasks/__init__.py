"""
Tasks for CodeReviewEnv — three difficulty levels.

- easy: Severity labeling (5 PRs per episode)
- medium: Queue prioritization (3 queues per episode)
- hard: Actionable feedback generation (3 PRs, multi-action)
"""

from tasks.task_easy import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask

__all__ = ["EasyTask", "MediumTask", "HardTask"]
