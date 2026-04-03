"""
CodeReviewEnv — OpenEnv-compliant RL environment for software code review.

Public API:
    from code_review_env import CodeReviewEnv, CodeReviewAction, CodeReviewObservation
"""

try:
    # When installed as a package (pip install -e .)
    from .models import CodeReviewAction, CodeReviewObservation, CodeReviewState
    from .client import CodeReviewEnv
except ImportError:
    # When running from project root (PYTHONPATH=.)
    from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
    from client import CodeReviewEnv

__all__ = [
    "CodeReviewEnv",
    "CodeReviewAction",
    "CodeReviewObservation",
    "CodeReviewState",
]
