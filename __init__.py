"""
CodeReviewEnv — MCP tool-calling RL environment for automated code review.
"""

from typing import Any

try:
    from .models import CodeReviewAction, CodeReviewObservation, CodeReviewState, MCPAction, MCPObservation
except ImportError:
    from models import CodeReviewAction, CodeReviewObservation, CodeReviewState, MCPAction, MCPObservation

__all__ = [
    "CodeReviewAction",
    "CodeReviewObservation",
    "CodeReviewState",
    "CodeReviewEnv",
    "MCPAction",
    "MCPObservation",
]


def __getattr__(name: str) -> Any:
    if name == "CodeReviewEnv":
        try:
            from .client import CodeReviewEnv as _CodeReviewEnv
        except ImportError:
            from client import CodeReviewEnv as _CodeReviewEnv
        return _CodeReviewEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
