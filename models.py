"""
CodeReviewEnv — OpenEnv-compliant typed models.

All models inherit from openenv.core.env_server base types
to ensure full compatibility with the OpenEnv framework.

Action, Observation, State are Pydantic BaseModel subclasses
with automatic serialization and validation.
"""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State


# ─── Action ──────────────────────────────────────────────────────────────────


class CodeReviewAction(Action):
    """Action space for all three CodeReviewEnv tasks.

    Easy:   action_type="label_severity", severity="critical"|"high"|"medium"|"low"|"none"
    Medium: action_type="prioritize", priority_order=["PR-001", "PR-002", ...]
    Hard:   action_type="add_comment"|"approve"|"request_changes"
    """

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(
        ..., description="One of: label_severity, prioritize, add_comment, approve, request_changes"
    )
    severity: Optional[str] = Field(
        default=None, description="Severity label for easy task"
    )
    priority_order: Optional[List[str]] = Field(
        default=None, description="Ordered list of PR IDs for medium task"
    )
    comment: Optional[str] = Field(
        default=None, description="Review comment text for hard task"
    )
    target_file: Optional[str] = Field(
        default=None, description="File path the comment targets"
    )
    target_line: Optional[int] = Field(
        default=None, description="Line number the comment targets"
    )


# ─── Observation ─────────────────────────────────────────────────────────────


class CodeReviewObservation(Observation):
    """Observation returned after reset() and step().

    Inherits done, reward, metadata from openenv Observation.
    Adds code-review-specific fields.
    """

    model_config = ConfigDict(extra="forbid")

    pr_id: str = Field(default="", description="Pull request identifier")
    title: str = Field(default="", description="PR title")
    description: str = Field(default="", description="PR description")
    author_experience: str = Field(default="", description="junior|mid|senior")
    files: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of changed files with diffs"
    )
    existing_comments: List[str] = Field(
        default_factory=list, description="Previous review comments"
    )
    review_queue: List[str] = Field(
        default_factory=list, description="Queue of PR IDs (medium task)"
    )
    step_number: int = Field(default=0, description="Current step in episode")
    episode_budget: int = Field(default=5, description="Steps remaining")
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None, description="Detailed reward component breakdown"
    )
    info: Optional[Dict[str, Any]] = Field(
        default=None, description="Grader info and ground truth"
    )


# ─── State ───────────────────────────────────────────────────────────────────


class CodeReviewState(State):
    """Extended state for CodeReviewEnv.

    Inherits episode_id, step_count from openenv State.
    Adds task tracking and trajectory history.
    """

    task: str = Field(default="easy", description="Current task difficulty")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    reviewed_prs: List[str] = Field(
        default_factory=list, description="PRs already reviewed"
    )
    pending_prs: List[str] = Field(
        default_factory=list, description="PRs remaining in episode"
    )
    total_reward: float = Field(default=0.0, description="Cumulative episode reward")
    trajectory: List[Dict[str, Any]] = Field(
        default_factory=list, description="Full (s,a,r,s') trajectory"
    )
