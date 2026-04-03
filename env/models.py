"""
Pydantic Models for CodeReviewEnv

Defines the complete type system for the Semantic MDP:
  - PRFile: individual file in a pull request
  - Observation: the full state visible to the agent (s ∈ S)
  - Action: the structured decision space (a ∈ A)
  - Reward: shaped reward with component breakdown (R: S×A×S' → [-1,1])
  - State: full environment state including trajectory history

All models are serializable to JSON for trajectory logging and API transport.
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any


class PRFile(BaseModel):
    """A single file within a pull request diff."""
    filename: str
    language: str  # python | javascript | java | go
    diff: str
    lines_changed: int
    has_tests: bool

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        allowed = {"python", "javascript", "java", "go", "rust", "typescript", "ruby"}
        if v not in allowed:
            raise ValueError(f"language must be one of {allowed}")
        return v


class Observation(BaseModel):
    """
    The agent's observation at each step — the semantic state s ∈ S.

    Unlike continuous MBRL state spaces (e.g. MuJoCo joint angles),
    this is structured text carrying semantic meaning: code diffs,
    author context, review history. A world model must learn to
    predict how review actions transform this state.
    """
    pr_id: str
    title: str
    description: str
    author_experience: str  # junior | mid | senior
    files: List[PRFile]
    existing_comments: List[str]
    review_queue: List[str]
    step_number: int
    episode_budget: int

    @field_validator("author_experience")
    @classmethod
    def validate_experience(cls, v: str) -> str:
        allowed = {"junior", "mid", "senior"}
        if v not in allowed:
            raise ValueError(f"author_experience must be one of {allowed}")
        return v


class Action(BaseModel):
    """
    The agent's action — a structured decision a ∈ A.

    The action space is heterogeneous: different action_types require
    different fields. This is fundamentally different from continuous
    action spaces in standard MBRL — it requires structured encoding
    for world model training.
    """
    action_type: str  # label_severity | prioritize | add_comment | approve | request_changes
    severity: Optional[str] = None       # critical | high | medium | low | none
    priority_order: Optional[List[str]] = None
    comment: Optional[str] = None
    target_file: Optional[str] = None
    target_line: Optional[int] = None

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        allowed = {"label_severity", "prioritize", "add_comment", "approve", "request_changes"}
        if v not in allowed:
            raise ValueError(f"action_type must be one of {allowed}")
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = {"critical", "high", "medium", "low", "none"}
            if v not in allowed:
                raise ValueError(f"severity must be one of {allowed}")
        return v


class Reward(BaseModel):
    """
    Shaped reward R: S × A × S' → [-1, 1].

    The breakdown dict exposes every component for analysis:
    step_reward, efficiency_bonus, coverage_bonus, consistency_penalty.
    This transparency is critical for reward attribution research.
    """
    value: float
    breakdown: Dict[str, float]
    reason: str

    @field_validator("value")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))


class State(BaseModel):
    """
    Full environment state including trajectory history.

    The trajectory list enables in-episode analysis and is the raw
    material for semantic world model training datasets.
    """
    current_pr: Observation
    reviewed_prs: List[str]
    pending_prs: List[str]
    total_reward: float
    step: int
    done: bool
    trajectory: List[Dict[str, Any]]
