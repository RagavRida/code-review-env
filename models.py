"""
CodeReviewEnv — OpenEnv-compliant typed models.

Multi-step action space for code review:
  - analyze:       inspect the code, get initial analysis (free action)
  - flag_line:     flag a specific line as buggy (intermediate reward)
  - request_hint:  get a hint about a bug (-0.05 penalty per hint)
  - submit_review: submit final structured review (full grading, ends episode)

Observations expose buggy code; gold answers stay hidden in State.
"""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State


# ─── Action ──────────────────────────────────────────────────────────────────


class CodeReviewAction(Action):
    """Multi-step action space for code review.

    Action types:
      analyze       — request deeper analysis of the code (step_reward = 0)
      flag_line     — flag a specific line as buggy (intermediate reward if correct)
      request_hint  — get a hint (-0.05 efficiency penalty)
      submit_review — submit final review (full 5-signal grading, ends episode)
    """

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(
        default="submit_review",
        description="One of: analyze, flag_line, request_hint, submit_review",
    )
    # Fields for flag_line
    line: Optional[int] = Field(
        default=None,
        description="Line number to flag (for flag_line action)",
    )
    # Fields for submit_review
    issues: List[str] = Field(
        default_factory=list,
        description="Descriptions of bugs found in the code",
    )
    flagged_lines: List[int] = Field(
        default_factory=list,
        description="Line numbers the agent believes contain bugs",
    )
    suggestion: str = Field(
        default="",
        description="Suggested fix — code patch or description",
    )
    comment: str = Field(
        default="",
        description="Natural-language review comment",
    )


# ─── Observation ─────────────────────────────────────────────────────────────


class CodeReviewObservation(Observation):
    """Observation returned after reset() and step().

    Inherits done, reward, metadata from openenv Observation.
    Adds the buggy code and episode context.
    Gold answers are NEVER exposed here — they live in State only.
    """

    model_config = ConfigDict(extra="allow")

    code: str = Field(default="", description="Buggy source code to review")
    language: str = Field(default="python", description="Programming language")
    difficulty: str = Field(default="easy", description="easy | medium | hard")
    instructions: str = Field(
        default="Review the code. Report bugs, flagged lines, and a suggested fix.",
        description="Task instructions for the agent",
    )
    step_number: int = Field(default=0, description="Current step in episode")
    episode_budget: int = Field(default=5, description="Max steps in this episode")
    hint: Optional[str] = Field(
        default=None,
        description="Hint text (if agent requested one)",
    )
    flagged_so_far: List[int] = Field(
        default_factory=list,
        description="Lines flagged in previous steps (for multi-step tracking)",
    )
    analysis: Optional[str] = Field(
        default=None,
        description="Analysis text from analyze action",
    )
    reward_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-signal reward breakdown for analysis",
    )


# ─── State ───────────────────────────────────────────────────────────────────


class CodeReviewState(State):
    """Full environment state — includes gold answers (hidden from agent).

    The gold_bugs list contains the injected bugs with their descriptions,
    affected lines, fixes, and types. This is used for grading and is
    NEVER sent to the agent in observations.
    """

    original_code: str = Field(default="", description="Clean code before injection")
    buggy_code: str = Field(default="", description="Code with injected bugs")
    gold_bugs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Injected bugs: [{description, lines, fix, bug_type}, ...]",
    )
    language: str = Field(default="python", description="Source language")
    difficulty: str = Field(default="easy", description="Difficulty tier")
    hint_count: int = Field(default=0, description="Hints requested (costs reward)")
    snippet_name: str = Field(default="", description="Which snippet was used")
