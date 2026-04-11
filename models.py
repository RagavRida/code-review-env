"""
CodeReviewEnv — OpenEnv MCP-compliant typed models.

Follows the same MCP tool-calling pattern as the reference environments
(calendar_env, repl_env). Agents interact via tool calls:

  ListToolsAction  — discover available tools
  ToolCallAction   — call a tool by name with arguments

Available tools:
  get_code       — get the buggy code snippet for review
  analyze_code   — get structural analysis of the code
  check_line     — check if a specific line contains a bug (+/- reward)
  get_hint       — get a progressive hint (costs efficiency)
  submit_review  — submit final structured review (ends episode)
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State


# ─── MCP Action (matches calendar_env pattern) ──────────────────────────────


class MCPAction(Action):
    """MCP tool-calling action. Agents either list tools or call a tool."""

    action_type: Literal["ListToolsAction", "ToolCallAction"] = Field(
        ..., description="Type of action to perform"
    )
    tool_name: Optional[str] = Field(
        None, description="Name of tool to call (required for ToolCallAction)"
    )
    arguments: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Arguments for the tool"
    )


class CodeReviewAction(MCPAction):
    """Code review action — inherits MCP tool-calling interface."""
    pass


# ─── MCP Observation (matches calendar_env pattern) ─────────────────────────


class MCPObservation(Observation):
    """MCP observation returned after tool calls."""

    success: bool = Field(True, description="Whether the action succeeded")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    tools_list: Optional[List[Dict[str, Any]]] = Field(
        None, description="Available tools (for ListToolsAction)"
    )
    tool_result: Optional[Dict[str, Any]] = Field(
        None, description="Result from tool call"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    done: bool = Field(False, description="Whether the episode is over")
    reward: Optional[float] = Field(None, description="Reward signal")


class CodeReviewObservation(MCPObservation):
    """Code review observation — inherits MCP observation."""
    pass


# ─── State ───────────────────────────────────────────────────────────────────


class CodeReviewState(State):
    """Full environment state — includes gold answers (hidden from agent)."""

    original_code: str = Field(default="", description="Clean code before injection")
    buggy_code: str = Field(default="", description="Code with injected bugs")
    gold_bugs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Injected bugs: [{description, lines, fix, bug_type}, ...]",
    )
    language: str = Field(default="python", description="Source language")
    difficulty: str = Field(default="easy", description="Difficulty tier")
    hint_count: int = Field(default=0, description="Hints requested")
    snippet_name: str = Field(default="", description="Which snippet was used")
    flagged_lines: List[int] = Field(default_factory=list, description="Lines flagged so far")
