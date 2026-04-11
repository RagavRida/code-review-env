"""
CodeReviewEnv Client — typed async/sync client for MCP tool-calling interface.

Usage:
    async with CodeReviewEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset(seed=42)

        # List available tools
        result = await env.step(CodeReviewAction(action_type="ListToolsAction"))

        # Call a tool
        result = await env.step(CodeReviewAction(
            action_type="ToolCallAction",
            tool_name="get_code",
            arguments={},
        ))
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

try:
    from .models import CodeReviewAction, CodeReviewObservation, CodeReviewState
except ImportError:
    from models import CodeReviewAction, CodeReviewObservation, CodeReviewState


class CodeReviewEnv(EnvClient[CodeReviewAction, CodeReviewObservation, CodeReviewState]):
    """OpenEnv client for CodeReviewEnv.

    Supports MCP tool-calling: ListToolsAction and ToolCallAction.
    """

    def _step_payload(self, action: CodeReviewAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CodeReviewObservation]:
        obs_data = payload.get("observation", payload.get("data", payload))
        observation = CodeReviewObservation(**obs_data)
        reward = payload.get("reward", getattr(observation, "reward", 0.0))
        done = payload.get("done", getattr(observation, "done", False))
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> CodeReviewState:
        state_data = payload.get("data", payload)
        return CodeReviewState(**state_data)

    # ─── Convenience methods ─────────────────────────────────────────

    async def list_tools(self) -> StepResult[CodeReviewObservation]:
        """List available tools."""
        return await self.step(CodeReviewAction(action_type="ListToolsAction"))

    async def call_tool(self, tool_name: str, **kwargs) -> StepResult[CodeReviewObservation]:
        """Call a tool by name."""
        return await self.step(CodeReviewAction(
            action_type="ToolCallAction",
            tool_name=tool_name,
            arguments=kwargs,
        ))
