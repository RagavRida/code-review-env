"""
CodeReviewEnv Client — typed async/sync client for interacting with the environment.

Usage:
    # Async (recommended)
    async with CodeReviewEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset(seed=42)
        result = await env.step(CodeReviewAction(action_type="label_severity", severity="high"))
        print(result.observation.pr_id, result.reward, result.done)

    # Sync
    with CodeReviewEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(seed=42)
        result = env.step(CodeReviewAction(action_type="label_severity", severity="high"))
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

try:
    from .models import CodeReviewAction, CodeReviewObservation, CodeReviewState
except ImportError:
    from models import CodeReviewAction, CodeReviewObservation, CodeReviewState


class CodeReviewEnv(EnvClient[CodeReviewAction, CodeReviewObservation, CodeReviewState]):
    """OpenEnv client for CodeReviewEnv.

    Handles WebSocket communication and type-safe parsing of
    actions and observations.
    """

    def _step_payload(self, action: CodeReviewAction) -> Dict[str, Any]:
        """Convert a CodeReviewAction to the JSON payload for the server."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CodeReviewObservation]:
        """Parse the server's JSON response into a typed StepResult."""
        obs_data = payload.get("observation", payload.get("data", payload))
        observation = CodeReviewObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CodeReviewState:
        """Parse the server's state response into a CodeReviewState."""
        state_data = payload.get("data", payload)
        return CodeReviewState(**state_data)
