"""
CodeReviewEnv Client — typed async/sync client for interacting with the environment.

Usage:
    # Async (recommended)
    async with CodeReviewEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset(seed=42)
        result = await env.step(CodeReviewAction(
            issues=["Off-by-one error in loop"],
            flagged_lines=[3],
            suggestion="Change < to <=",
            comment="Loop boundary is wrong."
        ))
        print(result.observation.code, result.reward, result.done)

    # Sync
    with CodeReviewEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(seed=42)
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
        reward = payload.get("reward", getattr(observation, "reward", 0.0))
        done = payload.get("done", getattr(observation, "done", False))
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CodeReviewState:
        """Parse the server's state response into a CodeReviewState."""
        state_data = payload.get("data", payload)
        return CodeReviewState(**state_data)
