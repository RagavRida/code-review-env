"""
FastAPI application for CodeReviewEnv — uses openenv create_app().

This automatically creates all required endpoints:
  /ws       — WebSocket for persistent sessions
  /health   — HTTP GET health check
  /reset    — HTTP POST reset environment
  /step     — HTTP POST take action
  /state    — HTTP GET current state
  /docs     — OpenAPI documentation
  /web      — Interactive web UI (when enabled)

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

from openenv.core.env_server import create_app

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction, CodeReviewObservation

# create_app takes:
#   env: factory callable -> Environment instance
#   action_cls: the Action subclass
#   observation_cls: the Observation subclass
#   env_name: used for web UI title
app = create_app(
    CodeReviewEnvironment,
    CodeReviewAction,
    CodeReviewObservation,
    env_name="code_review_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
