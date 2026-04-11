"""
FastAPI application for CodeReviewEnv.

Follows the OpenEnv reference pattern (calendar_env, repl_env):
  - create_app() with environment factory
  - MCP tool-calling interface
  - Concurrent session support
  - Gradio UI for interactive testing

Entry point: server.app:app (referenced in openenv.yaml)
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from pathlib import Path

# Ensure imports work both as package and standalone
SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
REPO_ROOT = SERVER_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openenv.core.env_server.http_server import create_app

try:
    from ..models import CodeReviewAction, CodeReviewObservation
    from .code_review_environment import CodeReviewEnvironment
except ImportError:
    from models import CodeReviewAction, CodeReviewObservation
    from server.code_review_environment import CodeReviewEnvironment

logger = logging.getLogger(__name__)


def create_code_review_environment() -> CodeReviewEnvironment:
    """Factory for creating fresh environment instances (one per session)."""
    return CodeReviewEnvironment()


# ─── Build the app using OpenEnv create_app ──────────────────────────────────

# Check if create_app supports gradio_builder
_sig = inspect.signature(create_app)

try:
    from server.gradio_ui import build_gradio_app

    if "gradio_builder" in _sig.parameters:
        app = create_app(
            create_code_review_environment,
            CodeReviewAction,
            CodeReviewObservation,
            env_name="code_review_env",
            max_concurrent_envs=50,
            gradio_builder=build_gradio_app,
        )
    else:
        app = create_app(
            create_code_review_environment,
            CodeReviewAction,
            CodeReviewObservation,
            env_name="code_review_env",
            max_concurrent_envs=50,
        )
except ImportError:
    app = create_app(
        create_code_review_environment,
        CodeReviewAction,
        CodeReviewObservation,
        env_name="code_review_env",
        max_concurrent_envs=50,
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int | None = None):
    """Run the CodeReviewEnv server with uvicorn."""
    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "7860"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
