"""
FastAPI application for CodeReviewEnv — uses openenv create_app().

This automatically creates all required endpoints:
  /ws       — WebSocket for persistent sessions
  /health   — HTTP GET health check (enhanced with task info)
  /reset    — HTTP POST reset environment
  /step     — HTTP POST take action
  /state    — HTTP GET current state
  /export_trajectory — GET trajectory export (JSONL)
  /docs     — OpenAPI documentation
  /web      — Interactive web UI (when enabled)

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import json
from fastapi import Query
from fastapi.responses import JSONResponse, PlainTextResponse

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


# ─── Enhanced /health endpoint ───────────────────────────────────────────────

@app.get("/health")
async def health():
    """Enhanced health check with task info for judges."""
    return {
        "status": "ok",
        "environment": "CodeReviewEnv",
        "version": "1.0.0",
        "tasks": [
            "bug_severity_labeling",
            "queue_prioritization",
            "multi_turn_review",
        ],
        "task_count": 3,
        "grader": "deterministic",
        "trajectory_export": True,
    }


# ─── Trajectory export endpoint ──────────────────────────────────────────────

# In-memory trajectory store (per-session)
_trajectory_store: dict = {}


@app.get("/export_trajectory")
async def export_trajectory(
    session_id: str = Query(default="latest", description="Session/episode ID"),
    format: str = Query(default="jsonl", description="Export format: jsonl or json"),
):
    """Export episode trajectory as JSONL for MBRL research.

    Each line is a (s, a, r, s', done) transition:
      {"state": {...}, "action": "...", "reward": 0.75, "next_state": {...}, "done": false}

    Usage:
        GET /export_trajectory?session_id=latest
        GET /export_trajectory?session_id=abc123&format=json
    """
    # Get the current env instance's trajectory
    trajectory = _trajectory_store.get(session_id, [])

    if not trajectory:
        # Try to get from the most recent episode
        return JSONResponse(
            content={
                "message": "No trajectory found. Run reset() + step() first.",
                "session_id": session_id,
                "available_sessions": list(_trajectory_store.keys()),
            },
            status_code=404,
        )

    if format == "json":
        return JSONResponse(content={"session_id": session_id, "transitions": trajectory})

    # JSONL format
    lines = [json.dumps(t) for t in trajectory]
    return PlainTextResponse(content="\n".join(lines), media_type="application/jsonl")


def store_transition(session_id: str, transition: dict):
    """Store a transition for later export. Called from CodeReviewEnvironment.step()."""
    if session_id not in _trajectory_store:
        _trajectory_store[session_id] = []
    _trajectory_store[session_id].append(transition)


def clear_trajectory(session_id: str):
    """Clear trajectory for a session. Called from CodeReviewEnvironment.reset()."""
    _trajectory_store[session_id] = []


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
