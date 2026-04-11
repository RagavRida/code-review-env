"""
FastAPI application for CodeReviewEnv — OpenEnv create_app() + MCP tools.

Endpoints (via OpenEnv framework):
  /ws       — WebSocket for persistent sessions
  /health   — HTTP GET health check
  /reset    — HTTP POST reset environment
  /step     — HTTP POST take action
  /state    — HTTP GET current state
  /docs     — OpenAPI documentation

MCP Tools (via FastMCP):
  get_code_snippet — returns current buggy code + metadata
  submit_review    — accepts structured review, returns reward
  request_hint     — returns a hint (costs -0.05 reward)
  get_state        — returns episode state summary

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import json
from typing import Dict, List, Optional

from fastapi import Query
from fastapi.responses import JSONResponse, PlainTextResponse

from openenv.core.env_server import create_app

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction, CodeReviewObservation

# ─── Create OpenEnv app with concurrent session support ─────────────────────

app = create_app(
    CodeReviewEnvironment,
    CodeReviewAction,
    CodeReviewObservation,
    env_name="code_review_env",
)


# ─── Enhanced /health endpoint ───────────────────────────────────────────────

@app.get("/health")
async def health():
    """Enhanced health check for judges and orchestrators."""
    return {
        "status": "ok",
        "environment": "CodeReviewEnv",
        "version": "2.0.0",
        "difficulty_tiers": ["easy", "medium", "hard"],
        "languages": ["python", "javascript", "go"],
        "reward_signals": [
            "bug_detection",
            "fix_quality",
            "line_precision",
            "comment_quality",
            "efficiency",
        ],
        "grader": "deterministic",
        "mcp_enabled": True,
        "supports_concurrent_sessions": True,
    }


# ─── MCP Tool Endpoints ─────────────────────────────────────────────────────
# These provide tool-calling style interaction alongside the standard
# reset/step API. Agents can use MCP tools for richer interaction.

# Per-session environment instances for MCP
_mcp_sessions: Dict[str, CodeReviewEnvironment] = {}


def _get_mcp_env(session_id: str) -> CodeReviewEnvironment:
    """Get or create an environment for an MCP session."""
    if session_id not in _mcp_sessions:
        _mcp_sessions[session_id] = CodeReviewEnvironment()
    return _mcp_sessions[session_id]


@app.post("/mcp/reset")
async def mcp_reset(
    session_id: str = Query(default="default"),
    seed: Optional[int] = Query(default=None),
    difficulty: str = Query(default="easy"),
):
    """MCP: Reset the environment for a new episode."""
    env = _get_mcp_env(session_id)
    obs = env.reset(seed=seed, difficulty=difficulty)
    return {
        "session_id": session_id,
        "code": obs.code,
        "language": obs.language,
        "difficulty": obs.difficulty,
        "instructions": obs.instructions,
    }


@app.post("/mcp/get_code_snippet")
async def mcp_get_code_snippet(session_id: str = Query(default="default")):
    """MCP tool: Get the current buggy code snippet for review."""
    env = _get_mcp_env(session_id)
    return env.get_code_snippet()


@app.post("/mcp/submit_review")
async def mcp_submit_review(
    session_id: str = Query(default="default"),
    issues: List[str] = Query(default=[]),
    flagged_lines: List[int] = Query(default=[]),
    suggestion: str = Query(default=""),
    comment: str = Query(default=""),
):
    """MCP tool: Submit a code review. Returns reward and done signal."""
    env = _get_mcp_env(session_id)
    action = CodeReviewAction(
        issues=issues,
        flagged_lines=flagged_lines,
        suggestion=suggestion,
        comment=comment,
    )
    obs = env.step(action)
    return {
        "reward": obs.reward,
        "done": obs.done,
        "breakdown": obs.reward_breakdown,
    }


@app.post("/mcp/request_hint")
async def mcp_request_hint(session_id: str = Query(default="default")):
    """MCP tool: Request a hint. Costs -0.05 reward."""
    env = _get_mcp_env(session_id)
    return env.request_hint()


@app.get("/mcp/get_state")
async def mcp_get_state(session_id: str = Query(default="default")):
    """MCP tool: Get current episode state summary."""
    env = _get_mcp_env(session_id)
    return env.get_state_summary()


# ─── Trajectory export ──────────────────────────────────────────────────────

_trajectory_store: dict = {}


@app.get("/export_trajectory")
async def export_trajectory(
    session_id: str = Query(default="latest"),
    format: str = Query(default="jsonl"),
):
    """Export episode trajectory as JSONL for MBRL research."""
    trajectory = _trajectory_store.get(session_id, [])
    if not trajectory:
        return JSONResponse(
            content={"message": "No trajectory found.", "session_id": session_id},
            status_code=404,
        )
    if format == "json":
        return JSONResponse(content={"session_id": session_id, "transitions": trajectory})
    lines = [json.dumps(t) for t in trajectory]
    return PlainTextResponse(content="\n".join(lines), media_type="application/jsonl")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
