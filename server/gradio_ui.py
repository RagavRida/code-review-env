"""
Gradio UI for interactive CodeReviewEnv testing.

Provides a web interface for manually reviewing code, calling tools,
and inspecting environment state — like repl_env's Gradio UI.
"""

from __future__ import annotations

import json
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None


def build_gradio_app(env_factory) -> Any:
    """Build a Gradio Blocks interface for interactive code review.

    Args:
        env_factory: callable that returns a CodeReviewEnvironment instance

    Returns:
        Gradio Blocks app
    """
    if gr is None:
        return None

    env = env_factory()

    def reset_env(seed: int, difficulty: str):
        obs = env.reset(seed=int(seed), difficulty=difficulty)
        tools = obs.tools_list or []
        tools_text = "\n".join(f"  - {t['name']}: {t['description']}" for t in tools)
        return (
            json.dumps(obs.tool_result, indent=2) if obs.tool_result else "Reset complete",
            f"Episode: {obs.metadata.get('episode_id', '?')}\n"
            f"Difficulty: {difficulty}\n"
            f"Language: {obs.metadata.get('language', '?')}\n\n"
            f"Available tools:\n{tools_text}",
            "",
        )

    def call_tool(tool_name: str, args_json: str):
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError:
            return "Error: Invalid JSON in arguments", "", ""

        from models import CodeReviewAction
        action = CodeReviewAction(
            action_type="ToolCallAction",
            tool_name=tool_name,
            arguments=args,
        )
        obs = env.step(action)

        result_text = json.dumps(obs.tool_result, indent=2) if obs.tool_result else "No result"
        if obs.error_message:
            result_text = f"Error: {obs.error_message}"

        state = env.state
        state_text = (
            f"Step: {state.step_count}\n"
            f"Done: {obs.done}\n"
            f"Reward: {obs.reward}\n"
            f"Hints used: {state.hint_count}\n"
            f"Lines flagged: {state.flagged_lines}"
        )

        traj = env.export_trajectory()
        traj_text = "\n".join(
            f"  [{t['step']}] {t['tool']}: reward={t['reward']}"
            for t in traj
        )

        return result_text, state_text, traj_text

    with gr.Blocks(title="CodeReviewEnv") as demo:
        gr.Markdown("# CodeReviewEnv — Interactive Testing")
        gr.Markdown("Review buggy code by calling tools. Discover tools → get code → analyze → check lines → submit review.")

        with gr.Row():
            seed_input = gr.Number(value=42, label="Seed", precision=0)
            difficulty_input = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Difficulty")
            reset_btn = gr.Button("Reset Episode", variant="primary")

        with gr.Row():
            tool_name = gr.Dropdown(
                choices=["get_code", "analyze_code", "check_line", "get_hint", "submit_review"],
                value="get_code",
                label="Tool",
            )
            tool_args = gr.Textbox(label="Arguments (JSON)", placeholder='{"line": 5}', lines=2)
            call_btn = gr.Button("Call Tool", variant="secondary")

        with gr.Row():
            result_output = gr.Textbox(label="Tool Result", lines=15, interactive=False)
            with gr.Column():
                state_output = gr.Textbox(label="State", lines=8, interactive=False)
                trajectory_output = gr.Textbox(label="Trajectory", lines=8, interactive=False)

        reset_btn.click(reset_env, inputs=[seed_input, difficulty_input], outputs=[result_output, state_output, trajectory_output])
        call_btn.click(call_tool, inputs=[tool_name, tool_args], outputs=[result_output, state_output, trajectory_output])

    return demo
