"""
Test suite for CodeReviewEnv v2 — MCP tool-calling pattern.
"""

import pytest
import random

from server.code_review_environment import CodeReviewEnvironment, TOOLS
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from snippet_bank import (
    SNIPPET_BANK, BUG_INJECTORS, REGEX_INJECTORS, AST_INJECTORS,
    generate_episode,
    off_by_one_injector, null_deref_injector, wrong_operator_injector,
    unused_var_injector, logic_inversion_injector,
    ast_comparison_flip_injector, ast_binop_swap_injector,
    ast_boolop_flip_injector, ast_return_negate_injector,
)
from reward import compute_reward, _line_f1, _comment_score


# ── Core Interface (MCP pattern) ─────────────────────────────────────────────

class TestCoreInterface:

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_reset_returns_tools_list(self, difficulty):
        """reset() returns observation with tools_list (like calendar_env)."""
        env = CodeReviewEnvironment()
        obs = env.reset(seed=42, difficulty=difficulty)
        assert isinstance(obs, CodeReviewObservation)
        assert obs.done is False
        assert obs.tools_list is not None
        assert len(obs.tools_list) == 8

    def test_list_tools_action(self):
        """ListToolsAction returns available tools."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(action_type="ListToolsAction"))
        assert obs.success is True
        assert obs.tools_list is not None
        tool_names = {t["name"] for t in obs.tools_list}
        assert tool_names == {"get_code", "run_code", "run_tests", "analyze_code", "check_line", "get_hint", "submit_fix", "submit_review"}

    def test_tool_call_get_code(self):
        """ToolCallAction with get_code returns source code."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="get_code",
        ))
        assert obs.success is True
        assert obs.tool_result is not None
        assert "code" in obs.tool_result
        assert len(obs.tool_result["code"]) > 0
        assert "language" in obs.tool_result

    def test_unknown_tool_returns_error(self):
        """Unknown tool name returns error."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="nonexistent_tool",
        ))
        assert obs.success is False
        assert obs.error_message is not None

    def test_state_contains_gold_bugs(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state
        assert isinstance(state, CodeReviewState)
        assert isinstance(state.gold_bugs, list)


# ── Tool Interaction Tests ───────────────────────────────────────────────────

class TestToolInteraction:

    def test_analyze_code_returns_structure(self):
        """analyze_code returns line count, function count, etc."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="analyze_code",
        ))
        assert obs.success is True
        result = obs.tool_result
        assert "total_lines" in result
        assert "functions" in result
        assert "analysis" in result

    def test_check_line_correct_gives_positive(self):
        """check_line near a bug gives positive reward."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state
        if not state.gold_bugs:
            pytest.skip("No bugs")
        gold_line = state.gold_bugs[0]["lines"][0]
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="check_line",
            arguments={"line": gold_line},
        ))
        assert obs.success is True
        assert obs.reward > 0
        assert obs.tool_result["is_suspicious"] is True

    def test_check_line_wrong_gives_negative(self):
        """check_line on wrong line gives negative reward."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="check_line",
            arguments={"line": 999},
        ))
        assert obs.reward < 0

    def test_get_hint_progressive(self):
        """Each hint is more specific."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        h1 = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="get_hint",
        ))
        h2 = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="get_hint",
        ))
        assert h1.tool_result["hint_count"] == 1
        assert h2.tool_result["hint_count"] == 2

    def test_submit_review_ends_episode(self):
        """submit_review returns done=True with full reward."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="submit_review",
            arguments={"issues": ["bug"], "comment": "review"},
        ))
        assert obs.done is True
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_max_steps_forces_submit(self):
        """After 10 tool calls without submit, environment auto-submits."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        for i in range(10):
            obs = env.step(CodeReviewAction(
                action_type="ToolCallAction", tool_name="analyze_code",
            ))
            if obs.done:
                break
        assert obs.done is True


# ── Multi-Step Trajectory ────────────────────────────────────────────────────

class TestMultiStepTrajectory:

    def test_full_trajectory(self):
        """Full tool-calling trajectory: list → get_code → check → submit."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state

        # Step 1: List tools
        obs = env.step(CodeReviewAction(action_type="ListToolsAction"))
        assert obs.done is False

        # Step 2: Get code
        obs = env.step(CodeReviewAction(action_type="ToolCallAction", tool_name="get_code"))
        assert obs.done is False

        # Step 3: Check a line
        if state.gold_bugs:
            line = state.gold_bugs[0]["lines"][0]
            obs = env.step(CodeReviewAction(
                action_type="ToolCallAction", tool_name="check_line",
                arguments={"line": line},
            ))
            assert obs.done is False

        # Step 4: Submit
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="submit_review",
            arguments={"issues": ["found bug"], "comment": "review done"},
        ))
        assert obs.done is True
        traj = env.export_trajectory()
        assert len(traj) >= 3

    def test_flagged_lines_carry_to_submit(self):
        """Lines checked earlier are included in final grading."""
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state
        if not state.gold_bugs:
            pytest.skip("No bugs")

        gold_line = state.gold_bugs[0]["lines"][0]
        env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="check_line",
            arguments={"line": gold_line},
        ))
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="submit_review",
            arguments={},
        ))
        # Should get line_precision credit from prior check_line
        breakdown = obs.tool_result.get("breakdown", {})
        assert breakdown.get("line_precision", 0) > 0


# ── Reward Quality ───────────────────────────────────────────────────────────

class TestRewardQuality:

    def test_perfect_review_high_reward(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state
        if not state.gold_bugs:
            pytest.skip("No bugs")
        gold = state.gold_bugs[0]
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="submit_review",
            arguments={
                "issues": [gold["description"]],
                "flagged_lines": gold["lines"],
                "suggestion": gold["fix"],
                "comment": f"Found {gold['bug_type']} error. {gold['fix']}",
            },
        ))
        assert obs.reward >= 0.5

    def test_empty_review_low_reward(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(
            action_type="ToolCallAction", tool_name="submit_review",
            arguments={},
        ))
        assert obs.reward < 0.15


# ── Concurrent Sessions ──────────────────────────────────────────────────────

class TestConcurrentSessions:

    def test_two_sessions_independent(self):
        env1 = CodeReviewEnvironment()
        env2 = CodeReviewEnvironment()
        env1.reset(seed=42, difficulty="easy")
        env2.reset(seed=99, difficulty="hard")
        env1.step(CodeReviewAction(action_type="ToolCallAction", tool_name="submit_review", arguments={}))
        assert env2.state.step_count == 0


# ── Bug Injectors ────────────────────────────────────────────────────────────

class TestBugInjectors:

    @pytest.mark.parametrize("injector,name", [
        (off_by_one_injector, "off_by_one"),
        (null_deref_injector, "null_deref"),
        (wrong_operator_injector, "wrong_operator"),
        (unused_var_injector, "unused_var"),
        (logic_inversion_injector, "logic_inversion"),
        (ast_comparison_flip_injector, "ast_comparison_flip"),
        (ast_binop_swap_injector, "ast_binop_swap"),
        (ast_boolop_flip_injector, "ast_boolop_flip"),
        (ast_return_negate_injector, "ast_return_negate"),
    ])
    def test_injector_works(self, injector, name):
        rng = random.Random(42)
        successes = sum(
            1 for s in SNIPPET_BANK if s.language == "python"
            and injector(s.code, rng) is not None
        )
        assert successes >= 1, f"{name} failed on all Python snippets"


# ── Procedural Generation ────────────────────────────────────────────────────

class TestProceduralGeneration:

    def test_different_seeds(self):
        codes = {generate_episode(seed=s, difficulty="easy")[1][:100] for s in range(10)}
        assert len(codes) >= 3

    def test_reproducible(self):
        _, c1, _ = generate_episode(seed=42, difficulty="easy")
        _, c2, _ = generate_episode(seed=42, difficulty="easy")
        assert c1 == c2

    def test_snippet_bank_size(self):
        assert len(SNIPPET_BANK) >= 30

    def test_covers_languages(self):
        assert {s.language for s in SNIPPET_BANK} >= {"python", "javascript", "go"}

    def test_injector_count(self):
        assert len(BUG_INJECTORS) == 9
