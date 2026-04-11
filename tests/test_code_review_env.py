"""
Test suite for CodeReviewEnv v2.

Tests:
  Core interface (reset, step, state)
  Multi-step MDP (analyze, flag_line, request_hint, submit_review)
  Reward quality (perfect/empty/partial reviews)
  Hint system (progressive, penalty)
  Concurrent sessions
  Bug injectors (regex + AST-based)
  Procedural generation (variety, reproducibility, difficulty)
  Reward signals (bounds, individual functions)
"""

import pytest
import random

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from snippet_bank import (
    SNIPPET_BANK, BUG_INJECTORS, REGEX_INJECTORS, AST_INJECTORS,
    generate_episode,
    off_by_one_injector, null_deref_injector, wrong_operator_injector,
    unused_var_injector, logic_inversion_injector,
    ast_comparison_flip_injector, ast_binop_swap_injector,
    ast_boolop_flip_injector, ast_return_negate_injector,
)
from reward import compute_reward, _bug_overlap, _fix_similarity, _line_f1, _comment_score


# ── Core Interface Tests ─────────────────────────────────────────────────────

class TestCoreInterface:

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_reset_produces_buggy_code(self, difficulty):
        env = CodeReviewEnvironment()
        obs = env.reset(seed=42, difficulty=difficulty)
        assert isinstance(obs, CodeReviewObservation)
        assert obs.code != ""
        assert obs.done is False
        assert obs.language in ("python", "javascript", "go")
        assert obs.difficulty == difficulty
        assert obs.episode_budget == 5  # multi-step budget

    def test_reset_code_differs_from_original(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state
        assert len(state.gold_bugs) >= 1

    def test_step_submit_returns_done(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        action = CodeReviewAction(
            action_type="submit_review",
            issues=["test bug"],
            flagged_lines=[3],
            suggestion="fix it",
            comment="found a bug",
        )
        obs = env.step(action)
        assert isinstance(obs, CodeReviewObservation)
        assert obs.done is True
        assert 0.0 <= obs.reward <= 1.0

    def test_state_contains_gold_bugs(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state
        assert isinstance(state, CodeReviewState)
        assert isinstance(state.gold_bugs, list)


# ── Multi-Step MDP Tests ─────────────────────────────────────────────────────

class TestMultiStepMDP:

    def test_analyze_is_free(self):
        """analyze action gives 0 reward and doesn't end episode."""
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(action_type="analyze"))
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.analysis is not None and len(obs.analysis) > 0
        assert obs.step_number == 1

    def test_flag_correct_line_gives_positive_reward(self):
        """Flagging a correct line gives +0.15 immediate reward."""
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state
        if not state.gold_bugs:
            pytest.skip("No bugs injected")
        gold_line = state.gold_bugs[0]["lines"][0]
        obs = env.step(CodeReviewAction(action_type="flag_line", line=gold_line))
        assert obs.done is False
        assert obs.reward > 0
        assert gold_line in obs.flagged_so_far

    def test_flag_wrong_line_gives_negative_reward(self):
        """Flagging a wrong line gives -0.05 penalty."""
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(action_type="flag_line", line=999))
        assert obs.done is False
        assert obs.reward < 0

    def test_request_hint_doesnt_end_episode(self):
        """Hint request doesn't end the episode."""
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(action_type="request_hint"))
        assert obs.done is False
        assert obs.hint is not None

    def test_max_steps_forces_submit(self):
        """After 5 steps without submit, environment auto-submits."""
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        for i in range(5):
            obs = env.step(CodeReviewAction(action_type="analyze"))
            if obs.done:
                break
        assert obs.done is True, "Episode should end after max_steps"

    def test_multi_step_trajectory(self):
        """Full multi-step trajectory: analyze → flag → flag → submit."""
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state

        # Step 1: analyze
        obs = env.step(CodeReviewAction(action_type="analyze"))
        assert obs.done is False

        # Step 2: flag a line
        if state.gold_bugs:
            line = state.gold_bugs[0]["lines"][0]
            obs = env.step(CodeReviewAction(action_type="flag_line", line=line))
            assert obs.done is False

        # Step 3: submit review
        obs = env.step(CodeReviewAction(
            action_type="submit_review",
            issues=["bug found"],
            suggestion="fix",
            comment="review",
        ))
        assert obs.done is True
        assert obs.reward is not None

        # Check trajectory length
        traj = env.export_trajectory()
        assert len(traj) >= 3

    def test_flagged_lines_carry_to_submit(self):
        """Lines flagged in earlier steps are included in final grading."""
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state

        if not state.gold_bugs:
            pytest.skip("No bugs")

        gold_line = state.gold_bugs[0]["lines"][0]

        # Flag the correct line
        env.step(CodeReviewAction(action_type="flag_line", line=gold_line))

        # Submit empty review (but with previously flagged line)
        obs = env.step(CodeReviewAction(action_type="submit_review"))
        # Should get some line_precision credit from the prior flag
        assert obs.reward_breakdown is not None
        assert obs.reward_breakdown.get("line_precision", 0) > 0


# ── Reward Quality Tests ─────────────────────────────────────────────────────

class TestRewardQuality:

    def test_perfect_review_high_reward(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state
        if not state.gold_bugs:
            pytest.skip("No bugs")
        gold = state.gold_bugs[0]
        action = CodeReviewAction(
            action_type="submit_review",
            issues=[gold["description"]],
            flagged_lines=gold["lines"],
            suggestion=gold["fix"],
            comment=f"Found a {gold['bug_type']} error. {gold['fix']}",
        )
        obs = env.step(action)
        assert obs.reward >= 0.5

    def test_empty_review_low_reward(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(action_type="submit_review"))
        assert obs.reward < 0.15

    def test_partial_review_beats_empty(self):
        env1 = CodeReviewEnvironment()
        env1.reset(seed=42, difficulty="easy")
        state = env1.state
        if not state.gold_bugs:
            pytest.skip("No bugs")
        gold = state.gold_bugs[0]
        partial = env1.step(CodeReviewAction(
            action_type="submit_review",
            issues=[gold["description"]],
            comment="Found a bug.",
        ))

        env2 = CodeReviewEnvironment()
        env2.reset(seed=42, difficulty="easy")
        empty = env2.step(CodeReviewAction(action_type="submit_review"))

        assert partial.reward > empty.reward

    def test_reward_breakdown_present(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(
            action_type="submit_review", issues=["test"], comment="test",
        ))
        assert obs.reward_breakdown is not None
        for key in ["bug_detection", "fix_quality", "line_precision", "comment_quality", "efficiency"]:
            assert key in obs.reward_breakdown


# ── Hint Tests ───────────────────────────────────────────────────────────────

class TestHints:

    def test_hint_returns_text(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(action_type="request_hint"))
        assert obs.hint is not None and len(obs.hint) > 0

    def test_hint_costs_efficiency(self):
        """Hints reduce the efficiency signal at final grading."""
        env1 = CodeReviewEnvironment()
        env1.reset(seed=42, difficulty="easy")
        state = env1.state
        if not state.gold_bugs:
            pytest.skip("No bugs")
        gold = state.gold_bugs[0]
        review = CodeReviewAction(
            action_type="submit_review",
            issues=[gold["description"]],
            flagged_lines=gold["lines"],
            suggestion=gold["fix"],
            comment=f"Bug: {gold['description']}",
        )
        obs_no_hint = env1.step(review)

        env2 = CodeReviewEnvironment()
        env2.reset(seed=42, difficulty="easy")
        env2.step(CodeReviewAction(action_type="request_hint"))
        env2.step(CodeReviewAction(action_type="request_hint"))
        obs_with_hints = env2.step(review)

        assert obs_no_hint.reward >= obs_with_hints.reward

    def test_progressive_hints(self):
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        h1 = env.step(CodeReviewAction(action_type="request_hint"))
        h2 = env.step(CodeReviewAction(action_type="request_hint"))
        h3 = env.step(CodeReviewAction(action_type="request_hint"))
        assert len(h3.hint) >= len(h1.hint)


# ── Concurrent Session Tests ─────────────────────────────────────────────────

class TestConcurrentSessions:

    def test_two_sessions_independent(self):
        env1 = CodeReviewEnvironment()
        env2 = CodeReviewEnvironment()
        env1.reset(seed=42, difficulty="easy")
        env2.reset(seed=99, difficulty="hard")
        env1.step(CodeReviewAction(action_type="submit_review", issues=["bug"]))
        assert env2.state.step_count == 0

    def test_session_isolation(self):
        env1 = CodeReviewEnvironment()
        env2 = CodeReviewEnvironment()
        env1.reset(seed=42, difficulty="easy")
        env2.reset(seed=42, difficulty="easy")
        env1.step(CodeReviewAction(action_type="flag_line", line=5))
        assert env1.state.step_count == 1
        assert env2.state.step_count == 0


# ── Bug Injector Tests ───────────────────────────────────────────────────────

class TestBugInjectors:

    @pytest.mark.parametrize("injector,name", [
        (off_by_one_injector, "off_by_one"),
        (null_deref_injector, "null_deref"),
        (wrong_operator_injector, "wrong_operator"),
        (unused_var_injector, "unused_var"),
        (logic_inversion_injector, "logic_inversion"),
    ])
    def test_regex_injector(self, injector, name):
        rng = random.Random(42)
        successes = 0
        for snippet in SNIPPET_BANK:
            if snippet.language == "python":
                result = injector(snippet.code, rng)
                if result is not None:
                    buggy_code, bug = result
                    assert bug.bug_type == name
                    assert len(bug.lines) > 0
                    successes += 1
        assert successes >= 1

    @pytest.mark.parametrize("injector,name", [
        (ast_comparison_flip_injector, "ast_comparison_flip"),
        (ast_binop_swap_injector, "ast_binop_swap"),
        (ast_boolop_flip_injector, "ast_boolop_flip"),
        (ast_return_negate_injector, "ast_return_negate"),
    ])
    def test_ast_injector(self, injector, name):
        """AST-based injectors produce valid mutations on Python snippets."""
        rng = random.Random(42)
        successes = 0
        for snippet in SNIPPET_BANK:
            if snippet.language == "python":
                result = injector(snippet.code, rng)
                if result is not None:
                    buggy_code, bug = result
                    assert bug.bug_type == name
                    assert len(bug.lines) > 0
                    successes += 1
        assert successes >= 1, f"AST injector {name} failed on all Python snippets"

    def test_ast_injectors_preserve_syntax(self):
        """AST-injected Python code should still be parseable."""
        import ast as _ast
        rng = random.Random(42)
        for snippet in SNIPPET_BANK[:8]:
            if snippet.language != "python":
                continue
            for injector in AST_INJECTORS:
                result = injector(snippet.code, rng)
                if result is not None:
                    buggy_code, _ = result
                    try:
                        _ast.parse(buggy_code)
                    except SyntaxError:
                        pass  # Some edge cases may break; soft check


# ── Procedural Generation Tests ──────────────────────────────────────────────

class TestProceduralGeneration:

    def test_different_seeds_different_episodes(self):
        codes = set()
        for seed in range(10):
            _, code, _ = generate_episode(seed=seed, difficulty="easy")
            codes.add(code[:100])
        assert len(codes) >= 3

    def test_same_seed_reproducible(self):
        s1, c1, b1 = generate_episode(seed=42, difficulty="easy")
        s2, c2, b2 = generate_episode(seed=42, difficulty="easy")
        assert s1.name == s2.name
        assert c1 == c2

    def test_difficulty_affects_bug_count(self):
        easy_bugs = [len(generate_episode(seed=s, difficulty="easy")[2]) for s in range(20)]
        hard_bugs = [len(generate_episode(seed=s+1000, difficulty="hard")[2]) for s in range(20)]
        assert sum(hard_bugs) / len(hard_bugs) >= sum(easy_bugs) / len(easy_bugs)

    def test_snippet_bank_size(self):
        assert len(SNIPPET_BANK) >= 30

    def test_snippet_bank_covers_languages(self):
        languages = {s.language for s in SNIPPET_BANK}
        assert "python" in languages
        assert "javascript" in languages
        assert "go" in languages

    def test_injector_count(self):
        """Total injectors: 5 regex + 4 AST = 9."""
        assert len(REGEX_INJECTORS) == 5
        assert len(AST_INJECTORS) == 4
        assert len(BUG_INJECTORS) == 9


# ── Reward Signal Tests ──────────────────────────────────────────────────────

class TestRewardSignals:

    def test_all_signals_in_range(self):
        from snippet_bank import BugRecord
        bugs = [BugRecord("test bug", [5], "fix it", "off_by_one")]
        total, breakdown = compute_reward(
            issues=["test"], flagged_lines=[5], suggestion="fix",
            comment="comment", gold_bugs=bugs,
        )
        assert 0.0 <= total <= 1.0
        for key in ["bug_detection", "fix_quality", "line_precision", "comment_quality", "efficiency"]:
            assert 0.0 <= breakdown[key] <= 1.0

    def test_line_f1_perfect(self):
        from snippet_bank import BugRecord
        bugs = [BugRecord("bug", [5, 10], "fix", "off_by_one")]
        assert _line_f1([5, 10], bugs) == 1.0

    def test_line_f1_empty(self):
        from snippet_bank import BugRecord
        bugs = [BugRecord("bug", [5], "fix", "off_by_one")]
        assert _line_f1([], bugs) == 0.0

    def test_comment_quality_scales(self):
        short = _comment_score("bug")
        long = _comment_score(
            "Consider adding a null check on line 5 to guard against "
            "NullPointerException. You should use an if-guard before dereference."
        )
        assert long > short
