"""
Test suite for CodeReviewEnv.

19 tests covering core interface, grader ranges, score variance,
reproducibility, exploit prevention, and agent baselines.
"""

import random
import statistics
import pytest

from env.base import CodeReviewEnv
from env.models import Action, Observation, Reward, State
from env.data_generator import PR_TEMPLATES


# ── Core Interface Tests ──────────────────────────────────────────────────────

class TestCoreInterface:
    """Tests for the basic OpenEnv interface contract."""

    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    def test_reset_returns_observation(self, task):
        """reset() must return valid Observation for all tasks."""
        env = CodeReviewEnv(task=task, seed=42)
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.pr_id
        assert obs.title
        assert len(obs.files) > 0

    def test_step_valid_action(self):
        """step() with valid action returns (Observation, Reward, bool, Dict)."""
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        action = Action(action_type="label_severity", severity="high")
        result = env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_invalid_action_no_crash(self):
        """Malformed action returns penalty reward, never raises."""
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        # Wrong action type for easy task
        action = Action(action_type="approve")
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, Reward)
        assert reward.value <= 0  # Should be penalized

    def test_reward_bounds(self):
        """reward.value always in [-1.0, 1.0] across many random actions."""
        random.seed(42)
        severities = ["critical", "high", "medium", "low", "none"]
        for _ in range(100):
            env = CodeReviewEnv(task="easy", seed=random.randint(1, 999))
            env.reset()
            sev = random.choice(severities)
            action = Action(action_type="label_severity", severity=sev)
            _, reward, _, _ = env.step(action)
            assert -1.0 <= reward.value <= 1.0, f"Reward {reward.value} out of bounds"

    def test_episode_terminates(self):
        """done=True exactly at episode_length steps for easy task."""
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        for i in range(4):
            action = Action(action_type="label_severity", severity="medium")
            _, _, done, _ = env.step(action)
            assert done is False, f"Episode terminated early at step {i}"
        action = Action(action_type="label_severity", severity="medium")
        _, _, done, _ = env.step(action)
        assert done is True, "Episode should be done after 5 steps"

    def test_state_trajectory_length(self):
        """state().trajectory length equals step count."""
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        for i in range(3):
            action = Action(action_type="label_severity", severity="high")
            env.step(action)
        s = env.state()
        assert isinstance(s, State)
        assert len(s.trajectory) == 3

    def test_export_trajectory_format(self):
        """All required keys present in each transition."""
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        for _ in range(5):
            action = Action(action_type="label_severity", severity="high")
            env.step(action)
        traj = env.export_trajectory()
        assert len(traj) == 5
        required = {"step", "state", "action", "reward", "next_state"}
        for t in traj:
            assert required <= set(t.keys()), f"Missing keys: {required - set(t.keys())}"


# ── Grader Range Tests ────────────────────────────────────────────────────────

class TestGraderRanges:
    """Tests that grader scores are within expected bounds."""

    def test_grader_easy_range(self):
        """Easy grader scores in [0, 1] for all test PRs."""
        from graders.grader_easy import EasyGrader
        for template in PR_TEMPLATES:
            grader = EasyGrader()
            for sev in ["critical", "high", "medium", "low", "none"]:
                action = Action(action_type="label_severity", severity=sev)
                reward, _ = grader.grade(action, template["pr_id"])
                assert -1.0 <= reward.value <= 1.0

    def test_grader_medium_range(self):
        """Medium grader scores in [0, 1] for all test queues."""
        from graders.grader_medium import MediumGrader
        from env.data_generator import DataGenerator
        grader = MediumGrader()
        gen = DataGenerator(seed=42)
        queues = gen.generate_medium_episode(num_queues=3, queue_size=5)
        for queue in queues:
            gt = gen.compute_priority_order(queue)
            action = Action(action_type="prioritize", priority_order=gt)
            reward, _ = grader.grade(action, queue, gt)
            assert 0.0 <= reward.value <= 1.0

    def test_grader_hard_range(self):
        """Hard grader scores in [-1, 1] for all test comment sets."""
        from graders.grader_hard import HardGrader
        for template in PR_TEMPLATES[:5]:
            grader = HardGrader()
            pr_id = template["pr_id"]
            if template["bug_lines"]:
                action = Action(
                    action_type="add_comment",
                    comment="Consider adding a check here",
                    target_file=template["filename"],
                    target_line=template["bug_lines"][0],
                )
                grader.add_comment(pr_id, action)
            reward, _ = grader.grade_pr(pr_id, "request_changes")
            assert -1.0 <= reward.value <= 1.0


# ── Score Variance Tests ──────────────────────────────────────────────────────

class TestScoreVariance:
    """Tests that scores have meaningful variance (not trivially constant)."""

    def test_score_variance_easy(self):
        """Easy task has score std > 0.05 across random actions."""
        from graders.grader_easy import EasyGrader
        random.seed(42)
        scores = []
        for template in PR_TEMPLATES:
            grader = EasyGrader()
            sev = random.choice(["critical", "high", "medium", "low", "none"])
            action = Action(action_type="label_severity", severity=sev)
            reward, _ = grader.grade(action, template["pr_id"])
            scores.append(reward.value)
        assert statistics.stdev(scores) > 0.05

    def test_score_variance_medium(self):
        """Medium task has score std > 0.05 across different orderings."""
        from graders.grader_medium import MediumGrader
        from env.data_generator import DataGenerator
        random.seed(42)
        scores = []
        gen = DataGenerator(seed=42)
        queues = gen.generate_medium_episode(num_queues=3, queue_size=5)
        for queue in queues:
            gt = gen.compute_priority_order(queue)
            # Test with various orderings
            for _ in range(5):
                order = list(gt)
                random.shuffle(order)
                grader = MediumGrader()
                action = Action(action_type="prioritize", priority_order=order)
                reward, _ = grader.grade(action, queue, gt)
                scores.append(reward.value)
        assert statistics.stdev(scores) > 0.05

    def test_score_variance_hard(self):
        """Hard task has score std > 0.10 across different feedback."""
        from graders.grader_hard import HardGrader
        scores = []
        for template in PR_TEMPLATES[:6]:
            # No comments + approve
            grader = HardGrader()
            reward, _ = grader.grade_pr(template["pr_id"], "approve")
            scores.append(reward.value)

            # Good comment + request_changes
            grader = HardGrader()
            if template["bug_lines"]:
                action = Action(
                    action_type="add_comment",
                    comment="Consider adding a null check to prevent crash",
                    target_file=template["filename"],
                    target_line=template["bug_lines"][0],
                )
                grader.add_comment(template["pr_id"], action)
            reward, _ = grader.grade_pr(template["pr_id"], "request_changes")
            scores.append(reward.value)
        assert statistics.stdev(scores) > 0.10


# ── Reproducibility Tests ─────────────────────────────────────────────────────

class TestReproducibility:
    """Tests that same seed produces identical results."""

    def test_reproducibility(self):
        """Two runs with seed=42 produce identical trajectories."""
        traj1 = self._run_episode(42)
        traj2 = self._run_episode(42)
        assert len(traj1) == len(traj2)
        for t1, t2 in zip(traj1, traj2):
            assert t1 == t2, "Trajectories differ with same seed"

    @staticmethod
    def _run_episode(seed):
        env = CodeReviewEnv(task="easy", seed=seed)
        env.reset()
        rewards = []
        for _ in range(5):
            action = Action(action_type="label_severity", severity="high")
            _, reward, _, _ = env.step(action)
            rewards.append(reward.value)
        return rewards


# ── Exploit Prevention Tests ──────────────────────────────────────────────────

class TestExploitPrevention:
    """Tests that gaming strategies are penalized."""

    def test_exploit_approve_no_comments(self):
        """Approving without comments scores 0.0 on hard task."""
        from graders.grader_hard import HardGrader
        grader = HardGrader()
        reward, _ = grader.grade_pr("PR-001", "approve")
        assert reward.value == 0.0

    def test_exploit_spam_comments(self):
        """Spamming comments scores lower than targeted comments."""
        from graders.grader_hard import HardGrader
        template = PR_TEMPLATES[0]  # Java null pointer
        pr_id = template["pr_id"]

        # Targeted: 3 relevant comments
        grader_targeted = HardGrader()
        for bl in template["bug_lines"][:3]:
            action = Action(
                action_type="add_comment",
                comment="Add null check guard here to prevent NullPointerException",
                target_file=template["filename"],
                target_line=bl,
            )
            grader_targeted.add_comment(pr_id, action)
        reward_targeted, _ = grader_targeted.grade_pr(pr_id, "request_changes")

        # Spam: 15 irrelevant comments
        grader_spam = HardGrader()
        for i in range(15):
            action = Action(
                action_type="add_comment",
                comment=f"Comment {i}",
                target_file=template["filename"],
                target_line=i + 100,
            )
            grader_spam.add_comment(pr_id, action)
        reward_spam, _ = grader_spam.grade_pr(pr_id, "request_changes")

        assert reward_targeted.value > reward_spam.value, (
            f"Targeted ({reward_targeted.value:.3f}) should beat spam ({reward_spam.value:.3f})"
        )

    def test_exploit_random_severity(self):
        """Random agent scores < 0.5 on easy task."""
        random.seed(42)
        scores = []
        for template in PR_TEMPLATES:
            from graders.grader_easy import EasyGrader
            grader = EasyGrader()
            sev = random.choice(["critical", "high", "medium", "low", "none"])
            action = Action(action_type="label_severity", severity=sev)
            reward, _ = grader.grade(action, template["pr_id"])
            scores.append(reward.value)
        assert statistics.mean(scores) < 0.5


# ── Agent Baseline Tests ─────────────────────────────────────────────────────

class TestAgentBaselines:
    """Tests that floor and ceiling agents perform as expected."""

    def test_perfect_agent_easy(self):
        """Perfect agent scores > 0.85 on easy task."""
        from graders.grader_easy import EasyGrader
        scores = []
        for template in PR_TEMPLATES:
            grader = EasyGrader()
            action = Action(
                action_type="label_severity",
                severity=template["ground_truth_severity"],
            )
            reward, _ = grader.grade(action, template["pr_id"])
            scores.append(reward.value)
        assert statistics.mean(scores) > 0.85

    def test_perfect_agent_hard(self):
        """Perfect agent scores > 0.75 on hard task."""
        from graders.grader_hard import HardGrader
        scores = []
        for template in PR_TEMPLATES[:3]:
            grader = HardGrader()
            pr_id = template["pr_id"]
            bug_cat = template["bug_category"]
            sev = template["ground_truth_severity"]

            # Add targeted, specific, actionable comment for each bug line
            from env.data_generator import BUG_KEYWORDS
            keywords = BUG_KEYWORDS.get(bug_cat, [])
            keyword = keywords[0] if keywords else "issue"

            for bl in template["bug_lines"]:
                action = Action(
                    action_type="add_comment",
                    comment=f"Consider adding a {keyword} check here. You should use proper handling to avoid this issue.",
                    target_file=template["filename"],
                    target_line=bl,
                )
                grader.add_comment(pr_id, action)

            decision = "request_changes" if sev != "none" else "approve"
            reward, _ = grader.grade_pr(pr_id, decision)
            scores.append(reward.value)

        assert statistics.mean(scores) > 0.75, f"Perfect agent scored {statistics.mean(scores):.3f}"
