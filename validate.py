#!/usr/bin/env python3
"""
OpenEnv Spec Compliance Validator for CodeReviewEnv

Runs all spec compliance checks. Exit 0 if all pass, exit 1 if any fail.
Includes both standard OpenEnv checks and research-grade statistical validity.
"""

import sys
import json
import os
import random
import statistics

import yaml

from env.base import CodeReviewEnv
from env.models import Action, Observation, Reward, State
from env.data_generator import PR_TEMPLATES


def check(name: str, condition: bool, reason: str = "") -> bool:
    """Print PASS or FAIL with reason."""
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if not condition and reason:
        msg += f" — {reason}"
    print(msg)
    return condition


def validate():
    """Run all validation checks."""
    print("=" * 60)
    print("CodeReviewEnv — OpenEnv Compliance Validation")
    print("=" * 60)
    results = []

    # ── 1. reset() returns valid Observation ────────────────────────
    print("\n--- Core Interface ---")
    for task in ["easy", "medium", "hard"]:
        try:
            env = CodeReviewEnv(task=task, seed=42)
            obs = env.reset()
            results.append(check(
                f"reset() returns Observation ({task})",
                isinstance(obs, Observation),
            ))
        except Exception as e:
            results.append(check(f"reset() returns Observation ({task})", False, str(e)))

    # ── 2. step() with valid action returns correct tuple ───────────
    try:
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        action = Action(action_type="label_severity", severity="high")
        result = env.step(action)
        results.append(check(
            "step() returns (Observation, Reward, bool, Dict)",
            len(result) == 4
            and isinstance(result[0], Observation)
            and isinstance(result[1], Reward)
            and isinstance(result[2], bool)
            and isinstance(result[3], dict),
        ))
    except Exception as e:
        results.append(check("step() returns correct tuple", False, str(e)))

    # ── 3. step() with invalid action doesn't raise ─────────────────
    try:
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        # Create a technically valid Action but with wrong type for task
        action = Action(action_type="approve")
        obs, reward, done, info = env.step(action)
        results.append(check(
            "step() with wrong action type doesn't crash",
            isinstance(reward, Reward) and reward.value <= 0,
        ))
    except Exception as e:
        results.append(check("step() with invalid action doesn't crash", False, str(e)))

    # ── 4. reward.value always in [-1.0, 1.0] ──────────────────────
    try:
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        all_in_range = True
        severities = ["critical", "high", "medium", "low", "none"]
        for sev in severities * 20:
            action = Action(action_type="label_severity", severity=sev)
            env2 = CodeReviewEnv(task="easy", seed=random.randint(1, 1000))
            env2.reset()
            _, reward, _, _ = env2.step(action)
            if reward.value < -1.0 or reward.value > 1.0:
                all_in_range = False
                break
        results.append(check(
            "reward.value always in [-1.0, 1.0]",
            all_in_range,
        ))
    except Exception as e:
        results.append(check("reward bounds", False, str(e)))

    # ── 5. done=True after episode_length steps ─────────────────────
    try:
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        done = False
        for i in range(5):
            action = Action(action_type="label_severity", severity="medium")
            _, _, done, _ = env.step(action)
        results.append(check(
            "done=True after episode_length steps (easy=5)",
            done is True,
        ))
    except Exception as e:
        results.append(check("episode terminates", False, str(e)))

    # ── 6. state() returns State with correct trajectory length ─────
    try:
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        for i in range(3):
            action = Action(action_type="label_severity", severity="high")
            env.step(action)
        s = env.state()
        results.append(check(
            "state() trajectory length matches step count",
            isinstance(s, State) and len(s.trajectory) == 3,
            f"Expected 3, got {len(s.trajectory) if isinstance(s, State) else 'N/A'}",
        ))
    except Exception as e:
        results.append(check("state() correct", False, str(e)))

    # ── 7. export_trajectory() format ───────────────────────────────
    try:
        env = CodeReviewEnv(task="easy", seed=42)
        env.reset()
        for i in range(5):
            action = Action(action_type="label_severity", severity="high")
            env.step(action)
        traj = env.export_trajectory()
        required_keys = {"step", "state", "action", "reward", "next_state"}
        has_keys = all(required_keys <= set(t.keys()) for t in traj)
        results.append(check(
            "export_trajectory() has required keys",
            len(traj) > 0 and has_keys,
        ))
    except Exception as e:
        results.append(check("export_trajectory() format", False, str(e)))

    # ── 8-10. Grader score ranges ───────────────────────────────────
    print("\n--- Grader Validation ---")

    # Easy grader
    try:
        from graders.grader_easy import EasyGrader
        grader = EasyGrader()
        all_valid = True
        for template in PR_TEMPLATES:
            grader.reset()
            for sev in ["critical", "high", "medium", "low", "none"]:
                action = Action(action_type="label_severity", severity=sev)
                reward, _ = grader.grade(action, template["pr_id"])
                if reward.value < -1.0 or reward.value > 1.0:
                    all_valid = False
        results.append(check("grader_easy scores in [-1, 1]", all_valid))
    except Exception as e:
        results.append(check("grader_easy range", False, str(e)))

    # Medium grader
    try:
        from graders.grader_medium import MediumGrader
        from env.data_generator import DataGenerator
        grader = MediumGrader()
        gen = DataGenerator(seed=42)
        queues = gen.generate_medium_episode(num_queues=3, queue_size=5)
        all_valid = True
        for queue in queues:
            gt_order = gen.compute_priority_order(queue)
            # Test with correct order
            action = Action(action_type="prioritize", priority_order=gt_order)
            reward, _ = grader.grade(action, queue, gt_order)
            if reward.value < 0.0 or reward.value > 1.0:
                all_valid = False
            # Test with reversed order
            action = Action(action_type="prioritize", priority_order=list(reversed(gt_order)))
            reward, _ = grader.grade(action, queue, gt_order)
            if reward.value < 0.0 or reward.value > 1.0:
                all_valid = False
        results.append(check("grader_medium scores in [0, 1]", all_valid))
    except Exception as e:
        results.append(check("grader_medium range", False, str(e)))

    # Hard grader
    try:
        from graders.grader_hard import HardGrader
        grader = HardGrader()
        all_valid = True
        for template in PR_TEMPLATES[:5]:
            grader.reset()
            pr_id = template["pr_id"]
            bug_lines = template["bug_lines"]
            if bug_lines:
                comment_action = Action(
                    action_type="add_comment",
                    comment="Consider adding null check here to prevent crash",
                    target_file=template["filename"],
                    target_line=bug_lines[0],
                )
                grader.add_comment(pr_id, comment_action)
            reward, _ = grader.grade_pr(pr_id, "request_changes")
            if reward.value < -1.0 or reward.value > 1.0:
                all_valid = False
        results.append(check("grader_hard scores in [-1, 1]", all_valid))
    except Exception as e:
        results.append(check("grader_hard range", False, str(e)))

    # ── 11. Score variance check ────────────────────────────────────
    print("\n--- Statistical Validity ---")
    try:
        scores = []
        for template in PR_TEMPLATES:
            grader = EasyGrader()
            # Test with random severity
            sev = random.choice(["critical", "high", "medium", "low", "none"])
            action = Action(action_type="label_severity", severity=sev)
            reward, _ = grader.grade(action, template["pr_id"])
            scores.append(reward.value)
        std = statistics.stdev(scores)
        results.append(check(
            f"Score variance: std={std:.3f} > 0.05",
            std > 0.05,
        ))
    except Exception as e:
        results.append(check("score variance", False, str(e)))

    # ── 12. Random agent < perfect agent ────────────────────────────
    try:
        # Perfect agent
        perfect_scores = []
        for template in PR_TEMPLATES:
            grader = EasyGrader()
            action = Action(action_type="label_severity", severity=template["ground_truth_severity"])
            reward, _ = grader.grade(action, template["pr_id"])
            perfect_scores.append(reward.value)
        perfect_mean = statistics.mean(perfect_scores)

        # Random agent
        random.seed(42)
        random_scores = []
        for template in PR_TEMPLATES:
            grader = EasyGrader()
            sev = random.choice(["critical", "high", "medium", "low", "none"])
            action = Action(action_type="label_severity", severity=sev)
            reward, _ = grader.grade(action, template["pr_id"])
            random_scores.append(reward.value)
        random_mean = statistics.mean(random_scores)

        gap = perfect_mean - random_mean
        results.append(check(
            f"Perfect ({perfect_mean:.2f}) > Random ({random_mean:.2f}) by {gap:.2f} (>0.3)",
            gap > 0.3,
        ))
    except Exception as e:
        results.append(check("perfect vs random gap", False, str(e)))

    # ── 13. Reproducibility ─────────────────────────────────────────
    try:
        env1 = CodeReviewEnv(task="easy", seed=42)
        obs1 = env1.reset()
        traj1 = []
        for _ in range(5):
            action = Action(action_type="label_severity", severity="high")
            _, reward, _, _ = env1.step(action)
            traj1.append(reward.value)

        env2 = CodeReviewEnv(task="easy", seed=42)
        obs2 = env2.reset()
        traj2 = []
        for _ in range(5):
            action = Action(action_type="label_severity", severity="high")
            _, reward, _, _ = env2.step(action)
            traj2.append(reward.value)

        results.append(check(
            "seed=42 reproducibility",
            obs1.pr_id == obs2.pr_id and traj1 == traj2,
        ))
    except Exception as e:
        results.append(check("reproducibility", False, str(e)))

    # ── 14. openenv.yaml valid ──────────────────────────────────────
    print("\n--- Packaging ---")
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        required = ["name", "version", "description", "tasks"]
        has_all = all(k in config for k in required)
        results.append(check(
            "openenv.yaml is valid with required fields",
            has_all,
        ))
    except Exception as e:
        results.append(check("openenv.yaml", False, str(e)))

    # ── 15. Dockerfile exists ───────────────────────────────────────
    results.append(check(
        "Dockerfile exists",
        os.path.exists("Dockerfile"),
    ))

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("✅ ALL CHECKS PASSED — OpenEnv compliant")
        return 0
    else:
        print(f"❌ {total - passed} checks FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(validate())
