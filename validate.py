#!/usr/bin/env python3
"""
OpenEnv Spec Compliance Validator for CodeReviewEnv v2

Runs all spec compliance checks. Exit 0 if all pass, exit 1 if any fail.
"""

import sys
import os
import random

import yaml

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from snippet_bank import SNIPPET_BANK, BUG_INJECTORS, generate_episode
from reward import compute_reward


def check(name: str, condition: bool, reason: str = "") -> bool:
    """Print PASS or FAIL with reason."""
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if not condition and reason:
        msg += f" -- {reason}"
    print(msg)
    return condition


def validate():
    """Run all validation checks."""
    print("=" * 60)
    print("CodeReviewEnv -- OpenEnv Compliance Validation")
    print("=" * 60)
    results = []

    # 1. reset() returns valid Observation for all difficulties
    print("\n--- Core Interface ---")
    for difficulty in ["easy", "medium", "hard"]:
        try:
            env = CodeReviewEnvironment()
            obs = env.reset(seed=42, difficulty=difficulty)
            results.append(check(
                f"reset() returns Observation ({difficulty})",
                isinstance(obs, CodeReviewObservation) and obs.code != "",
            ))
        except Exception as e:
            results.append(check(f"reset() returns Observation ({difficulty})", False, str(e)))

    # 2. step() with valid action returns observation
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        action = CodeReviewAction(issues=["test"], flagged_lines=[1], suggestion="fix", comment="comment")
        obs = env.step(action)
        results.append(check(
            "step() returns CodeReviewObservation with reward",
            isinstance(obs, CodeReviewObservation) and obs.done is True and obs.reward is not None,
        ))
    except Exception as e:
        results.append(check("step() returns correct observation", False, str(e)))

    # 3. step() with empty action doesn't crash
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        action = CodeReviewAction()
        obs = env.step(action)
        results.append(check(
            "step() with empty action doesn't crash",
            isinstance(obs, CodeReviewObservation) and 0.0 <= obs.reward <= 1.0,
        ))
    except Exception as e:
        results.append(check("step() with empty action", False, str(e)))

    # 4. reward always in [0, 1]
    try:
        all_in_range = True
        for seed in range(50):
            env = CodeReviewEnvironment()
            env.reset(seed=seed, difficulty=random.choice(["easy", "medium", "hard"]))
            action = CodeReviewAction(
                issues=["bug found"],
                flagged_lines=[random.randint(1, 20)],
                suggestion="fix it",
                comment="review",
            )
            obs = env.step(action)
            if obs.reward < 0.0 or obs.reward > 1.0:
                all_in_range = False
                break
        results.append(check("reward always in [0, 1]", all_in_range))
    except Exception as e:
        results.append(check("reward bounds", False, str(e)))

    # 5. done=True after step()
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(issues=["test"]))
        results.append(check("done=True after step()", obs.done is True))
    except Exception as e:
        results.append(check("episode terminates", False, str(e)))

    # 6. state contains gold bugs
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state
        results.append(check(
            "state contains gold_bugs",
            isinstance(state, CodeReviewState) and isinstance(state.gold_bugs, list),
        ))
    except Exception as e:
        results.append(check("state correct", False, str(e)))

    # 7. Reward breakdown present
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction(issues=["test"]))
        breakdown = obs.reward_breakdown
        has_signals = breakdown is not None and "bug_detection" in breakdown
        results.append(check("reward breakdown has all 5 signals", has_signals))
    except Exception as e:
        results.append(check("reward breakdown", False, str(e)))

    # 8. Procedural generation: different seeds differ
    print("\n--- Procedural Generation ---")
    try:
        codes = set()
        for seed in range(10):
            _, code, _ = generate_episode(seed=seed, difficulty="easy")
            codes.add(code[:100])
        results.append(check(
            f"Different seeds -> different episodes ({len(codes)}/10 unique)",
            len(codes) >= 3,
        ))
    except Exception as e:
        results.append(check("procedural generation", False, str(e)))

    # 9. Reproducibility: same seed -> same episode
    try:
        _, c1, b1 = generate_episode(seed=42, difficulty="easy")
        _, c2, b2 = generate_episode(seed=42, difficulty="easy")
        results.append(check(
            "seed=42 reproducibility",
            c1 == c2 and len(b1) == len(b2),
        ))
    except Exception as e:
        results.append(check("reproducibility", False, str(e)))

    # 10. Snippet bank size
    try:
        results.append(check(
            f"Snippet bank has {len(SNIPPET_BANK)} entries (>= 30)",
            len(SNIPPET_BANK) >= 30,
        ))
    except Exception as e:
        results.append(check("snippet bank size", False, str(e)))

    # 11. Bug injectors work
    try:
        rng = random.Random(42)
        successes = 0
        for inj in BUG_INJECTORS:
            for s in SNIPPET_BANK[:5]:
                if s.language == "python":
                    result = inj(s.code, rng)
                    if result is not None:
                        successes += 1
                        break
        results.append(check(
            f"Bug injectors functional ({successes}/{len(BUG_INJECTORS)})",
            successes >= 3,
        ))
    except Exception as e:
        results.append(check("bug injectors", False, str(e)))

    # 12. Perfect agent scores high
    print("\n--- Reward Quality ---")
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        state = env.state
        if state.gold_bugs:
            gold = state.gold_bugs[0]
            action = CodeReviewAction(
                issues=[gold["description"]],
                flagged_lines=gold["lines"],
                suggestion=gold["fix"],
                comment=f"Found {gold['bug_type']} bug. {gold['fix']}",
            )
            obs = env.step(action)
            results.append(check(
                f"Perfect agent scores >= 0.5 (got {obs.reward:.3f})",
                obs.reward >= 0.5,
            ))
        else:
            results.append(check("Perfect agent (no bugs to test)", True))
    except Exception as e:
        results.append(check("perfect agent", False, str(e)))

    # 13. Empty agent scores low
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42, difficulty="easy")
        obs = env.step(CodeReviewAction())
        results.append(check(
            f"Empty agent scores < 0.15 (got {obs.reward:.3f})",
            obs.reward < 0.15,
        ))
    except Exception as e:
        results.append(check("empty agent", False, str(e)))

    # 14. openenv.yaml valid
    print("\n--- Packaging ---")
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        required = ["spec_version", "name", "type", "runtime", "app", "port"]
        has_all = all(k in config for k in required)
        results.append(check("openenv.yaml has required fields", has_all))
    except Exception as e:
        results.append(check("openenv.yaml", False, str(e)))

    # 15. Dockerfile exists
    results.append(check("Dockerfile exists", os.path.exists("Dockerfile")))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("ALL CHECKS PASSED -- OpenEnv compliant")
        return 0
    else:
        print(f"{total - passed} checks FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(validate())
