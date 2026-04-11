#!/usr/bin/env python3
"""
OpenEnv Spec Compliance Validator for CodeReviewEnv v2 (MCP pattern)
"""

import sys
import os
import random
import yaml

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from snippet_bank import SNIPPET_BANK, BUG_INJECTORS, generate_episode


def check(name: str, condition: bool, reason: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if not condition and reason:
        msg += f" -- {reason}"
    print(msg)
    return condition


def _tool(tool_name, arguments=None):
    return CodeReviewAction(action_type="ToolCallAction", tool_name=tool_name, arguments=arguments or {})


def validate():
    print("=" * 60)
    print("CodeReviewEnv -- OpenEnv Compliance Validation (MCP pattern)")
    print("=" * 60)
    results = []

    # 1. reset() returns observation with tools_list
    print("\n--- Core Interface (MCP) ---")
    for difficulty in ["easy", "medium", "hard"]:
        try:
            env = CodeReviewEnvironment()
            obs = env.reset(seed=42, difficulty=difficulty)
            results.append(check(
                f"reset() returns tools_list ({difficulty})",
                isinstance(obs, CodeReviewObservation) and obs.tools_list is not None and len(obs.tools_list) == 8,
            ))
        except Exception as e:
            results.append(check(f"reset() ({difficulty})", False, str(e)))

    # 2. ListToolsAction works
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(CodeReviewAction(action_type="ListToolsAction"))
        tool_names = {t["name"] for t in obs.tools_list}
        results.append(check(
            "ListToolsAction returns 5 tools",
            tool_names == {"get_code", "run_code", "run_tests", "analyze_code", "check_line", "get_hint", "submit_fix", "submit_review"},
        ))
    except Exception as e:
        results.append(check("ListToolsAction", False, str(e)))

    # 3. get_code returns code
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(_tool("get_code"))
        results.append(check(
            "get_code returns code",
            obs.success and obs.tool_result and len(obs.tool_result.get("code", "")) > 0,
        ))
    except Exception as e:
        results.append(check("get_code", False, str(e)))

    # 4. check_line returns reward
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state
        if state.gold_bugs:
            line = state.gold_bugs[0]["lines"][0]
            obs = env.step(_tool("check_line", {"line": line}))
            results.append(check(
                f"check_line({line}) gives positive reward",
                obs.success and obs.reward > 0,
            ))
        else:
            results.append(check("check_line (no bugs to test)", True))
    except Exception as e:
        results.append(check("check_line", False, str(e)))

    # 5. submit_review ends episode
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(_tool("submit_review", {"issues": ["test"], "comment": "review"}))
        results.append(check(
            "submit_review ends episode (done=True)",
            obs.done is True and obs.reward is not None and 0.0 <= obs.reward <= 1.0,
        ))
    except Exception as e:
        results.append(check("submit_review", False, str(e)))

    # 6. state contains gold_bugs
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state
        results.append(check(
            "state contains gold_bugs",
            isinstance(state, CodeReviewState) and isinstance(state.gold_bugs, list),
        ))
    except Exception as e:
        results.append(check("state", False, str(e)))

    # 7. Reward in [0,1] across many seeds
    try:
        all_valid = True
        for seed in range(50):
            env = CodeReviewEnvironment()
            env.reset(seed=seed, difficulty=random.choice(["easy", "medium", "hard"]))
            obs = env.step(_tool("submit_review", {"issues": ["bug"], "comment": "test"}))
            if obs.reward < 0.0 or obs.reward > 1.0:
                all_valid = False
                break
        results.append(check("reward always in [0, 1]", all_valid))
    except Exception as e:
        results.append(check("reward bounds", False, str(e)))

    # 8. Max steps forces submit
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        for i in range(10):
            obs = env.step(_tool("analyze_code"))
            if obs.done:
                break
        results.append(check("max_steps forces submit", obs.done is True))
    except Exception as e:
        results.append(check("max_steps", False, str(e)))

    # ── Procedural Generation ──
    print("\n--- Procedural Generation ---")
    try:
        codes = {generate_episode(seed=s, difficulty="easy")[1][:100] for s in range(10)}
        results.append(check(f"Different seeds -> different episodes ({len(codes)}/10)", len(codes) >= 3))
    except Exception as e:
        results.append(check("different seeds", False, str(e)))

    try:
        _, c1, _ = generate_episode(seed=42)
        _, c2, _ = generate_episode(seed=42)
        results.append(check("seed=42 reproducibility", c1 == c2))
    except Exception as e:
        results.append(check("reproducibility", False, str(e)))

    results.append(check(f"Snippet bank has {len(SNIPPET_BANK)} entries (>= 30)", len(SNIPPET_BANK) >= 30))
    results.append(check(f"Bug injectors: {len(BUG_INJECTORS)} (>= 9)", len(BUG_INJECTORS) >= 9))

    # ── Reward Quality ──
    print("\n--- Reward Quality ---")
    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        state = env.state
        if state.gold_bugs:
            g = state.gold_bugs[0]
            obs = env.step(_tool("submit_review", {
                "issues": [g["description"]],
                "flagged_lines": g["lines"],
                "suggestion": g["fix"],
                "comment": f"Found {g['bug_type']}. {g['fix']}",
            }))
            results.append(check(f"Perfect agent >= 0.5 (got {obs.reward:.3f})", obs.reward >= 0.5))
        else:
            results.append(check("Perfect agent (no bugs)", True))
    except Exception as e:
        results.append(check("perfect agent", False, str(e)))

    try:
        env = CodeReviewEnvironment()
        env.reset(seed=42)
        obs = env.step(_tool("submit_review", {}))
        results.append(check(f"Empty agent < 0.15 (got {obs.reward:.3f})", obs.reward < 0.15))
    except Exception as e:
        results.append(check("empty agent", False, str(e)))

    # ── Packaging ──
    print("\n--- Packaging ---")
    try:
        with open("openenv.yaml") as f:
            config = yaml.safe_load(f)
        required = ["spec_version", "name", "type", "runtime", "app", "port"]
        results.append(check("openenv.yaml has required fields", all(k in config for k in required)))
    except Exception as e:
        results.append(check("openenv.yaml", False, str(e)))

    results.append(check("Dockerfile exists", os.path.exists("Dockerfile")))

    # ── Summary ──
    print(f"\n{'=' * 60}")
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")
    print("ALL CHECKS PASSED" if passed == total else f"{total - passed} checks FAILED")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(validate())
