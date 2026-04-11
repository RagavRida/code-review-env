#!/usr/bin/env python3
"""
CodeReviewEnv -- Baseline Agent (MCP tool-calling pattern)
==========================================================
Runs a heuristic agent using the MCP tool-calling interface:
  1. get_code → retrieve code
  2. analyze_code → structural analysis
  3. check_line × N → flag suspicious lines
  4. submit_review → final review

No LLM required. Uses keyword heuristics.
"""

import json
import os
import re
import statistics
import time
from typing import Dict, List, Tuple

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction


def _tool_call(tool_name: str, arguments: Dict = None) -> CodeReviewAction:
    """Helper to create a ToolCallAction."""
    return CodeReviewAction(
        action_type="ToolCallAction",
        tool_name=tool_name,
        arguments=arguments or {},
    )


def heuristic_find_bugs(code: str) -> Tuple[List[str], List[int], str]:
    """Simple keyword-based bug detection. Returns (issues, lines, suggestion)."""
    issues = []
    flagged = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        stripped = line.strip().lower()
        if stripped.startswith('#') or stripped.startswith('//'):
            continue

        if re.search(r'<=\s*len\b', line) or re.search(r'range\(.*-\s*1\)', line):
            issues.append(f"Potential off-by-one error on line {i}")
            flagged.append(i)
        if re.search(r'\w\s*-\s*\w', line) and 'range(' in line:
            issues.append(f"Suspicious range bound on line {i}")
            flagged.append(i)

    if not issues:
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.strip().startswith(('#', '//', 'def ', 'func ')):
                issues.append(f"Potential issue on line {i}")
                flagged.append(i)
                break

    suggestion = f"Check boundary conditions on lines {flagged}" if flagged else "Review carefully"
    return issues, flagged, suggestion


def run_episode(difficulty: str, seed: int) -> Tuple[float, int]:
    """Run one episode using MCP tool-calling."""
    env = CodeReviewEnvironment()
    env.reset(seed=seed, difficulty=difficulty)
    steps = 0

    # Step 1: Get code
    obs = env.step(_tool_call("get_code"))
    steps += 1
    code = obs.tool_result.get("code", "") if obs.tool_result else ""

    # Step 2: Analyze
    obs = env.step(_tool_call("analyze_code"))
    steps += 1

    # Step 3: Find bugs heuristically and check lines
    issues, flagged, suggestion = heuristic_find_bugs(code)
    for line in flagged[:2]:
        obs = env.step(_tool_call("check_line", {"line": line}))
        steps += 1
        if obs.done:
            return obs.reward or 0.0, steps

    # Step 4: Submit review
    obs = env.step(_tool_call("submit_review", {
        "issues": issues,
        "flagged_lines": flagged,
        "suggestion": suggestion,
        "comment": f"Found {len(issues)} potential issue(s).",
    }))
    steps += 1
    return obs.reward or 0.0, steps


def main():
    SEED = 42
    N_EPISODES = 5

    print("=" * 60)
    print("CodeReviewEnv -- Baseline Agent (MCP tool-calling)")
    print("=" * 60)

    start = time.time()
    results: Dict[str, Dict] = {}

    for diff in ["easy", "medium", "hard"]:
        print(f"\n--- {diff.upper()} ---")
        scores = []
        for ep in range(N_EPISODES):
            score, steps = run_episode(diff, SEED + ep)
            scores.append(score)
            print(f"  Episode {ep + 1}: score={score:.4f} ({steps} steps)")

        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        results[diff] = {"mean": round(mean, 4), "std": round(std, 4), "scores": [round(s, 4) for s in scores]}
        print(f"  -> Mean: {mean:.4f} +/- {std:.4f}")

    elapsed = time.time() - start
    composite = round(sum(r["mean"] for r in results.values()) / len(results), 4)

    print(f"\n{'=' * 60}")
    for tier in ["easy", "medium", "hard"]:
        r = results[tier]
        print(f"{tier:<10} | {r['mean']:>8.4f} | {r['std']:>8.4f}")
    print(f"{'=' * 60}")
    print(f"Composite: {composite:.4f} | Elapsed: {elapsed:.1f}s")

    os.makedirs("baseline", exist_ok=True)
    with open("baseline/heuristic_results.json", "w") as f:
        json.dump({"agent": "heuristic_mcp_baseline", "composite": composite, **results}, f, indent=2)


if __name__ == "__main__":
    main()
