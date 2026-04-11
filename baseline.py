#!/usr/bin/env python3
"""
CodeReviewEnv -- Baseline Agent Script
======================================
Runs a simple heuristic agent (no LLM) against all three difficulty tiers
and reports per-tier scores. Verifies the environment works end-to-end
without requiring any API keys.

For LLM-based evaluation, use inference.py instead.

Usage:
    python baseline.py
"""

import json
import os
import re
import sys
import statistics
import time
from typing import Dict, List, Tuple

from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction


# ── Heuristic Agent ──────────────────────────────────────────────────────────

def heuristic_review(code: str, language: str) -> CodeReviewAction:
    """Simple keyword-based heuristic reviewer. No LLM needed."""
    issues = []
    flagged_lines = []
    suggestions = []
    lines = code.split('\n')

    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()

        # Skip comments
        if line_lower.startswith('#') or line_lower.startswith('//'):
            continue

        # Check for common bug patterns
        if re.search(r'<=\s*len\b', line) or re.search(r'<\s*len\b.*-\s*1', line):
            issues.append(f"Potential off-by-one error on line {i}")
            flagged_lines.append(i)
            suggestions.append(f"Check boundary condition on line {i}")

        # Look for missing null guards (dereference without prior check)
        if re.search(r'\.\w+', line) and 'none' not in line_lower and 'null' not in line_lower:
            if i > 1 and 'if' not in lines[i-2].lower():
                pass  # Could flag but too many false positives

        # Look for suspicious operator patterns
        if re.search(r'\w\s*-\s*\w', line) and 'range(' in line:
            issues.append(f"Suspicious range bound on line {i}")
            flagged_lines.append(i)

    if not issues:
        # Fallback: flag first non-trivial line
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.strip().startswith(('#', '//', 'def ', 'func ', 'function ')):
                issues.append(f"Potential issue on line {i}")
                flagged_lines.append(i)
                break

    suggestion = "; ".join(suggestions) if suggestions else "Review the code for potential bugs."
    comment = f"Found {len(issues)} potential issue(s). " + (suggestions[0] if suggestions else "Please review carefully.")

    return CodeReviewAction(
        issues=issues,
        flagged_lines=flagged_lines,
        suggestion=suggestion,
        comment=comment,
    )


# ── Task Runner ──────────────────────────────────────────────────────────────

def run_episode(difficulty: str, seed: int) -> Tuple[float, int]:
    """Run one multi-step episode with heuristic agent. Returns (reward, steps)."""
    env = CodeReviewEnvironment()
    obs = env.reset(seed=seed, difficulty=difficulty)
    steps = 0

    # Step 1: Analyze
    analyze = CodeReviewAction(action_type="analyze")
    obs = env.step(analyze)
    steps += 1

    # Step 2: Flag suspicious lines
    review = heuristic_review(obs.code, obs.language)
    for line in review.flagged_lines[:2]:
        flag = CodeReviewAction(action_type="flag_line", line=line)
        obs = env.step(flag)
        steps += 1
        if obs.done:
            return obs.reward or 0.0, steps

    # Step 3: Submit full review
    review.action_type = "submit_review"
    result = env.step(review)
    steps += 1
    return result.reward or 0.0, steps


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SEED = 42
    N_EPISODES = 5

    print("=" * 60)
    print("CodeReviewEnv -- Baseline Heuristic Agent")
    print("=" * 60)

    start_time = time.time()
    all_results: Dict[str, Dict] = {}

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n--- {difficulty.upper()} ---")
        scores = []

        for ep in range(N_EPISODES):
            ep_seed = SEED + ep
            score, steps = run_episode(difficulty, ep_seed)
            scores.append(score)
            print(f"  Episode {ep + 1}: score={score:.4f} ({steps} step)")

        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        all_results[difficulty] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "scores": [round(s, 4) for s in scores],
        }
        print(f"  -> Mean: {mean:.4f} +/- {std:.4f}")

    elapsed = time.time() - start_time
    composite = round(
        sum(r["mean"] for r in all_results.values()) / len(all_results), 4
    )

    # Summary Table
    print(f"\n{'=' * 60}")
    print(f"{'Tier':<10} | {'Mean':>8} | {'Std':>8} | {'Scores'}")
    print(f"{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 20}")
    for tier in ["easy", "medium", "hard"]:
        r = all_results[tier]
        scores_str = ", ".join(f"{s:.3f}" for s in r["scores"])
        print(f"{tier:<10} | {r['mean']:>8.4f} | {r['std']:>8.4f} | [{scores_str}]")
    print(f"{'=' * 60}")
    print(f"Composite Score: {composite:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Save Results
    output = {
        "agent": "heuristic_baseline",
        "composite": composite,
        "seed": SEED,
        "episodes_per_tier": N_EPISODES,
        **all_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline", "heuristic_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
