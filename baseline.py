#!/usr/bin/env python3
"""
CodeReviewEnv — Baseline Agent Script
======================================
Runs a simple heuristic agent (no LLM) against all three tasks
and reports per-task scores. This verifies the environment works
end-to-end without requiring any API keys.

For LLM-based evaluation, use inference.py instead.

Usage:
    python baseline.py
"""

import json
import os
import sys
import statistics
import time
from typing import Dict, List, Tuple

from env.base import CodeReviewEnv
from env.models import Action


# ─── Heuristic Agent ────────────────────────────────────────────────────────

def heuristic_easy_action(obs) -> Action:
    """Simple heuristic: guess severity based on keywords in diff."""
    diff_text = ""
    for f in obs.files:
        diff_text += f.diff.lower()

    if any(kw in diff_text for kw in ["injection", "secret", "hardcoded", "plaintext", "md5"]):
        severity = "critical"
    elif any(kw in diff_text for kw in ["null", "none", "nil", "race", "mutex", "lock"]):
        severity = "high"
    elif any(kw in diff_text for kw in ["bug", "error", "exception", "off-by-one", "boundary"]):
        severity = "medium"
    elif any(kw in diff_text for kw in ["o(n)", "performance", "loop", "cache", "index"]):
        severity = "low"
    else:
        severity = "none"

    return Action(action_type="label_severity", severity=severity)


def heuristic_medium_action(obs) -> Action:
    """Simple heuristic: return queue in original order (no reordering)."""
    return Action(action_type="prioritize", priority_order=list(obs.review_queue))


def heuristic_hard_action(obs, step_in_pr: int) -> Action:
    """Simple heuristic: add one generic comment then request_changes."""
    if step_in_pr == 0 and obs.files:
        f = obs.files[0]
        return Action(
            action_type="add_comment",
            comment="Consider reviewing this section for potential issues.",
            target_file=f.filename,
            target_line=10,
        )
    return Action(action_type="request_changes")


# ─── Task Runners ───────────────────────────────────────────────────────────

def run_easy_episode(seed: int) -> Tuple[float, int]:
    """Run one easy episode. Returns (mean_reward, steps)."""
    env = CodeReviewEnv(task="easy", seed=seed)
    obs = env.reset()
    rewards = []
    done = False

    while not done:
        action = heuristic_easy_action(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward.value)

    mean = statistics.mean(rewards) if rewards else 0.0
    return mean, len(rewards)


def run_medium_episode(seed: int) -> Tuple[float, int]:
    """Run one medium episode. Returns (mean_reward, steps)."""
    env = CodeReviewEnv(task="medium", seed=seed)
    obs = env.reset()
    rewards = []
    done = False

    while not done:
        action = heuristic_medium_action(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward.value)

    mean = statistics.mean(rewards) if rewards else 0.0
    return mean, len(rewards)


def run_hard_episode(seed: int) -> Tuple[float, int]:
    """Run one hard episode. Returns (mean_reward, steps)."""
    env = CodeReviewEnv(task="hard", seed=seed)
    obs = env.reset()
    rewards = []
    done = False
    step_in_pr = 0

    while not done:
        action = heuristic_hard_action(obs, step_in_pr)
        obs, reward, done, info = env.step(action)
        rewards.append(reward.value)

        if action.action_type in ("approve", "request_changes"):
            step_in_pr = 0
        else:
            step_in_pr += 1

    # Filter out comment acks for PR-level scoring
    pr_rewards = [r for r in rewards if abs(r - 0.05) > 0.01]
    mean = statistics.mean(pr_rewards) if pr_rewards else 0.0
    return mean, len(rewards)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    SEED = 42
    N_EPISODES = 3

    print("=" * 60)
    print("CodeReviewEnv — Baseline Heuristic Agent")
    print("=" * 60)

    start_time = time.time()
    all_results: Dict[str, Dict] = {}

    for task_name, runner in [("easy", run_easy_episode), ("medium", run_medium_episode), ("hard", run_hard_episode)]:
        print(f"\n--- {task_name.upper()} Task ---")
        scores = []

        for ep in range(N_EPISODES):
            ep_seed = SEED + ep
            score, steps = runner(ep_seed)
            scores.append(score)
            print(f"  Episode {ep + 1}: score={score:.4f} ({steps} steps)")

        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        all_results[task_name] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "scores": [round(s, 4) for s in scores],
        }
        print(f"  → Mean: {mean:.4f} ± {std:.4f}")

    elapsed = time.time() - start_time
    composite = round(
        sum(r["mean"] for r in all_results.values()) / len(all_results), 4
    )

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'Task':<10} | {'Mean':>8} | {'Std':>8} | {'Scores'}")
    print(f"{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 20}")
    for task in ["easy", "medium", "hard"]:
        r = all_results[task]
        scores_str = ", ".join(f"{s:.3f}" for s in r["scores"])
        print(f"{task:<10} | {r['mean']:>8.4f} | {r['std']:>8.4f} | [{scores_str}]")
    print(f"{'=' * 60}")
    print(f"Composite Score: {composite:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")

    # ── Save Results ─────────────────────────────────────────────────
    output = {
        "agent": "heuristic_baseline",
        "composite": composite,
        "seed": SEED,
        "episodes_per_task": N_EPISODES,
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
