#!/usr/bin/env python3
"""
Baseline Agent — GPT-4o-mini on CodeReviewEnv

Runs the GPT-4o-mini model against all three tasks and records scores.
Requires OPENAI_API_KEY environment variable.

Usage:
    OPENAI_API_KEY=sk-... python baseline/run_baseline.py
"""

import json
import os
import sys
import statistics
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.base import CodeReviewEnv
from env.models import Action


def run_baseline():
    """Run GPT-4o-mini baseline across all tasks via OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY or OPENAI_API_KEY not set.")
        print("Usage: OPENROUTER_API_KEY=sk-... python baseline/run_baseline.py")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    model = "openai/gpt-4o-mini"
    seed = 42
    n_episodes = 3

    results = {}

    for task in ["easy", "medium", "hard"]:
        print(f"\n{'='*40}")
        print(f"Running {task} task — {n_episodes} episodes")
        print(f"{'='*40}")

        episode_scores = []

        for ep in range(n_episodes):
            episode_seed = seed + ep
            env = CodeReviewEnv(task=task, seed=episode_seed)
            obs = env.reset()
            system_prompt = env.get_system_prompt()

            step_rewards = []
            done = False
            max_steps = 50

            while not done and len(step_rewards) < max_steps:
                # Build user message from observation
                user_msg = json.dumps(obs.model_dump(), indent=2, default=str)

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0,
                        seed=seed,
                        max_tokens=500,
                    )

                    response_text = response.choices[0].message.content.strip()

                    # Try to parse JSON — handle common LLM output quirks
                    # 1. Strip markdown code blocks
                    if "```" in response_text:
                        import re
                        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
                        if code_match:
                            response_text = code_match.group(1).strip()

                    # 2. Extract first JSON object (ignore trailing explanation text)
                    brace_start = response_text.find("{")
                    if brace_start >= 0:
                        depth = 0
                        for i, ch in enumerate(response_text[brace_start:], start=brace_start):
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    response_text = response_text[brace_start:i+1]
                                    break

                    action_dict = json.loads(response_text)
                    action = Action(**action_dict)

                except Exception as e:
                    print(f"  Parse error at step {len(step_rewards)}: {e}")
                    # Fallback action
                    if task == "easy":
                        action = Action(action_type="label_severity", severity="none")
                    elif task == "medium":
                        action = Action(
                            action_type="prioritize",
                            priority_order=obs.review_queue,
                        )
                    else:
                        action = Action(action_type="approve")

                obs, reward, done, info = env.step(action)
                step_rewards.append(reward.value)

            ep_score = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
            episode_scores.append(round(ep_score, 2))
            print(f"  Episode {ep + 1}: score={ep_score:.3f} ({len(step_rewards)} steps)")

        mean_score = statistics.mean(episode_scores)
        std_score = statistics.stdev(episode_scores) if len(episode_scores) > 1 else 0.0

        results[task] = {
            "mean": round(mean_score, 2),
            "std": round(std_score, 2),
            "episodes": episode_scores,
        }

        print(f"\n  {task}: mean={mean_score:.3f} std={std_score:.3f}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Task':<10} | {'Episodes':>8} | {'Mean':>6} | {'Std':>6} | {'Min':>6} | {'Max':>6}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for task in ["easy", "medium", "hard"]:
        r = results[task]
        eps = r["episodes"]
        print(
            f"{task:<10} | {len(eps):>8} | {r['mean']:>6.2f} | {r['std']:>6.2f} | "
            f"{min(eps):>6.2f} | {max(eps):>6.2f}"
        )

    # Save results
    output = {
        **results,
        "model": model,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_baseline()
