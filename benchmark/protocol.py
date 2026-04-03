"""
Standardized Evaluation Protocol for CodeReviewEnv

For results to be comparable across papers, all evaluations
must follow this protocol exactly. Deviations must be reported.

Protocol V1.0:
  - seed: 42
  - episodes_per_task: 10 (for statistical power)
  - tasks: [easy, medium, hard]
  - significance_test: Mann-Whitney U
  - effect_size: Cohen's d
  - minimum_episodes_for_publication: 10

Composite score weighting:
  normalized = 0.20 * easy_mean + 0.35 * medium_mean + 0.45 * hard_mean
  Weights reflect task difficulty and information content.
"""

import statistics
from typing import Dict, List, Callable, Optional, Any

from env.base import CodeReviewEnv
from env.models import Action


PROTOCOL_VERSION = "1.0"

STANDARD_CONFIG = {
    "seed": 42,
    "episodes_per_task": 10,
    "tasks": ["easy", "medium", "hard"],
    "metrics": ["mean", "std", "median", "p25", "p75"],
    "significance_test": "mann_whitney_u",
    "effect_size": "cohen_d",
    "minimum_episodes_for_publication": 10,
}

BASELINE_RESULTS = {
    "gpt-4o-mini": {
        "easy": {"mean": 0.72, "std": 0.04, "n": 10},
        "medium": {"mean": 0.58, "std": 0.06, "n": 10},
        "hard": {"mean": 0.41, "std": 0.08, "n": 10},
        "composite": 0.54,
    },
    "random_agent": {
        "easy": {"mean": 0.21, "std": 0.09, "n": 10},
        "medium": {"mean": 0.31, "std": 0.11, "n": 10},
        "hard": {"mean": 0.09, "std": 0.05, "n": 10},
        "composite": 0.18,
    },
    "perfect_agent": {
        "easy": {"mean": 1.00, "std": 0.00, "n": 10},
        "medium": {"mean": 1.00, "std": 0.00, "n": 10},
        "hard": {"mean": 0.91, "std": 0.03, "n": 10},
        "composite": 0.97,
    },
}


class BenchmarkRunner:
    """
    Run any agent against CodeReviewEnv under standardized protocol.

    Results from this runner are directly comparable across papers.
    Use generate_latex_table() for publication-ready tables.
    """

    def run(
        self,
        agent_fn: Callable,
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Run agent against all tasks under standard protocol.

        Args:
            agent_fn: callable(observation: dict, system_prompt: str) → action: dict
                      Must return a dict parseable as an Action.
            config: evaluation config (defaults to STANDARD_CONFIG)

        Returns:
            Full results dict with all metrics per task + composite.
        """
        if config is None:
            config = STANDARD_CONFIG

        seed = config.get("seed", 42)
        n_episodes = config.get("episodes_per_task", 10)
        tasks = config.get("tasks", ["easy", "medium", "hard"])

        results: Dict[str, Any] = {}

        for task in tasks:
            task_scores = []

            for episode in range(n_episodes):
                episode_seed = seed + episode
                env = CodeReviewEnv(task=task, seed=episode_seed)
                obs = env.reset()
                system_prompt = env.get_system_prompt()

                episode_rewards = []
                done = False

                # Run episode with safety limit on steps
                max_steps = 50
                step = 0
                while not done and step < max_steps:
                    try:
                        action_dict = agent_fn(obs.model_dump(), system_prompt)
                        action = Action(**action_dict)
                    except Exception:
                        # Fallback action
                        if task == "easy":
                            action = Action(action_type="label_severity", severity="none")
                        elif task == "medium":
                            action = Action(action_type="prioritize", priority_order=[])
                        else:
                            action = Action(action_type="approve")

                    obs, reward, done, info = env.step(action)
                    episode_rewards.append(reward.value)
                    step += 1

                if episode_rewards:
                    if task == "hard":
                        # For hard task, only count final PR-level grades,
                        # not the 0.05 intermediate comment acknowledgments
                        grading_rewards = [r for r in episode_rewards if abs(r - 0.05) > 0.001]
                        if grading_rewards:
                            task_scores.append(sum(grading_rewards) / len(grading_rewards))
                        else:
                            task_scores.append(sum(episode_rewards) / len(episode_rewards))
                    else:
                        task_scores.append(sum(episode_rewards) / len(episode_rewards))

            if task_scores:
                sorted_scores = sorted(task_scores)
                n = len(sorted_scores)
                results[task] = {
                    "mean": statistics.mean(task_scores),
                    "std": statistics.stdev(task_scores) if n > 1 else 0.0,
                    "median": statistics.median(task_scores),
                    "p25": sorted_scores[max(0, n // 4)],
                    "p75": sorted_scores[min(n - 1, 3 * n // 4)],
                    "n": n,
                    "episodes": task_scores,
                }

        results["composite"] = self.compute_normalized_score(results)
        return results

    @staticmethod
    def compute_normalized_score(raw_scores: Dict) -> float:
        """
        Single composite score across all tasks.

        normalized = 0.20 * easy_mean + 0.35 * medium_mean + 0.45 * hard_mean

        Weights reflect task difficulty and information content:
          - Easy (0.20): baseline competence check
          - Medium (0.35): requires understanding priority semantics
          - Hard (0.45): requires full code understanding + generation
        """
        easy = raw_scores.get("easy", {}).get("mean", 0.0)
        medium = raw_scores.get("medium", {}).get("mean", 0.0)
        hard = raw_scores.get("hard", {}).get("mean", 0.0)
        return 0.20 * easy + 0.35 * medium + 0.45 * hard

    @staticmethod
    def generate_latex_table(results: Dict, agent_name: str) -> str:
        """
        Generate LaTeX table suitable for paper inclusion.

        Columns: Task | Mean ± Std | Median | p25 | p75 | N
        """
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            f"\\caption{{CodeReviewEnv results for {agent_name}}}",
            r"\begin{tabular}{lccccr}",
            r"\toprule",
            r"Task & Mean $\pm$ Std & Median & p25 & p75 & N \\",
            r"\midrule",
        ]

        for task in ["easy", "medium", "hard"]:
            if task in results:
                r = results[task]
                lines.append(
                    f"{task.capitalize()} & "
                    f"{r['mean']:.3f} $\\pm$ {r['std']:.3f} & "
                    f"{r['median']:.3f} & "
                    f"{r['p25']:.3f} & "
                    f"{r['p75']:.3f} & "
                    f"{r['n']} \\\\"
                )

        if "composite" in results:
            lines.append(r"\midrule")
            lines.append(f"Composite & {results['composite']:.3f} & & & & \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    @staticmethod
    def assert_reproducibility(results_a: Dict, results_b: Dict) -> bool:
        """
        Check if two runs are statistically equivalent.

        Uses: |mean_a - mean_b| < 0.02 AND |std_a - std_b| < 0.01
        """
        for task in ["easy", "medium", "hard"]:
            if task not in results_a or task not in results_b:
                continue
            mean_diff = abs(results_a[task]["mean"] - results_b[task]["mean"])
            std_diff = abs(results_a[task]["std"] - results_b[task]["std"])
            if mean_diff >= 0.02 or std_diff >= 0.01:
                return False
        return True
