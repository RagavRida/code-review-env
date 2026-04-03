"""
Agent Capability Profiler

Analyzes trajectory data to characterize agent behavior along
research-relevant dimensions. Enables comparative analysis
across different agent architectures.

Research motivation:
  Standard benchmarks report only aggregate scores. This profiler
  enables fine-grained behavioral analysis:
  - Exploration rate: is the agent stuck in a policy rut?
  - Reward trajectory shape: does the agent learn within episodes?
  - Action distribution: does the agent exploit trivial strategies?
  - Severity calibration: how well-calibrated are predictions?
"""

import json
import os
import hashlib
import statistics
from typing import Dict, List, Tuple, Optional

from env.data_generator import SEVERITY_ORDER


class AgentProfiler:
    """
    Analyzes trajectory data for research-grade agent characterization.

    Use this to compare agent architectures beyond simple score tables.
    Generate reports suitable for paper appendices.
    """

    def load_trajectories(self, directory: str) -> List[Dict]:
        """
        Load all JSONL trajectory files from directory.

        Each file is one episode, each line is one (s, a, r, s') transition.
        """
        trajectories = []
        if not os.path.exists(directory):
            return trajectories

        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(directory, filename)
                episode = []
                with open(filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            episode.append(json.loads(line))
                if episode:
                    trajectories.append(episode)
        return trajectories

    def compute_exploration_rate(self, trajectories: List[List[Dict]]) -> float:
        """
        Fraction of unique (state_hash, action_type) pairs visited.

        Low exploration = agent stuck in policy rut (always same action).
        High exploration = agent adapts to different states.

        State hash uses PR title + author_experience to avoid
        hash collisions on diff content.
        """
        if not trajectories:
            return 0.0

        unique_pairs = set()
        total_steps = 0

        for episode in trajectories:
            for transition in episode:
                state = transition.get("state", {})
                action = transition.get("action", {})

                # Hash state on key semantic features
                state_key = f"{state.get('pr_id', '')}_{state.get('title', '')}"
                state_hash = hashlib.md5(state_key.encode()).hexdigest()[:8]

                action_type = action.get("action_type", "unknown")
                pair = (state_hash, action_type)
                unique_pairs.add(pair)
                total_steps += 1

        if total_steps == 0:
            return 0.0

        return len(unique_pairs) / total_steps

    def compute_reward_trajectory_shape(self, trajectories: List[List[Dict]]) -> Dict:
        """
        Analyze how reward evolves across steps within episodes.

        Returns:
            slope: linear regression slope of reward over steps
            variance: reward variance within episodes
            monotonic_fraction: fraction of episodes with monotonically
                               increasing reward (in-context learning signal)
        """
        if not trajectories:
            return {"slope": 0.0, "variance": 0.0, "monotonic_fraction": 0.0}

        all_slopes = []
        all_variances = []
        monotonic_count = 0

        for episode in trajectories:
            rewards = [t.get("reward", {}).get("value", 0.0) for t in episode]
            if len(rewards) < 2:
                continue

            # Simple linear regression slope
            n = len(rewards)
            x_mean = (n - 1) / 2
            y_mean = sum(rewards) / n
            numerator = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(rewards))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0.0
            all_slopes.append(slope)

            # Variance
            if len(rewards) > 1:
                all_variances.append(statistics.variance(rewards))

            # Monotonicity check
            is_monotonic = all(rewards[i] <= rewards[i + 1] for i in range(len(rewards) - 1))
            if is_monotonic:
                monotonic_count += 1

        n_episodes = len(trajectories)
        return {
            "slope": statistics.mean(all_slopes) if all_slopes else 0.0,
            "variance": statistics.mean(all_variances) if all_variances else 0.0,
            "monotonic_fraction": monotonic_count / n_episodes if n_episodes > 0 else 0.0,
        }

    def compute_action_distribution(self, trajectories: List[List[Dict]]) -> Dict:
        """
        Distribution of action_types across all steps.

        Reveals if agent exploits (e.g., always approve) or uses
        the full action space. Uniform distribution over valid
        actions suggests genuine exploration.
        """
        counts: Dict[str, int] = {}
        total = 0

        for episode in trajectories:
            for transition in episode:
                action = transition.get("action", {})
                action_type = action.get("action_type", "unknown")
                counts[action_type] = counts.get(action_type, 0) + 1
                total += 1

        if total == 0:
            return {}

        return {k: {"count": v, "fraction": v / total} for k, v in sorted(counts.items())}

    def compute_severity_calibration(self, trajectories: List[List[Dict]]) -> Dict:
        """
        For easy task: calibration curve of predicted vs true severity.

        Similar to probability calibration in classification literature.
        Returns fraction of correct predictions per severity level.
        """
        correct_by_severity: Dict[str, int] = {s: 0 for s in SEVERITY_ORDER}
        total_by_severity: Dict[str, int] = {s: 0 for s in SEVERITY_ORDER}

        for episode in trajectories:
            for transition in episode:
                action = transition.get("action", {})
                if action.get("action_type") != "label_severity":
                    continue

                predicted = action.get("severity", "none")
                # Get true severity from reward reason
                reason = transition.get("reward", {}).get("reason", "")
                true_sev = None
                if "Truth:" in reason:
                    parts = reason.split("Truth:")
                    if len(parts) > 1:
                        true_sev = parts[1].strip().split()[0].strip(",")

                if true_sev and true_sev in total_by_severity:
                    total_by_severity[true_sev] += 1
                    if predicted == true_sev:
                        correct_by_severity[true_sev] += 1

        calibration = {}
        for sev in SEVERITY_ORDER:
            total = total_by_severity[sev]
            if total > 0:
                calibration[sev] = {
                    "accuracy": correct_by_severity[sev] / total,
                    "total": total,
                    "correct": correct_by_severity[sev],
                }
            else:
                calibration[sev] = {"accuracy": 0.0, "total": 0, "correct": 0}

        return calibration

    def compare_agents(self, agent_a_dir: str, agent_b_dir: str) -> Dict:
        """
        Statistical comparison between two agents.

        Uses Mann-Whitney U test for non-parametric comparison
        and Cohen's d for effect size measurement.
        """
        traj_a = self.load_trajectories(agent_a_dir)
        traj_b = self.load_trajectories(agent_b_dir)

        scores_a = [
            statistics.mean([t.get("reward", {}).get("value", 0.0) for t in ep])
            for ep in traj_a if ep
        ]
        scores_b = [
            statistics.mean([t.get("reward", {}).get("value", 0.0) for t in ep])
            for ep in traj_b if ep
        ]

        if not scores_a or not scores_b:
            return {"error": "Insufficient trajectory data for comparison"}

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.001
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.001

        # Cohen's d effect size
        pooled_std = ((std_a ** 2 + std_b ** 2) / 2) ** 0.5
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        # Mann-Whitney U (simplified — counts wins)
        u_stat = 0
        for sa in scores_a:
            for sb in scores_b:
                if sa > sb:
                    u_stat += 1
                elif sa == sb:
                    u_stat += 0.5

        n_a, n_b = len(scores_a), len(scores_b)
        max_u = n_a * n_b

        return {
            "agent_a": {"mean": mean_a, "std": std_a, "n": n_a},
            "agent_b": {"mean": mean_b, "std": std_b, "n": n_b},
            "cohens_d": cohens_d,
            "effect_interpretation": self._interpret_effect(abs(cohens_d)),
            "mann_whitney_u": u_stat,
            "u_normalized": u_stat / max_u if max_u > 0 else 0.0,
        }

    @staticmethod
    def _interpret_effect(d: float) -> str:
        """Interpret Cohen's d effect size (Cohen, 1988)."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def generate_report(self, trajectories: List[List[Dict]], agent_name: str) -> str:
        """
        Generate markdown report with all metrics.

        Format suitable for inclusion in paper appendix.
        """
        exploration = self.compute_exploration_rate(trajectories)
        shape = self.compute_reward_trajectory_shape(trajectories)
        actions = self.compute_action_distribution(trajectories)
        calibration = self.compute_severity_calibration(trajectories)

        # Compute aggregate scores
        episode_scores = []
        for ep in trajectories:
            if ep:
                rewards = [t.get("reward", {}).get("value", 0.0) for t in ep]
                episode_scores.append(statistics.mean(rewards))

        report = f"# Agent Profile: {agent_name}\n\n"
        report += f"## Summary\n"
        report += f"- Episodes: {len(trajectories)}\n"
        if episode_scores:
            report += f"- Mean score: {statistics.mean(episode_scores):.3f}\n"
            if len(episode_scores) > 1:
                report += f"- Score std: {statistics.stdev(episode_scores):.3f}\n"
        report += f"- Exploration rate: {exploration:.3f}\n"
        report += f"- Reward slope: {shape['slope']:.4f}\n"
        report += f"- Monotonic episodes: {shape['monotonic_fraction']:.1%}\n\n"

        report += f"## Action Distribution\n"
        report += "| Action | Count | Fraction |\n"
        report += "|--------|-------|----------|\n"
        for atype, info in actions.items():
            report += f"| {atype} | {info['count']} | {info['fraction']:.2%} |\n"
        report += "\n"

        report += f"## Severity Calibration\n"
        report += "| Severity | Accuracy | Total | Correct |\n"
        report += "|----------|----------|-------|---------|\n"
        for sev, info in calibration.items():
            report += f"| {sev} | {info['accuracy']:.2%} | {info['total']} | {info['correct']} |\n"

        return report
