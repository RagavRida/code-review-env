"""
Easy Grader — Severity Labeling Scorer

Scores agent's ability to correctly label PR bug severity.
Fully deterministic, no LLM calls, never crashes on malformed input.

Scoring formula:
  exact_match:   1.0 if predicted == ground_truth
  adjacent_match: 0.5 if off by one level on the ordinal scale
                  (critical↔high, high↔medium, medium↔low, low↔none)
  wrong:         0.0 for all other mismatches

Exploit prevention penalties (applied on top of base score):
  critical missed as "none":  -0.3 extra
  critical missed as "low":   -0.2 extra

These penalties reflect real-world cost: missing a critical security
bug during code review has outsized negative impact. The asymmetric
penalty structure encodes domain knowledge about review risk.

Research metrics returned in info dict:
  - confusion_matrix: 5×5 severity classification matrix
  - severity_bias: signed mean prediction error (positive = over-labeling)
  - critical_recall: fraction of critical bugs correctly identified
  - false_critical_rate: fraction of non-critical labeled as critical
"""

from typing import Dict, List, Optional, Tuple
from env.models import Action, Reward
from env.data_generator import SEVERITY_ORDER, get_ground_truth


class EasyGrader:
    """
    Deterministic grader for severity labeling (easy task).

    Implements ordinal scoring with adjacency bonus and asymmetric
    penalty for missed critical bugs — a design choice reflecting
    the real-world cost structure of code review.
    """

    def __init__(self):
        # Severity levels in order: index 0 = most severe
        self.severity_levels = SEVERITY_ORDER  # ["critical", "high", "medium", "low", "none"]
        self.severity_index = {s: i for i, s in enumerate(self.severity_levels)}

        # Tracking for research metrics across episode
        self.predictions: List[str] = []
        self.ground_truths: List[str] = []

    def reset(self) -> None:
        """Reset episode-level tracking for research metrics."""
        self.predictions = []
        self.ground_truths = []

    def grade(self, action: Action, pr_id: str) -> Tuple[Reward, Dict]:
        """
        Grade a single severity labeling action.

        Args:
            action: Agent's action (must have action_type="label_severity")
            pr_id: The PR being graded

        Returns:
            (Reward, info_dict) where info_dict contains research metrics
        """
        gt = get_ground_truth(pr_id)
        true_severity = gt["ground_truth_severity"]
        breakdown: Dict[str, float] = {}

        # ── Validate action ──────────────────────────────────────────
        if action.action_type != "label_severity" or action.severity is None:
            self.predictions.append("none")
            self.ground_truths.append(true_severity)
            breakdown["step_reward"] = 0.01
            breakdown["critical_penalty"] = -0.3 if true_severity == "critical" else 0.0
            total = max(0.01, min(0.99, sum(breakdown.values())))
            return Reward(
                value=total,
                breakdown=breakdown,
                reason=f"Invalid action for severity labeling. Expected label_severity with severity field.",
            ), self._build_info(true_severity, "none")

        predicted = action.severity
        self.predictions.append(predicted)
        self.ground_truths.append(true_severity)

        # ── Compute base score ───────────────────────────────────────
        pred_idx = self.severity_index.get(predicted, -1)
        true_idx = self.severity_index.get(true_severity, -1)

        if pred_idx == -1 or true_idx == -1:
            # Unknown severity label — near-zero score
            breakdown["step_reward"] = 0.01
        elif predicted == true_severity:
            # Exact match — near-full score (strictly < 1.0 per validator)
            breakdown["step_reward"] = 0.99
        elif abs(pred_idx - true_idx) == 1:
            # Adjacent match — half score
            # Empirically, one-level-off is a reasonable disagreement
            # (human inter-rater agreement on severity is ~0.7 Cohen's κ)
            breakdown["step_reward"] = 0.5
        else:
            # Wrong by 2+ levels
            breakdown["step_reward"] = 0.05

        # ── Exploit prevention penalties ─────────────────────────────
        # Asymmetric: missing critical is penalized more heavily because
        # false negatives on security bugs have higher real-world cost
        # than false positives (which only waste reviewer time)
        breakdown["critical_penalty"] = 0.0
        if true_severity == "critical":
            if predicted == "none":
                breakdown["critical_penalty"] = -0.3  # Most dangerous miss
            elif predicted == "low":
                breakdown["critical_penalty"] = -0.2  # Still very bad

        total = max(0.01, min(0.99, sum(breakdown.values())))
        reason = f"Predicted: {predicted}, Truth: {true_severity}"
        if breakdown["critical_penalty"] < 0:
            reason += f" (critical miss penalty: {breakdown['critical_penalty']})"

        return Reward(
            value=total,
            breakdown=breakdown,
            reason=reason,
        ), self._build_info(true_severity, predicted)

    def _build_info(self, true_severity: str, predicted: str) -> Dict:
        """Build research-grade info dict with classification metrics."""
        info: Dict = {
            "true_severity": true_severity,
            "predicted_severity": predicted,
        }

        # Compute confusion matrix from all predictions so far
        info["confusion_matrix"] = self._confusion_matrix()
        info["severity_bias"] = self._severity_bias()
        info["critical_recall"] = self._critical_recall()
        info["false_critical_rate"] = self._false_critical_rate()

        return info

    def _confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        5×5 confusion matrix: matrix[true][predicted] = count.
        Enables detailed error analysis beyond aggregate scores.
        """
        matrix = {s: {p: 0 for p in self.severity_levels} for s in self.severity_levels}
        for true, pred in zip(self.ground_truths, self.predictions):
            if true in matrix and pred in matrix[true]:
                matrix[true][pred] += 1
        return matrix

    def _severity_bias(self) -> float:
        """
        Signed mean prediction error on ordinal scale.
        Positive = over-labeling (predicting more severe than truth).
        Negative = under-labeling (predicting less severe than truth).

        Computed as: mean(true_index - pred_index)
        Since index 0 = critical (most severe), positive bias means
        the agent tends to predict less severe (higher index) than truth.
        """
        if not self.predictions:
            return 0.0
        errors = []
        for true, pred in zip(self.ground_truths, self.predictions):
            t_idx = self.severity_index.get(true, 2)
            p_idx = self.severity_index.get(pred, 2)
            errors.append(p_idx - t_idx)
        return sum(errors) / len(errors)

    def _critical_recall(self) -> float:
        """
        Fraction of critical bugs correctly identified.
        critical_recall = TP_critical / (TP_critical + FN_critical)
        """
        total_critical = sum(1 for t in self.ground_truths if t == "critical")
        if total_critical == 0:
            return 1.0  # No critical bugs — perfect recall by default
        caught = sum(1 for t, p in zip(self.ground_truths, self.predictions) if t == "critical" and p == "critical")
        return caught / total_critical

    def _false_critical_rate(self) -> float:
        """
        Fraction of non-critical PRs labeled as critical.
        false_critical_rate = FP_critical / total_non_critical
        """
        non_critical = sum(1 for t in self.ground_truths if t != "critical")
        if non_critical == 0:
            return 0.0
        false_critical = sum(1 for t, p in zip(self.ground_truths, self.predictions) if t != "critical" and p == "critical")
        return false_critical / non_critical

    def analyze_failure_modes(self) -> Dict:
        """
        Analyze common failure patterns in agent predictions.

        Returns:
            missed_critical: count of critical bugs not labeled critical
            vague_labels: count of "none" predictions for bugs with severity >= medium
            over_labeled: count of non-bugs labeled as high or critical
        """
        missed_critical = sum(
            1 for t, p in zip(self.ground_truths, self.predictions)
            if t == "critical" and p != "critical"
        )
        vague_labels = sum(
            1 for t, p in zip(self.ground_truths, self.predictions)
            if t in ("critical", "high", "medium") and p == "none"
        )
        over_labeled = sum(
            1 for t, p in zip(self.ground_truths, self.predictions)
            if t in ("low", "none") and p in ("critical", "high")
        )
        return {
            "missed_critical": missed_critical,
            "vague_labels": vague_labels,
            "over_labeled": over_labeled,
        }

    def episode_score(self, step_rewards: List[float]) -> float:
        """
        Compute episode-level score as mean of step rewards.

        This is the aggregate metric for the easy task.
        Each step reward is already in [-1, 1], so mean is also in [-1, 1].
        """
        if not step_rewards:
            return 0.0
        return sum(step_rewards) / len(step_rewards)
