"""
Medium Grader — Queue Prioritization Scorer

Scores agent's ability to order a PR review queue by priority.
Uses Kendall Tau rank correlation — a standard non-parametric
measure of ordinal association between two rankings.

Scoring formula:
  base_score = (kendall_tau + 1) / 2   # normalize from [-1,1] to [0,1]

Exploit prevention penalties:
  -0.3 if any critical PR not in top 2 positions
  -0.2 if security_vulnerability PR not in position 1

  final_score = max(0.0, base_score + penalties)

Priority ordering ground truth:
  1. Security PRs first (sql_injection, security_vulnerability)
  2. By severity: critical > high > medium > low > none
  3. Within same severity: junior authors first

Research metrics returned in info dict:
  - kendall_tau: raw Kendall Tau correlation [-1, 1]
  - spearman_rho: alternative rank correlation for comparison
  - top_k_precision: precision@k for k=1,2,3
  - critical_displacement: mean positional error of critical PRs
"""

from typing import Dict, List, Tuple
from env.models import Action, Reward
from env.data_generator import get_ground_truth


class MediumGrader:
    """
    Deterministic grader for queue prioritization (medium task).

    Uses Kendall Tau for ranking quality — the standard metric
    for comparing permutations in information retrieval and
    recommendation systems research.
    """

    def __init__(self):
        self.scores: List[float] = []

    def reset(self) -> None:
        """Reset episode tracking."""
        self.scores = []

    def grade(
        self, action: Action, queue_templates: List[Dict], ground_truth_order: List[str]
    ) -> Tuple[Reward, Dict]:
        """
        Grade a prioritization action against ground truth ordering.

        Args:
            action: Agent's action (must have action_type="prioritize")
            queue_templates: List of PR template dicts in the queue
            ground_truth_order: Correct PR ordering (list of pr_ids)

        Returns:
            (Reward, info_dict) with research metrics
        """
        breakdown: Dict[str, float] = {}

        # ── Validate action ──────────────────────────────────────────
        if action.action_type != "prioritize" or not action.priority_order:
            breakdown["step_reward"] = 0.0
            breakdown["critical_position_penalty"] = 0.0
            breakdown["security_position_penalty"] = 0.0
            return Reward(
                value=0.0,
                breakdown=breakdown,
                reason="Invalid action for prioritization. Expected prioritize with priority_order.",
            ), self._build_info([], ground_truth_order, queue_templates)

        predicted_order = action.priority_order

        # ── Compute Kendall Tau ──────────────────────────────────────
        # Kendall Tau measures the fraction of concordant vs discordant
        # pairs. tau = (concordant - discordant) / (n*(n-1)/2)
        # Range: [-1, 1] where 1 = identical ordering
        tau = self._kendall_tau(predicted_order, ground_truth_order)
        base_score = (tau + 1.0) / 2.0  # normalize to [0, 1]
        breakdown["step_reward"] = base_score

        # ── Exploit prevention: critical PR position ─────────────────
        # Critical bugs must be reviewed first — penalize if any critical
        # PR is placed lower than its ground truth position
        # Only triggers when agent genuinely deprioritizes critical PRs
        critical_ids = self._get_critical_ids(queue_templates)
        breakdown["critical_position_penalty"] = 0.0
        if critical_ids:
            # Check: are critical PRs in the top positions matching GT?
            n_critical = len(critical_ids)
            top_n = min(n_critical, 2)  # At most 2 slots to check
            gt_top = set(ground_truth_order[:top_n])
            pred_top = set(predicted_order[:top_n]) if len(predicted_order) >= top_n else set(predicted_order)
            # Only penalize if agent puts non-critical in top slots when GT has critical
            critical_in_gt_top = gt_top & set(critical_ids)
            critical_in_pred_top = pred_top & set(critical_ids)
            if len(critical_in_pred_top) < len(critical_in_gt_top):
                breakdown["critical_position_penalty"] = -0.3

        # ── Exploit prevention: security PR must be first ────────────
        # Security vulnerabilities should be at position 0 if GT says so
        security_ids = self._get_security_ids(queue_templates)
        breakdown["security_position_penalty"] = 0.0
        if security_ids and ground_truth_order and predicted_order:
            # Only penalize if GT has a security PR at position 0 but agent doesn't
            gt_first = ground_truth_order[0]
            pred_first = predicted_order[0]
            if gt_first in security_ids and pred_first not in security_ids:
                breakdown["security_position_penalty"] = -0.2

        total = max(0.0, min(1.0, sum(breakdown.values())))
        reason = f"Kendall Tau: {tau:.3f}, normalized: {base_score:.3f}"
        if breakdown["critical_position_penalty"] < 0:
            reason += f", critical position penalty: {breakdown['critical_position_penalty']}"
        if breakdown["security_position_penalty"] < 0:
            reason += f", security position penalty: {breakdown['security_position_penalty']}"

        reward = Reward(value=total, breakdown=breakdown, reason=reason)
        self.scores.append(total)

        return reward, self._build_info(predicted_order, ground_truth_order, queue_templates)

    def _kendall_tau(self, predicted: List[str], truth: List[str]) -> float:
        """
        Compute Kendall Tau rank correlation between two orderings.

        Implementation note: We compute this without scipy to avoid
        dependency issues in minimal environments. The formula is:
            tau = (concordant - discordant) / (n * (n - 1) / 2)

        Only considers items present in both lists.
        """
        # Build rank mapping from truth
        common = [x for x in predicted if x in truth]
        if len(common) < 2:
            return 0.0

        # Create rank dict based on predicted order
        pred_rank = {item: i for i, item in enumerate(predicted) if item in truth}
        truth_rank = {item: i for i, item in enumerate(truth) if item in predicted}

        concordant = 0
        discordant = 0
        n = len(common)

        for i in range(n):
            for j in range(i + 1, n):
                item_i = common[i]
                item_j = common[j]
                # Compare relative ordering in predicted vs truth
                pred_diff = pred_rank.get(item_i, 0) - pred_rank.get(item_j, 0)
                truth_diff = truth_rank.get(item_i, 0) - truth_rank.get(item_j, 0)

                if pred_diff * truth_diff > 0:
                    concordant += 1
                elif pred_diff * truth_diff < 0:
                    discordant += 1
                # ties (diff == 0) are neither concordant nor discordant

        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 0.0

        return (concordant - discordant) / total_pairs

    def _spearman_rho(self, predicted: List[str], truth: List[str]) -> float:
        """
        Spearman's rank correlation coefficient.

        rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
        where d_i is the difference in ranks for item i.

        Provides an alternative rank correlation — Spearman uses
        rank differences while Kendall uses concordant pairs.
        """
        common = [x for x in truth if x in predicted]
        n = len(common)
        if n < 2:
            return 0.0

        pred_rank = {item: i for i, item in enumerate(predicted)}
        truth_rank = {item: i for i, item in enumerate(truth)}

        d_squared_sum = sum(
            (pred_rank.get(item, 0) - truth_rank.get(item, 0)) ** 2
            for item in common
        )

        return 1.0 - (6.0 * d_squared_sum) / (n * (n ** 2 - 1))

    def _get_critical_ids(self, templates: List[Dict]) -> List[str]:
        """Get PR IDs with critical severity from queue."""
        return [t["pr_id"] for t in templates if t.get("ground_truth_severity") == "critical"]

    def _get_security_ids(self, templates: List[Dict]) -> List[str]:
        """Get PR IDs with security-related bugs."""
        security_categories = {"sql_injection", "security_vulnerability"}
        return [t["pr_id"] for t in templates if t.get("bug_category") in security_categories]

    def _top_k_precision(self, predicted: List[str], truth: List[str], k: int) -> float:
        """
        Precision@k: fraction of top-k predicted items in top-k truth.

        Standard information retrieval metric applied to prioritization.
        """
        if k <= 0 or not predicted or not truth:
            return 0.0
        top_k_pred = set(predicted[:k])
        top_k_truth = set(truth[:k])
        return len(top_k_pred & top_k_truth) / k

    def _critical_displacement(self, predicted: List[str], truth: List[str], templates: List[Dict]) -> float:
        """
        Mean positional error of critical PRs.

        displacement_i = |predicted_position - truth_position|
        Returns mean displacement for critical PRs.
        Lower is better.
        """
        critical_ids = self._get_critical_ids(templates)
        if not critical_ids:
            return 0.0

        truth_rank = {item: i for i, item in enumerate(truth)}
        pred_rank = {item: i for i, item in enumerate(predicted)}

        displacements = []
        for cid in critical_ids:
            if cid in pred_rank and cid in truth_rank:
                displacements.append(abs(pred_rank[cid] - truth_rank[cid]))
            else:
                displacements.append(len(truth))  # max displacement if missing

        return sum(displacements) / len(displacements) if displacements else 0.0

    def _build_info(self, predicted: List[str], truth: List[str], templates: List[Dict]) -> Dict:
        """Build research-grade info dict."""
        info = {
            "kendall_tau": self._kendall_tau(predicted, truth),
            "spearman_rho": self._spearman_rho(predicted, truth),
            "top_k_precision": {
                "p@1": self._top_k_precision(predicted, truth, 1),
                "p@2": self._top_k_precision(predicted, truth, 2),
                "p@3": self._top_k_precision(predicted, truth, 3),
            },
            "critical_displacement": self._critical_displacement(predicted, truth, templates),
            "predicted_order": predicted,
            "ground_truth_order": truth,
        }
        return info

    def episode_score(self, step_rewards: List[float]) -> float:
        """Compute episode-level score as mean of step rewards."""
        if not step_rewards:
            return 0.0
        return sum(step_rewards) / len(step_rewards)
