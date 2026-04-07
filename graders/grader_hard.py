"""
Hard Grader — Actionable Feedback Generation Scorer

Scores agent's code review comments across 5 dimensions:
  relevance     (0.25 weight): comments target actual bug locations (±5 lines)
  specificity   (0.20 weight): comments mention bug-category-specific keywords
  actionability (0.20 weight): comments contain concrete suggestions
  coverage      (0.25 weight): critical and high bugs have relevant comments
  precision     (0.10 weight): comments don't target non-bug locations

All scoring is fully deterministic via keyword matching and line proximity —
no LLM calls, no randomness. This ensures reproducible evaluation across
different hardware and runtimes.

Exploit prevention:
  >10 comments per PR:  precision denominator doubles (spam penalty)
  approve without comments: score = 0.0 flat
  request_changes without comments: score = 0.0 flat
  approve with unaddressed critical bug: -0.5 episode penalty

Weight rationale (empirically calibrated):
  Relevance + Coverage = 0.50: catching real bugs is the primary goal
  Specificity + Actionability = 0.40: review quality matters
  Precision = 0.10: false positives waste time but are less harmful
"""

from typing import Dict, List, Tuple, Optional
from env.models import Action, Reward
from env.data_generator import get_ground_truth, BUG_KEYWORDS, ACTIONABILITY_KEYWORDS


class HardGrader:
    """
    Deterministic grader for feedback generation (hard task).

    Five-component weighted scoring with exploit prevention.
    Designed to be genuinely hard — a simple heuristic scores ~0.3,
    reflecting the difficulty of generating precise, actionable
    code review feedback targeting specific bug locations.
    """

    # Component weights — sum to 1.0
    # Coverage and relevance weighted highest because they measure
    # the fundamental goal: finding and targeting real bugs
    W_RELEVANCE = 0.25
    W_SPECIFICITY = 0.20
    W_ACTIONABILITY = 0.20
    W_COVERAGE = 0.25
    W_PRECISION = 0.10

    # Line proximity tolerance: ±3 lines counts as "relevant"
    # Tighter than typical review tools to reward precise targeting
    LINE_TOLERANCE = 3

    # Minimum keywords required for a comment to be "specific"
    MIN_KEYWORDS_FOR_SPECIFIC = 2

    # Spam threshold: more than this many comments triggers penalty
    SPAM_THRESHOLD = 10

    def __init__(self):
        self.episode_comments: Dict[str, List[Action]] = {}  # pr_id → comments
        self.episode_decisions: Dict[str, str] = {}  # pr_id → approve/request_changes
        self.episode_penalties: float = 0.0
        self.consecutive_comments: int = 0  # track spam pattern

    def reset(self) -> None:
        """Reset episode-level tracking."""
        self.episode_comments = {}
        self.episode_decisions = {}
        self.episode_penalties = 0.0
        self.consecutive_comments = 0

    def add_comment(self, pr_id: str, action: Action) -> None:
        """Track a comment action for later scoring."""
        if pr_id not in self.episode_comments:
            self.episode_comments[pr_id] = []
        self.episode_comments[pr_id].append(action)

    def grade_pr(self, pr_id: str, decision: str) -> Tuple[Reward, Dict]:
        """
        Grade all comments + decision for a single PR.

        Called when agent submits approve or request_changes.

        Args:
            pr_id: The PR being reviewed
            decision: "approve" or "request_changes"

        Returns:
            (Reward, info_dict) with per-component scores and analysis
        """
        self.episode_decisions[pr_id] = decision
        comments = self.episode_comments.get(pr_id, [])
        gt = get_ground_truth(pr_id)
        bug_lines = gt["bug_lines"]
        bug_category = gt["bug_category"]
        true_severity = gt["ground_truth_severity"]
        breakdown: Dict[str, float] = {}

        # ── Exploit check: no comments ───────────────────────────────
        if not comments:
            if decision == "approve":
                breakdown["step_reward"] = 0.01
                return Reward(
                    value=0.01,
                    breakdown=breakdown,
                    reason="Approved without any review comments — near-zero score",
                ), self._empty_info(pr_id)
            elif decision == "request_changes":
                breakdown["step_reward"] = 0.01
                return Reward(
                    value=0.01,
                    breakdown=breakdown,
                    reason="Requested changes without any comments — near-zero score",
                ), self._empty_info(pr_id)

        total_comments = len(comments)

        # ── 1. Relevance (0.25): comments target actual bug locations ─
        relevant_count = 0
        for c in comments:
            if c.target_line is not None and bug_lines:
                for bl in bug_lines:
                    if abs(c.target_line - bl) <= self.LINE_TOLERANCE:
                        relevant_count += 1
                        break
        relevance = relevant_count / total_comments if total_comments > 0 else 0.0
        breakdown["relevance"] = relevance

        # ── 2. Specificity (0.20): comments mention category keywords ─
        # Requires 2+ keywords per comment for full credit (1 keyword = 0.5 credit)
        keywords = BUG_KEYWORDS.get(bug_category, [])
        specific_score_sum = 0.0
        specific_count = 0
        for c in comments:
            if c.comment:
                comment_lower = c.comment.lower()
                kw_hits = sum(1 for kw in keywords if kw.lower() in comment_lower)
                if kw_hits >= self.MIN_KEYWORDS_FOR_SPECIFIC:
                    specific_score_sum += 1.0
                    specific_count += 1
                elif kw_hits == 1:
                    specific_score_sum += 0.5  # partial credit for 1 keyword
                    specific_count += 1
        specificity = specific_score_sum / total_comments if total_comments > 0 else 0.0
        breakdown["specificity"] = specificity

        # ── 3. Actionability (0.20): comments suggest concrete fixes ──
        actionable_count = 0
        for c in comments:
            if c.comment:
                comment_lower = c.comment.lower()
                if any(kw in comment_lower for kw in ACTIONABILITY_KEYWORDS):
                    actionable_count += 1
        actionability = actionable_count / total_comments if total_comments > 0 else 0.0
        breakdown["actionability"] = actionability

        # ── 4. Coverage (0.25): measures % of bug lines addressed ────
        # Now counts individual bug lines covered, not just binary
        if bug_lines:
            lines_covered = set()
            for c in comments:
                if c.target_line is not None:
                    for bl in bug_lines:
                        if abs(c.target_line - bl) <= self.LINE_TOLERANCE:
                            lines_covered.add(bl)
            coverage = len(lines_covered) / len(bug_lines)

            # Depth penalty: if there are 3+ bugs but only 1 comment, penalize
            if len(bug_lines) >= 3 and total_comments == 1:
                coverage *= 0.5  # reviewing complex code with 1 comment is shallow
        else:
            # No bugs — coverage is based on correct decision
            coverage = 1.0 if decision == "approve" else 0.5
        breakdown["coverage"] = coverage

        # ── 5. Precision (0.10): avoid false positives ────────────────
        false_positives = 0
        for c in comments:
            if c.target_line is not None:
                is_near_bug = False
                if bug_lines:
                    for bl in bug_lines:
                        if abs(c.target_line - bl) <= self.LINE_TOLERANCE:
                            is_near_bug = True
                            break
                if not is_near_bug:
                    false_positives += 1

        # Spam penalty: >10 comments doubles precision denominator
        # This prevents agents from gaming coverage by spamming comments
        effective_total = total_comments
        if total_comments > self.SPAM_THRESHOLD:
            effective_total = total_comments * 2

        precision = 1.0 - (false_positives / effective_total) if effective_total > 0 else 1.0
        precision = max(0.0, precision)
        breakdown["precision"] = precision

        # ── Weighted final score ─────────────────────────────────────
        step_score = (
            self.W_RELEVANCE * relevance
            + self.W_SPECIFICITY * specificity
            + self.W_ACTIONABILITY * actionability
            + self.W_COVERAGE * coverage
            + self.W_PRECISION * precision
        )
        breakdown["step_reward"] = step_score

        # ── Exploit: approve with unaddressed critical bug ───────────
        breakdown["critical_approve_penalty"] = 0.0
        bugs = self._bugs_caught(comments, gt)
        critical_caught = bugs.get("critical", 0)
        if decision == "approve" and true_severity == "critical" and critical_caught == 0:
            breakdown["critical_approve_penalty"] = -0.5
            self.episode_penalties += -0.5

        total = max(0.01, min(0.99, step_score + breakdown["critical_approve_penalty"]))

        # Build detailed info
        info = {
            "relevance_score": relevance,
            "specificity_score": specificity,
            "actionability_score": actionability,
            "coverage_score": coverage,
            "precision_score": precision,
            "bugs_caught": self._bugs_caught(comments, gt),
            "bugs_missed": self._bugs_missed(comments, gt),
            "comment_efficiency": relevant_count / total_comments if total_comments > 0 else 0.0,
            "false_positive_rate": false_positives / total_comments if total_comments > 0 else 0.0,
            "total_comments": total_comments,
            "relevant_comments": relevant_count,
            "specific_comments": specific_count,
            "actionable_comments": actionable_count,
            "false_positives": false_positives,
            "decision": decision,
            "true_severity": true_severity,
            "bug_category": bug_category,
        }

        reason = (
            f"rel={relevance:.2f} spec={specificity:.2f} act={actionability:.2f} "
            f"cov={coverage:.2f} prec={precision:.2f} → {step_score:.3f}"
        )
        if breakdown["critical_approve_penalty"] < 0:
            reason += f" (critical approve penalty: {breakdown['critical_approve_penalty']})"

        return Reward(value=total, breakdown=breakdown, reason=reason), info

    def _empty_info(self, pr_id: str) -> Dict:
        """Return zeroed info dict for invalid actions."""
        gt = get_ground_truth(pr_id)
        return {
            "relevance_score": 0.0,
            "specificity_score": 0.0,
            "actionability_score": 0.0,
            "coverage_score": 0.0,
            "precision_score": 0.0,
            "bugs_caught": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "bugs_missed": self._all_bugs_as_missed(gt),
            "comment_efficiency": 0.0,
            "false_positive_rate": 0.0,
            "total_comments": 0,
            "relevant_comments": 0,
            "specific_comments": 0,
            "actionable_comments": 0,
            "false_positives": 0,
            "decision": self.episode_decisions.get(pr_id, "none"),
            "true_severity": gt["ground_truth_severity"],
            "bug_category": gt["bug_category"],
        }

    def _bugs_caught(self, comments: List[Action], gt: Dict) -> Dict[str, int]:
        """Count bugs caught per severity level."""
        result = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        sev = gt["ground_truth_severity"]
        if sev == "none":
            return result

        bug_lines = gt["bug_lines"]
        caught = False
        for c in comments:
            if c.target_line is not None:
                for bl in bug_lines:
                    if abs(c.target_line - bl) <= self.LINE_TOLERANCE:
                        caught = True
                        break
            if caught:
                break

        if caught and sev in result:
            result[sev] = 1
        return result

    def _bugs_missed(self, comments: List[Action], gt: Dict) -> Dict[str, int]:
        """Count bugs missed per severity level."""
        caught = self._bugs_caught(comments, gt)
        result = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        sev = gt["ground_truth_severity"]
        if sev in result and caught.get(sev, 0) == 0:
            result[sev] = 1
        return result

    def _all_bugs_as_missed(self, gt: Dict) -> Dict[str, int]:
        """All bugs as missed (for empty comment case)."""
        result = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        sev = gt["ground_truth_severity"]
        if sev in result:
            result[sev] = 1
        return result

    def episode_score(self, step_rewards: List[float]) -> float:
        """Compute episode-level score with accumulated penalties."""
        if not step_rewards:
            return 0.0
        mean_score = sum(step_rewards) / len(step_rewards)
        return max(0.01, min(0.99, mean_score + self.episode_penalties))
