"""
Reward Computation — Multi-signal shaped reward for code review quality.

Five normalized signals combined with fixed weights:
  bug_detection   (0.40) — did the agent find the injected bugs?
  fix_quality     (0.25) — how close is the suggestion to the gold fix?
  line_precision  (0.15) — F1 score of flagged lines vs gold lines
  comment_quality (0.10) — is the natural-language comment helpful?
  efficiency      (0.10) — penalty for excessive steps / hint usage

All sub-signals are normalized to [0, 1], then weighted.
The final reward is in [0, 1] with difficulty multiplier for hard tasks.

No LLM calls — all scoring is deterministic via string matching,
difflib similarity, and F1 computation.
"""

import difflib
import re
from typing import Dict, List, Tuple

from snippet_bank import BugRecord


# ─── Signal Weights ──────────────────────────────────────────────────────────

WEIGHTS = {
    "bug_detection": 0.40,
    "fix_quality": 0.25,
    "line_precision": 0.15,
    "comment_quality": 0.10,
    "efficiency": 0.10,
}

# Keywords that indicate the agent understands the bug
BUG_TYPE_KEYWORDS = {
    "off_by_one": ["off-by-one", "off by one", "boundary", "fence", "<=", ">=", "<", ">", "range", "index"],
    "null_deref": ["null", "none", "nil", "undefined", "guard", "check", "dereference", "missing check"],
    "wrong_operator": ["operator", "wrong", "+", "-", "*", "/", "swap", "arithmetic"],
    "unused_var": ["unused", "dead", "shadow", "shadowed", "unreachable", "redundant"],
    "logic_inversion": ["invert", "flip", "wrong", "opposite", "and", "or", "boolean", "==", "!="],
    # AST-based injector types (same keyword families but more precise)
    "ast_comparison_flip": ["comparison", "operator", "<", "<=", ">", ">=", "==", "!=", "boundary", "condition"],
    "ast_binop_swap": ["operator", "arithmetic", "+", "-", "*", "//", "swap", "wrong"],
    "ast_boolop_flip": ["boolean", "and", "or", "logic", "condition", "flip"],
    "ast_return_negate": ["return", "wrong value", "negated", "inverted", "True", "False", "0", "1", "-1"],
}

# Keywords that indicate actionable comments
ACTIONABLE_KEYWORDS = [
    "use", "replace", "add", "remove", "consider", "should", "instead",
    "refactor", "fix", "change", "wrap", "guard", "check", "validate",
    "ensure", "avoid", "handle", "return", "throw",
]

# Difficulty multipliers for reward scaling
DIFFICULTY_MULTIPLIER = {
    "easy": 1.0,
    "medium": 1.0,
    "hard": 1.5,
}


# ─── Individual Signal Functions ─────────────────────────────────────────────

def _bug_overlap(issues: List[str], gold_bugs: List[BugRecord]) -> float:
    """Score bug detection: what fraction of gold bugs did the agent identify?

    For each gold bug, check if any reported issue mentions the bug type
    keywords or describes the same bug. Uses fuzzy keyword matching.

    Returns: float in [0, 1]
    """
    if not gold_bugs:
        return 1.0 if not issues else 0.5  # No bugs to find

    if not issues:
        return 0.0

    matched = 0
    issues_lower = [iss.lower() for iss in issues]

    for bug in gold_bugs:
        keywords = BUG_TYPE_KEYWORDS.get(bug.bug_type, [])
        bug_desc_lower = bug.description.lower()

        # Check if any issue mentions relevant keywords
        found = False
        for iss in issues_lower:
            # Keyword match
            kw_hits = sum(1 for kw in keywords if kw in iss)
            if kw_hits >= 1:
                found = True
                break
            # Fuzzy description match
            similarity = difflib.SequenceMatcher(None, iss, bug_desc_lower).ratio()
            if similarity > 0.3:
                found = True
                break
            # Line number mention
            for line in bug.lines:
                if str(line) in iss:
                    found = True
                    break
            if found:
                break

        if found:
            matched += 1

    return matched / len(gold_bugs)


def _fix_similarity(suggestion: str, gold_bugs: List[BugRecord]) -> float:
    """Score fix quality: how close is the suggestion to the gold fixes?

    Uses difflib SequenceMatcher for text similarity, plus keyword overlap.

    Returns: float in [0, 1]
    """
    if not gold_bugs:
        return 1.0 if not suggestion else 0.5

    if not suggestion:
        return 0.0

    suggestion_lower = suggestion.lower()
    similarities = []

    for bug in gold_bugs:
        fix_lower = bug.fix.lower()

        # Text similarity
        text_sim = difflib.SequenceMatcher(None, suggestion_lower, fix_lower).ratio()

        # Keyword overlap
        fix_words = set(re.findall(r'\w+', fix_lower))
        suggestion_words = set(re.findall(r'\w+', suggestion_lower))
        if fix_words:
            keyword_overlap = len(fix_words & suggestion_words) / len(fix_words)
        else:
            keyword_overlap = 0.0

        # Combined score
        similarities.append(0.6 * text_sim + 0.4 * keyword_overlap)

    return max(similarities) if similarities else 0.0


def _line_f1(flagged_lines: List[int], gold_bugs: List[BugRecord], tolerance: int = 3) -> float:
    """Score line-level precision: F1 score of flagged lines vs gold lines.

    Tolerance: a flagged line within ±3 of a gold line counts as a hit.

    Returns: float in [0, 1]
    """
    gold_lines = set()
    for bug in gold_bugs:
        gold_lines.update(bug.lines)

    if not gold_lines and not flagged_lines:
        return 1.0
    if not gold_lines:
        return 0.0 if flagged_lines else 1.0
    if not flagged_lines:
        return 0.0

    # True positives: flagged lines near gold lines
    tp = 0
    matched_gold = set()
    for fl in flagged_lines:
        for gl in gold_lines:
            if abs(fl - gl) <= tolerance and gl not in matched_gold:
                tp += 1
                matched_gold.add(gl)
                break

    precision = tp / len(flagged_lines) if flagged_lines else 0.0
    recall = tp / len(gold_lines) if gold_lines else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def _comment_score(comment: str) -> float:
    """Score comment quality: length, specificity, and actionability.

    Heuristic-based scoring:
      - Length: comments < 10 chars get 0.0, 10-50 chars get partial, 50+ full
      - Actionability: presence of suggestion keywords
      - Specificity: mentions line numbers, variable names, or code constructs

    Returns: float in [0, 1]
    """
    if not comment:
        return 0.0

    score = 0.0
    comment_lower = comment.lower()

    # Length component (0-0.3)
    if len(comment) >= 50:
        score += 0.3
    elif len(comment) >= 20:
        score += 0.2
    elif len(comment) >= 10:
        score += 0.1

    # Actionability (0-0.4)
    action_hits = sum(1 for kw in ACTIONABLE_KEYWORDS if kw in comment_lower)
    score += min(0.4, action_hits * 0.1)

    # Specificity (0-0.3)
    specificity = 0.0
    # Mentions line numbers
    if re.search(r'line\s*\d+', comment_lower):
        specificity += 0.15
    # Mentions code constructs
    if re.search(r'`[^`]+`|"[^"]+"|\b(function|variable|method|class|loop|condition)\b', comment_lower):
        specificity += 0.15
    score += min(0.3, specificity)

    return min(1.0, score)


def _efficiency_score(step_count: int, hint_count: int = 0) -> float:
    """Score efficiency: penalize excessive steps and hint usage.

    Each hint costs 0.05 from efficiency.
    More steps = lower efficiency.

    Returns: float in [0, 1]
    """
    step_penalty = max(0.0, 1.0 - step_count / 10.0)
    hint_penalty = max(0.0, 1.0 - hint_count * 0.1)
    return min(step_penalty, hint_penalty)


# ─── Main Reward Function ───────────────────────────────────────────────────

def compute_reward(
    issues: List[str],
    flagged_lines: List[int],
    suggestion: str,
    comment: str,
    gold_bugs: List[BugRecord],
    step_count: int = 1,
    hint_count: int = 0,
    difficulty: str = "easy",
) -> Tuple[float, Dict[str, float]]:
    """Compute the shaped, multi-signal reward for a code review.

    Args:
        issues: Bug descriptions submitted by the agent
        flagged_lines: Line numbers the agent flagged
        suggestion: Suggested fix text
        comment: Natural-language review comment
        gold_bugs: Ground truth bugs from injection
        step_count: Number of steps taken this episode
        hint_count: Number of hints requested (costs reward)
        difficulty: Difficulty tier for reward scaling

    Returns:
        (total_reward, breakdown_dict) where total_reward ∈ [0, 1]
        and breakdown_dict has per-signal scores.
    """
    signals = {
        "bug_detection": _bug_overlap(issues, gold_bugs),
        "fix_quality": _fix_similarity(suggestion, gold_bugs),
        "line_precision": _line_f1(flagged_lines, gold_bugs),
        "comment_quality": _comment_score(comment),
        "efficiency": _efficiency_score(step_count, hint_count),
    }

    # Weighted sum
    raw_reward = sum(WEIGHTS[k] * signals[k] for k in WEIGHTS)

    # Difficulty multiplier (hard tasks can earn up to 1.5x on bug_detection)
    multiplier = DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)
    if multiplier > 1.0:
        # Apply multiplier only to bug_detection signal
        bonus = (multiplier - 1.0) * WEIGHTS["bug_detection"] * signals["bug_detection"]
        raw_reward += bonus
        signals["difficulty_bonus"] = bonus

    # Clamp to [0, 1]
    total = max(0.0, min(1.0, raw_reward))

    # Build breakdown
    breakdown = {k: round(v, 4) for k, v in signals.items()}
    breakdown["weighted_total"] = round(total, 4)

    return total, breakdown
