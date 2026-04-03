"""
Benchmark Agents — Random and Perfect baselines.

These establish the floor and ceiling of the CodeReviewEnv benchmark.
Every new agent should be compared against these two.

RandomAgent:  picks actions uniformly at random (floor)
PerfectAgent: reads ground truth, always correct (ceiling)
"""

import random
from typing import Dict, List

from env.data_generator import PR_TEMPLATES, get_ground_truth, DataGenerator, SEVERITY_ORDER
from env.models import Action


class RandomAgent:
    """
    Picks action_type uniformly at random, random severity.

    Establishes the benchmark floor — any useful agent must
    significantly outperform random. Expected composite score ~0.18.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def act(self, observation: Dict, system_prompt: str) -> Dict:
        """
        Generate random action based on system prompt content.

        Detects task from system prompt to generate valid action types.
        """
        if "label_severity" in system_prompt:
            return self._act_easy(observation)
        elif "prioritize" in system_prompt:
            return self._act_medium(observation)
        else:
            return self._act_hard(observation)

    def _act_easy(self, observation: Dict) -> Dict:
        """Random severity label."""
        severity = self.rng.choice(SEVERITY_ORDER)
        return {"action_type": "label_severity", "severity": severity}

    def _act_medium(self, observation: Dict) -> Dict:
        """Random queue ordering."""
        queue = list(observation.get("review_queue", []))
        self.rng.shuffle(queue)
        return {"action_type": "prioritize", "priority_order": queue}

    def _act_hard(self, observation: Dict) -> Dict:
        """Random comments then random decision."""
        # 50% chance to comment, 50% to decide
        if self.rng.random() < 0.5:
            files = observation.get("files", [])
            target_file = files[0].get("filename", "unknown.py") if files else "unknown.py"
            return {
                "action_type": "add_comment",
                "comment": "Looks fine to me.",
                "target_file": target_file,
                "target_line": self.rng.randint(1, 50),
            }
        else:
            decision = self.rng.choice(["approve", "request_changes"])
            return {"action_type": decision}


class PerfectAgent:
    """
    Reads ground truth from FIXED_TEST_SUITE, always correct.

    Establishes the benchmark ceiling — represents optimal behavior
    given full knowledge of bug locations and severities.
    Expected composite score ~0.97.
    """

    def __init__(self, seed: int = 42):
        self.generator = DataGenerator(seed=seed)
        self._severity_cache: Dict[str, str] = {
            t["pr_id"]: t["ground_truth_severity"] for t in PR_TEMPLATES
        }
        self._template_cache: Dict[str, Dict] = {
            t["pr_id"]: t for t in PR_TEMPLATES
        }
        self._comment_count: Dict[str, int] = {}

    def act(self, observation: Dict, system_prompt: str) -> Dict:
        """
        Generate perfect action based on ground truth.

        Detects task from system prompt to generate correct responses.
        """
        if "label_severity" in system_prompt:
            return self._act_easy(observation)
        elif "prioritize" in system_prompt:
            return self._act_medium(observation)
        else:
            return self._act_hard(observation)

    def _act_easy(self, observation: Dict) -> Dict:
        """Return ground truth severity."""
        pr_id = observation.get("pr_id", "")
        severity = self._severity_cache.get(pr_id, "medium")
        return {"action_type": "label_severity", "severity": severity}

    def _act_medium(self, observation: Dict) -> Dict:
        """Return ground truth priority ordering."""
        queue_ids = observation.get("review_queue", [])
        # Build queue templates from cache
        queue_templates = []
        for pr_id in queue_ids:
            if pr_id in self._template_cache:
                queue_templates.append(self._template_cache[pr_id])

        if queue_templates:
            order = self.generator.compute_priority_order(queue_templates)
        else:
            order = queue_ids

        return {"action_type": "prioritize", "priority_order": order}

    def _act_hard(self, observation: Dict) -> Dict:
        """
        Generate targeted, specific, actionable comments then decide.

        Strategy: comment on each bug line with category-specific keywords,
        then request_changes if bugs exist, approve if clean.
        """
        pr_id = observation.get("pr_id", "")
        template = self._template_cache.get(pr_id, {})
        bug_lines = template.get("bug_lines", [])
        bug_category = template.get("bug_category", "")
        severity = template.get("ground_truth_severity", "none")

        # Track comments per PR
        if pr_id not in self._comment_count:
            self._comment_count[pr_id] = 0

        # Add one comment per bug line, then decide
        if self._comment_count[pr_id] < len(bug_lines):
            idx = self._comment_count[pr_id]
            line = bug_lines[idx]
            self._comment_count[pr_id] += 1

            # Generate category-specific comment with actionability keywords
            from env.data_generator import BUG_KEYWORDS
            keywords = BUG_KEYWORDS.get(bug_category, ["issue"])
            kw = keywords[0] if keywords else "issue"

            comment = self._generate_targeted_comment(bug_category, kw)

            files = observation.get("files", [])
            target_file = files[0].get("filename", "unknown") if files else "unknown"

            return {
                "action_type": "add_comment",
                "comment": comment,
                "target_file": target_file,
                "target_line": line,
            }
        else:
            # All bugs commented — make decision
            self._comment_count[pr_id] = 0
            if severity in ("critical", "high", "medium"):
                return {"action_type": "request_changes"}
            else:
                return {"action_type": "approve"}

    @staticmethod
    def _generate_targeted_comment(bug_category: str, keyword: str) -> str:
        """Generate a relevant, specific, actionable comment."""
        comments = {
            "null_pointer": f"You should add a {keyword} check guard here to prevent NullPointerException. Consider using Optional or adding an early return.",
            "sql_injection": f"This is vulnerable to {keyword}. You should use parameterized queries instead of string concatenation to sanitize user input.",
            "race_condition": f"This has a {keyword} condition. You should add a mutex lock or use atomic operations to ensure thread-safe concurrent access.",
            "logic_error": f"The {keyword} here has an off-by-one boundary issue. Consider checking the edge case and adjusting the logic.",
            "missing_error_handling": f"Missing {keyword} handling here. You should add a try/catch block and handle the error case gracefully.",
            "security_vulnerability": f"This leaks {keyword} information. You should encrypt sensitive data and avoid exposing secrets in logs.",
            "performance_issue": f"This has O(n) {keyword} complexity. Consider adding an index or cache to optimize the query performance.",
            "style_only": f"The {keyword} here doesn't follow conventions. Consider renaming for consistency with the codebase style.",
        }
        return comments.get(bug_category, f"Consider reviewing the {keyword} usage here. You should refactor for clarity.")
