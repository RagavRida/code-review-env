"""
CodeReviewEnvironment — OpenEnv-compliant multi-step RL environment for code review.

Multi-step MDP design:
  reset()  →  observe buggy code (done=False)
  step(analyze)       →  get structural analysis of the code (free, done=False)
  step(flag_line)     →  flag a line as buggy; immediate reward if correct (done=False)
  step(request_hint)  →  get a progressive hint; costs -0.05 efficiency (done=False)
  step(submit_review) →  full 5-signal grading; incorporates all prior flags (done=True)

Episode ends on submit_review OR after max_steps (5).
If agent hits max_steps without submitting, a forced grading uses accumulated flags.

Procedural generation: every seed produces a unique episode via
snippet bank + AST/regex bug injectors.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from snippet_bank import generate_episode, BugRecord
from reward import compute_reward, _line_f1

MAX_STEPS = 5
LINE_TOLERANCE = 3


class CodeReviewEnvironment(
    Environment[CodeReviewAction, CodeReviewObservation, CodeReviewState]
):
    """OpenEnv-compliant multi-step code review RL environment.

    Agents can take up to 5 actions per episode:
      analyze        — free structural analysis
      flag_line      — intermediate line-flagging with immediate reward
      request_hint   — progressive hints with efficiency cost
      submit_review  — final grading across 5 signals

    This is a genuine multi-step MDP where earlier decisions (which lines
    to flag, whether to request hints) affect the final reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any):
        super().__init__()
        self._episode_id = ""
        self._step_count = 0
        self._total_reward = 0.0
        self._difficulty = "easy"
        self._hint_count = 0
        self._trajectory: List[Dict[str, Any]] = []
        self._flagged_lines: List[int] = []
        self._analysis_text: Optional[str] = None
        self._last_hint: Optional[str] = None

        # Gold state (hidden from agent)
        self._original_code = ""
        self._buggy_code = ""
        self._gold_bugs: List[BugRecord] = []
        self._language = "python"
        self._snippet_name = ""
        self._done = False

        # Auto-reset
        self._auto_reset(seed=42, difficulty="easy")

    def _auto_reset(self, seed: int, difficulty: str) -> None:
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._difficulty = difficulty
        self._hint_count = 0
        self._trajectory = []
        self._flagged_lines = []
        self._analysis_text = None
        self._last_hint = None
        self._done = False

        snippet, buggy_code, gold_bugs = generate_episode(seed=seed, difficulty=difficulty)
        self._original_code = snippet.code
        self._buggy_code = buggy_code
        self._gold_bugs = gold_bugs
        self._language = snippet.language
        self._snippet_name = snippet.name

    # ─── OpenEnv API ─────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        """Reset and return initial observation with buggy code."""
        actual_seed = seed if seed is not None else 42
        difficulty = kwargs.get("difficulty") or kwargs.get("task", self._difficulty)
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "easy"

        self._difficulty = difficulty
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._hint_count = 0
        self._trajectory = []
        self._flagged_lines = []
        self._analysis_text = None
        self._last_hint = None
        self._done = False

        snippet, buggy_code, gold_bugs = generate_episode(seed=actual_seed, difficulty=difficulty)
        self._original_code = snippet.code
        self._buggy_code = buggy_code
        self._gold_bugs = gold_bugs
        self._language = snippet.language
        self._snippet_name = snippet.name

        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: CodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        """Execute one step. Supports 4 action types for multi-step review."""
        if self._done:
            return self._build_observation(reward=0.0, done=True)

        self._step_count += 1
        action_type = getattr(action, 'action_type', 'submit_review') or 'submit_review'

        if action_type == "analyze":
            return self._handle_analyze(action)
        elif action_type == "flag_line":
            return self._handle_flag_line(action)
        elif action_type == "request_hint":
            return self._handle_request_hint(action)
        elif action_type == "submit_review":
            return self._handle_submit_review(action)
        else:
            # Unknown action type — treat as submit_review
            return self._handle_submit_review(action)

    @property
    def state(self) -> CodeReviewState:
        """Full environment state — includes gold answers for debugging."""
        return CodeReviewState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            original_code=self._original_code,
            buggy_code=self._buggy_code,
            gold_bugs=[
                {"description": b.description, "lines": b.lines, "fix": b.fix, "bug_type": b.bug_type}
                for b in self._gold_bugs
            ],
            language=self._language,
            difficulty=self._difficulty,
            hint_count=self._hint_count,
            snippet_name=self._snippet_name,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CodeReviewEnv",
            description=(
                "Multi-step Semantic MDP for code review. "
                "Agents analyze code, flag buggy lines, request hints, "
                "and submit structured reviews. 5-signal shaped reward."
            ),
            version="2.0.0",
            author="CodeReviewEnv Team",
        )

    # ─── Action Handlers ─────────────────────────────────────────────

    def _handle_analyze(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Analyze action: provide structural analysis of the code (free)."""
        lines = self._buggy_code.split('\n')
        n_lines = len(lines)
        n_functions = sum(1 for l in lines if l.strip().startswith(('def ', 'func ', 'function ')))
        n_conditionals = sum(1 for l in lines if any(kw in l for kw in ('if ', 'elif ', 'else:', 'while ', 'for ')))

        self._analysis_text = (
            f"Code has {n_lines} lines, {n_functions} function(s), "
            f"{n_conditionals} conditional/loop statement(s). "
            f"Language: {self._language}. "
            f"Look for boundary conditions, null checks, operator usage, and boolean logic."
        )

        reward = 0.0  # Free action — no reward or penalty
        self._record_transition("analyze", reward, {})

        if self._step_count >= MAX_STEPS:
            return self._force_submit()

        return self._build_observation(reward=reward, done=False)

    def _handle_flag_line(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Flag a line as buggy. Immediate reward if within tolerance of a gold bug line."""
        line = action.line
        if line is None:
            # Try flagged_lines as fallback
            if action.flagged_lines:
                line = action.flagged_lines[0]
            else:
                line = 0

        reward = 0.0
        breakdown = {}

        if line > 0 and line not in self._flagged_lines:
            self._flagged_lines.append(line)

            # Check if this line is near any gold bug
            hit = False
            for bug in self._gold_bugs:
                for bl in bug.lines:
                    if abs(line - bl) <= LINE_TOLERANCE:
                        hit = True
                        break
                if hit:
                    break

            if hit:
                reward = 0.15  # Immediate positive signal for correct flag
                breakdown["line_flag_hit"] = 0.15
            else:
                reward = -0.05  # Small penalty for false flag
                breakdown["line_flag_miss"] = -0.05
        else:
            reward = 0.0
            breakdown["duplicate_or_invalid"] = 0.0

        self._total_reward += reward
        self._record_transition("flag_line", reward, breakdown)

        if self._step_count >= MAX_STEPS:
            return self._force_submit()

        return self._build_observation(reward=reward, done=False, breakdown=breakdown)

    def _handle_request_hint(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Request a hint. Costs efficiency penalty but helps find bugs."""
        self._hint_count += 1

        if not self._gold_bugs:
            hint = "The code looks clean — no obvious bugs."
        else:
            bug_idx = min(self._hint_count - 1, len(self._gold_bugs) - 1)
            bug = self._gold_bugs[bug_idx]

            if self._hint_count == 1:
                hint = f"Look for a {bug.bug_type.replace('_', ' ')} bug in the code."
            elif self._hint_count == 2:
                hint = f"There's a {bug.bug_type.replace('_', ' ')} near line {bug.lines[0]}."
            else:
                hint = f"Bug on line {bug.lines[0]}: {bug.description}"

        self._last_hint = hint

        reward = 0.0  # No immediate reward, but costs efficiency at final grading
        self._record_transition("request_hint", reward, {"hint_count": self._hint_count})

        if self._step_count >= MAX_STEPS:
            return self._force_submit()

        return self._build_observation(reward=reward, done=False)

    def _handle_submit_review(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Submit final review. Full 5-signal grading. Ends episode."""
        # Merge any previously flagged lines into the submission
        all_flagged = list(set(self._flagged_lines + (action.flagged_lines or [])))

        total_reward, breakdown = compute_reward(
            issues=action.issues or [],
            flagged_lines=all_flagged,
            suggestion=action.suggestion or "",
            comment=action.comment or "",
            gold_bugs=self._gold_bugs,
            step_count=self._step_count,
            hint_count=self._hint_count,
            difficulty=self._difficulty,
        )

        self._total_reward += total_reward
        self._done = True

        self._record_transition("submit_review", total_reward, breakdown)

        return self._build_observation(reward=total_reward, done=True, breakdown=breakdown)

    def _force_submit(self) -> CodeReviewObservation:
        """Auto-submit when max steps reached. Uses accumulated flags."""
        total_reward, breakdown = compute_reward(
            issues=[],
            flagged_lines=self._flagged_lines,
            suggestion="",
            comment="",
            gold_bugs=self._gold_bugs,
            step_count=self._step_count,
            hint_count=self._hint_count,
            difficulty=self._difficulty,
        )

        self._total_reward += total_reward
        self._done = True

        self._record_transition("forced_submit", total_reward, breakdown)

        return self._build_observation(reward=total_reward, done=True, breakdown=breakdown)

    # ─── MCP Tool Methods ────────────────────────────────────────────

    def get_code_snippet(self) -> Dict[str, Any]:
        return {
            "code": self._buggy_code,
            "language": self._language,
            "difficulty": self._difficulty,
            "snippet_name": self._snippet_name,
            "step_count": self._step_count,
            "done": self._done,
        }

    def request_hint(self) -> Dict[str, Any]:
        self._hint_count += 1
        if not self._gold_bugs:
            return {"hint": "No bugs found.", "hint_count": self._hint_count, "penalty": 0.05}
        bug = self._gold_bugs[min(self._hint_count - 1, len(self._gold_bugs) - 1)]
        if self._hint_count == 1:
            hint = f"Look for a {bug.bug_type.replace('_', ' ')} bug."
        elif self._hint_count == 2:
            hint = f"Bug near line {bug.lines[0]}."
        else:
            hint = f"Line {bug.lines[0]}: {bug.description}"
        return {"hint": hint, "hint_count": self._hint_count, "penalty": 0.05 * self._hint_count}

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "max_steps": MAX_STEPS,
            "difficulty": self._difficulty,
            "language": self._language,
            "hint_count": self._hint_count,
            "flagged_lines": self._flagged_lines,
            "done": self._done,
            "total_reward": self._total_reward,
        }

    # ─── Internal helpers ────────────────────────────────────────────

    def _build_observation(
        self,
        reward: float,
        done: bool,
        breakdown: Optional[Dict[str, float]] = None,
    ) -> CodeReviewObservation:
        return CodeReviewObservation(
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._episode_id,
                "step": self._step_count,
                "difficulty": self._difficulty,
                "language": self._language,
            },
            code=self._buggy_code,
            language=self._language,
            difficulty=self._difficulty,
            instructions=(
                "Review the code. You can:\n"
                "  analyze       — get structural analysis (free)\n"
                "  flag_line     — flag a line number as buggy (immediate feedback)\n"
                "  request_hint  — get a hint (costs efficiency)\n"
                "  submit_review — submit final review (ends episode)\n"
                f"Step {self._step_count}/{MAX_STEPS} | "
                f"Language: {self._language} | Difficulty: {self._difficulty}"
            ),
            step_number=self._step_count,
            episode_budget=MAX_STEPS - self._step_count,
            hint=self._last_hint,
            flagged_so_far=list(self._flagged_lines),
            analysis=self._analysis_text,
            reward_breakdown=breakdown,
        )

    def _record_transition(self, action_type: str, reward: float, breakdown: Dict) -> None:
        self._trajectory.append({
            "step": self._step_count,
            "action_type": action_type,
            "reward": reward,
            "breakdown": breakdown,
            "flagged_lines": list(self._flagged_lines),
            "hint_count": self._hint_count,
        })

    def export_trajectory(self) -> List[Dict]:
        return list(self._trajectory)

    def get_system_prompt(self) -> str:
        return (
            "You are a senior software engineer performing code review.\n"
            "You will receive a code snippet that may contain bugs.\n\n"
            "You have up to 5 actions per episode:\n"
            "  analyze       — get structural analysis of the code\n"
            "  flag_line     — flag a specific line as buggy (immediate feedback)\n"
            "  request_hint  — get a hint about a bug (costs efficiency)\n"
            "  submit_review — submit your final review\n\n"
            "For flag_line:\n"
            '  {"action_type": "flag_line", "line": 7}\n\n'
            "For submit_review:\n"
            '  {"action_type": "submit_review", "issues": [...], '
            '"flagged_lines": [...], "suggestion": "...", "comment": "..."}\n'
        )
