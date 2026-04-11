"""
CodeReviewEnvironment — MCP tool-calling RL environment with REAL code execution.

Like repl_env: agents run real code, see real failures, and submit fixes
that are verified by actual test execution. Not a keyword matcher.

Tools:
  get_code      — retrieve the buggy source code
  run_code      — EXECUTE the code and see output/errors (real Python executor)
  run_tests     — run test cases against the code (see which tests fail)
  analyze_code  — structural analysis
  check_line    — check if a line is near a bug (immediate feedback)
  get_hint      — progressive hints (costs efficiency)
  submit_fix    — submit fixed code; tests are re-run to verify the fix works
  submit_review — submit text review (alternative to submit_fix)

MDP: up to 10 tool calls per episode.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from snippet_bank import generate_episode, BugRecord
from reward import compute_reward
from server.code_executor import execute_code, apply_fix_and_test, SNIPPET_TESTS

MAX_STEPS = 10
LINE_TOLERANCE = 3

TOOLS = [
    {
        "name": "get_code",
        "description": "Get the buggy source code to review. Returns code with line numbers, language, and difficulty.",
        "parameters": {},
    },
    {
        "name": "run_code",
        "description": "Execute the current code and see stdout/stderr/errors. Like a real REPL — see what actually happens when you run it.",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute (optional — defaults to the buggy snippet)"},
        },
    },
    {
        "name": "run_tests",
        "description": "Run test cases against the current buggy code. See which tests pass and which fail. The failures reveal the bugs.",
        "parameters": {},
    },
    {
        "name": "analyze_code",
        "description": "Structural analysis: line count, functions, conditionals, complexity hints.",
        "parameters": {},
    },
    {
        "name": "check_line",
        "description": "Check if a specific line is near a known bug. Immediate reward: +0.15 hit, -0.05 miss.",
        "parameters": {
            "line": {"type": "integer", "description": "Line number to check (1-indexed)"},
        },
    },
    {
        "name": "get_hint",
        "description": "Get a progressive hint about the bugs. Costs -0.05 efficiency per hint.",
        "parameters": {},
    },
    {
        "name": "submit_fix",
        "description": "Submit fixed Python code. The environment runs tests against your fix. If tests pass, you get high reward. This is the BEST way to end an episode.",
        "parameters": {
            "fixed_code": {"type": "string", "description": "The corrected source code"},
            "comment": {"type": "string", "description": "What you fixed and why"},
        },
    },
    {
        "name": "submit_review",
        "description": "Submit a text-based code review (alternative to submit_fix). Lower reward ceiling than submit_fix.",
        "parameters": {
            "issues": {"type": "array", "items": {"type": "string"}, "description": "Bug descriptions found"},
            "flagged_lines": {"type": "array", "items": {"type": "integer"}, "description": "Buggy line numbers"},
            "suggestion": {"type": "string", "description": "Suggested fix description"},
            "comment": {"type": "string", "description": "Review comment"},
        },
    },
]


class CodeReviewEnvironment(
    Environment[CodeReviewAction, CodeReviewObservation, CodeReviewState]
):
    """MCP tool-calling code review environment.

    Agents discover tools via ListToolsAction, then call them via ToolCallAction.
    This matches the pattern used by calendar_env and repl_env in OpenEnv.

    Tools: get_code, analyze_code, check_line, get_hint, submit_review
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
        self._done = False

        # Gold state
        self._original_code = ""
        self._buggy_code = ""
        self._gold_bugs: List[BugRecord] = []
        self._language = "python"
        self._snippet_name = ""

        self._auto_reset(seed=42, difficulty="easy")

    def _auto_reset(self, seed: int, difficulty: str) -> None:
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._difficulty = difficulty
        self._hint_count = 0
        self._trajectory = []
        self._flagged_lines = []
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
        self._done = False

        snippet, buggy_code, gold_bugs = generate_episode(seed=actual_seed, difficulty=difficulty)
        self._original_code = snippet.code
        self._buggy_code = buggy_code
        self._gold_bugs = gold_bugs
        self._language = snippet.language
        self._snippet_name = snippet.name

        return CodeReviewObservation(
            success=True,
            tools_list=TOOLS,
            tool_result={"message": "Environment reset. Use ListToolsAction to discover available tools, then call them."},
            metadata={"episode_id": self._episode_id, "difficulty": difficulty, "language": self._language},
            done=False,
            reward=None,
        )

    def step(
        self,
        action: CodeReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CodeReviewObservation:
        if self._done:
            return CodeReviewObservation(
                success=False,
                error_message="Episode already done",
                metadata={"episode_id": self._episode_id},
                done=True,
                reward=0.0,
            )

        action_type = getattr(action, "action_type", "ToolCallAction")

        if action_type == "ListToolsAction":
            return self._handle_list_tools()
        elif action_type == "ToolCallAction":
            return self._handle_tool_call(action)
        else:
            return CodeReviewObservation(
                success=False,
                error_message=f"Unknown action_type: {action_type}. Use ListToolsAction or ToolCallAction.",
                metadata={"episode_id": self._episode_id},
                done=False,
                reward=0.0,
            )

    @property
    def state(self) -> CodeReviewState:
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
            flagged_lines=list(self._flagged_lines),
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CodeReviewEnv",
            description=(
                "MCP tool-calling environment for automated code review. "
                "Agents use tools to analyze code, check lines, get hints, "
                "and submit structured reviews. 5-signal shaped reward."
            ),
            version="2.0.0",
            author="CodeReviewEnv Team",
        )

    # ─── Action Handlers ─────────────────────────────────────────────

    def _handle_list_tools(self) -> CodeReviewObservation:
        """Return the list of available tools."""
        return CodeReviewObservation(
            success=True,
            tools_list=TOOLS,
            metadata={
                "episode_id": self._episode_id,
                "step": self._step_count,
                "steps_remaining": MAX_STEPS - self._step_count,
            },
            done=False,
            reward=0.0,
        )

    def _handle_tool_call(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Dispatch a tool call to the appropriate handler."""
        tool_name = action.tool_name or ""
        arguments = action.arguments or {}

        self._step_count += 1

        handlers = {
            "get_code": self._tool_get_code,
            "run_code": self._tool_run_code,
            "run_tests": self._tool_run_tests,
            "analyze_code": self._tool_analyze_code,
            "check_line": self._tool_check_line,
            "get_hint": self._tool_get_hint,
            "submit_fix": self._tool_submit_fix,
            "submit_review": self._tool_submit_review,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return CodeReviewObservation(
                success=False,
                error_message=f"Unknown tool: {tool_name}. Available: {list(handlers.keys())}",
                metadata={"episode_id": self._episode_id, "step": self._step_count},
                done=False,
                reward=0.0,
            )

        result = handler(arguments)

        # Force-submit if max steps reached
        if not self._done and self._step_count >= MAX_STEPS:
            return self._force_submit()

        return result

    # ─── Tool Implementations ────────────────────────────────────────

    def _tool_get_code(self, args: Dict) -> CodeReviewObservation:
        """Tool: get_code — return the buggy code for review."""
        # Add line numbers for easier reference
        lines = self._buggy_code.split('\n')
        numbered = '\n'.join(f"L{i+1}: {line}" for i, line in enumerate(lines))

        result = {
            "code": self._buggy_code,
            "code_with_line_numbers": numbered,
            "language": self._language,
            "difficulty": self._difficulty,
            "total_lines": len(lines),
        }
        self._record("get_code", 0.0)

        return CodeReviewObservation(
            success=True,
            tool_result=result,
            metadata={"episode_id": self._episode_id, "step": self._step_count},
            done=False,
            reward=0.0,
        )

    def _tool_analyze_code(self, args: Dict) -> CodeReviewObservation:
        """Tool: analyze_code — structural analysis of the code."""
        lines = self._buggy_code.split('\n')
        n_lines = len(lines)
        n_functions = sum(1 for l in lines if l.strip().startswith(('def ', 'func ', 'function ')))
        n_conditionals = sum(1 for l in lines if any(kw in l for kw in ('if ', 'elif ', 'else:', 'while ', 'for ')))
        n_returns = sum(1 for l in lines if 'return' in l)

        result = {
            "total_lines": n_lines,
            "functions": n_functions,
            "conditionals_and_loops": n_conditionals,
            "return_statements": n_returns,
            "language": self._language,
            "analysis": (
                f"Code has {n_lines} lines, {n_functions} function(s), "
                f"{n_conditionals} conditional/loop(s), {n_returns} return(s). "
                f"Check boundary conditions, null guards, operator usage, and boolean logic."
            ),
        }
        self._record("analyze_code", 0.0)

        return CodeReviewObservation(
            success=True,
            tool_result=result,
            metadata={"episode_id": self._episode_id, "step": self._step_count},
            done=False,
            reward=0.0,
        )

    def _tool_check_line(self, args: Dict) -> CodeReviewObservation:
        """Tool: check_line — check if a line is near a bug. Immediate reward."""
        line = args.get("line", 0)
        if not isinstance(line, int) or line <= 0:
            return CodeReviewObservation(
                success=False,
                error_message="'line' must be a positive integer",
                metadata={"episode_id": self._episode_id, "step": self._step_count},
                done=False,
                reward=0.0,
            )

        reward = 0.0
        hit = False

        if line not in self._flagged_lines:
            self._flagged_lines.append(line)
            for bug in self._gold_bugs:
                for bl in bug.lines:
                    if abs(line - bl) <= LINE_TOLERANCE:
                        hit = True
                        break
                if hit:
                    break

            reward = 0.15 if hit else -0.05
        else:
            reward = 0.0  # Duplicate flag

        self._total_reward += reward
        self._record("check_line", reward)

        result = {
            "line": line,
            "is_suspicious": hit,
            "feedback": "This line is near a known bug location." if hit else "No bug detected near this line.",
            "flagged_lines_so_far": list(self._flagged_lines),
        }

        return CodeReviewObservation(
            success=True,
            tool_result=result,
            metadata={"episode_id": self._episode_id, "step": self._step_count},
            done=False,
            reward=reward,
        )

    def _tool_get_hint(self, args: Dict) -> CodeReviewObservation:
        """Tool: get_hint — progressive hint, costs efficiency."""
        self._hint_count += 1

        if not self._gold_bugs:
            hint = "The code appears clean — no obvious bugs detected."
        else:
            bug_idx = min(self._hint_count - 1, len(self._gold_bugs) - 1)
            bug = self._gold_bugs[bug_idx]
            if self._hint_count == 1:
                hint = f"Look for a {bug.bug_type.replace('_', ' ')} bug in the code."
            elif self._hint_count == 2:
                hint = f"There's a {bug.bug_type.replace('_', ' ')} near line {bug.lines[0]}."
            else:
                hint = f"Bug on line {bug.lines[0]}: {bug.description}"

        self._record("get_hint", 0.0)

        result = {
            "hint": hint,
            "hint_count": self._hint_count,
            "efficiency_cost": f"-{0.05 * self._hint_count:.2f} at final grading",
        }

        return CodeReviewObservation(
            success=True,
            tool_result=result,
            metadata={"episode_id": self._episode_id, "step": self._step_count},
            done=False,
            reward=0.0,
        )

    def _tool_run_code(self, args: Dict) -> CodeReviewObservation:
        """Tool: run_code — EXECUTE Python code and see real output/errors."""
        if self._language != "python":
            return CodeReviewObservation(
                success=True,
                tool_result={"error": f"Execution only supported for Python, got {self._language}",
                             "language": self._language},
                metadata={"episode_id": self._episode_id, "step": self._step_count},
                done=False, reward=0.0,
            )

        code_to_run = args.get("code", self._buggy_code)
        exec_result = execute_code(code_to_run)
        self._record("run_code", 0.0)

        return CodeReviewObservation(
            success=True,
            tool_result={
                "executed": True,
                "success": exec_result["success"],
                "stdout": exec_result["stdout"],
                "stderr": exec_result["stderr"],
                "error": exec_result["error"],
            },
            metadata={"episode_id": self._episode_id, "step": self._step_count},
            done=False, reward=0.0,
        )

    def _tool_run_tests(self, args: Dict) -> CodeReviewObservation:
        """Tool: run_tests — run test cases against the buggy code. See real failures."""
        if self._language != "python":
            return CodeReviewObservation(
                success=True,
                tool_result={"error": f"Tests only available for Python, got {self._language}",
                             "tests_available": False},
                metadata={"episode_id": self._episode_id, "step": self._step_count},
                done=False, reward=0.0,
            )

        test_code = SNIPPET_TESTS.get(self._snippet_name, "")
        if not test_code:
            return CodeReviewObservation(
                success=True,
                tool_result={"tests_available": False, "message": "No test cases for this snippet."},
                metadata={"episode_id": self._episode_id, "step": self._step_count},
                done=False, reward=0.0,
            )

        exec_result = execute_code(self._buggy_code, test_code)
        self._record("run_tests", 0.0)

        return CodeReviewObservation(
            success=True,
            tool_result={
                "tests_available": True,
                "code_executed": exec_result["success"],
                "code_error": exec_result["error"],
                "tests_passed": exec_result["tests_passed"],
                "tests_failed": exec_result["tests_failed"],
                "test_results": exec_result["test_results"],
                "total_tests": exec_result["tests_passed"] + exec_result["tests_failed"],
            },
            metadata={"episode_id": self._episode_id, "step": self._step_count},
            done=False, reward=0.0,
        )

    def _tool_submit_fix(self, args: Dict) -> CodeReviewObservation:
        """Tool: submit_fix — submit corrected code, verified by test execution.

        This is the highest-reward path: if the fix passes all tests,
        the agent gets near-perfect score.
        """
        fixed_code = args.get("fixed_code", "")
        comment = args.get("comment", "")

        if not fixed_code:
            return CodeReviewObservation(
                success=False,
                error_message="fixed_code is required",
                metadata={"episode_id": self._episode_id, "step": self._step_count},
                done=False, reward=0.0,
            )

        # Run tests against the fix
        test_code = SNIPPET_TESTS.get(self._snippet_name, "")
        if test_code and self._language == "python":
            exec_result = apply_fix_and_test(self._buggy_code, fixed_code, test_code)
            total_tests = exec_result["tests_passed"] + exec_result["tests_failed"]
            fix_pass_rate = exec_result["tests_passed"] / total_tests if total_tests > 0 else 0.0
        else:
            exec_result = {"tests_passed": 0, "tests_failed": 0, "test_results": []}
            fix_pass_rate = 0.0

        # Compute reward: test-based (0.60) + review-based (0.40)
        # Test pass rate is the primary signal — this is what makes us different
        test_reward = fix_pass_rate * 0.60

        # Also compute text-based reward for the comment
        _, text_breakdown = compute_reward(
            issues=[f"Fixed: {comment}"] if comment else [],
            flagged_lines=self._flagged_lines,
            suggestion=fixed_code[:200],
            comment=comment,
            gold_bugs=self._gold_bugs,
            step_count=self._step_count,
            hint_count=self._hint_count,
            difficulty=self._difficulty,
        )
        text_reward = text_breakdown.get("weighted_total", 0.0) * 0.40

        total_reward = min(1.0, test_reward + text_reward)

        self._total_reward += total_reward
        self._done = True
        self._record("submit_fix", total_reward)

        breakdown = {
            "test_pass_rate": round(fix_pass_rate, 4),
            "test_reward": round(test_reward, 4),
            "text_reward": round(text_reward, 4),
            "tests_passed": exec_result["tests_passed"],
            "tests_failed": exec_result["tests_failed"],
            "total_reward": round(total_reward, 4),
        }

        return CodeReviewObservation(
            success=True,
            tool_result={
                "fix_accepted": fix_pass_rate > 0.5,
                "test_results": exec_result["test_results"],
                "tests_passed": exec_result["tests_passed"],
                "tests_failed": exec_result["tests_failed"],
                "reward": total_reward,
                "breakdown": breakdown,
            },
            metadata={"episode_id": self._episode_id, "step": self._step_count, "breakdown": breakdown},
            done=True,
            reward=total_reward,
        )

    def _tool_submit_review(self, args: Dict) -> CodeReviewObservation:
        """Tool: submit_review — full 5-signal grading. Ends episode."""
        issues = args.get("issues", [])
        flagged = list(set(self._flagged_lines + args.get("flagged_lines", [])))
        suggestion = args.get("suggestion", "")
        comment = args.get("comment", "")

        if not isinstance(issues, list):
            issues = [str(issues)] if issues else []
        if not isinstance(flagged, list):
            flagged = []

        total_reward, breakdown = compute_reward(
            issues=issues,
            flagged_lines=flagged,
            suggestion=suggestion,
            comment=comment,
            gold_bugs=self._gold_bugs,
            step_count=self._step_count,
            hint_count=self._hint_count,
            difficulty=self._difficulty,
        )

        self._total_reward += total_reward
        self._done = True
        self._record("submit_review", total_reward)

        result = {
            "reward": total_reward,
            "breakdown": breakdown,
            "total_episode_reward": self._total_reward,
            "steps_used": self._step_count,
            "hints_used": self._hint_count,
            "lines_flagged": flagged,
        }

        return CodeReviewObservation(
            success=True,
            tool_result=result,
            metadata={"episode_id": self._episode_id, "step": self._step_count, "breakdown": breakdown},
            done=True,
            reward=total_reward,
        )

    def _force_submit(self) -> CodeReviewObservation:
        """Auto-submit when max steps reached."""
        return self._tool_submit_review({
            "issues": [],
            "flagged_lines": self._flagged_lines,
            "suggestion": "",
            "comment": "",
        })

    # ─── Helpers ─────────────────────────────────────────────────────

    def _record(self, tool_name: str, reward: float) -> None:
        self._trajectory.append({
            "step": self._step_count,
            "tool": tool_name,
            "reward": reward,
            "flagged_lines": list(self._flagged_lines),
            "hint_count": self._hint_count,
        })

    def export_trajectory(self) -> List[Dict]:
        return list(self._trajectory)
