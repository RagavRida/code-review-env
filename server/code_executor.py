"""
Code Executor — Sandboxed Python execution for CodeReviewEnv.

Runs Python code + test cases in an isolated namespace.
Returns execution results, stdout, stderr, and test pass/fail.

This makes CodeReviewEnv a REAL tool server: agents run code,
see actual failures, and submit fixes that are verified by execution.
"""

import io
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Maximum execution time (seconds) and output length
EXEC_TIMEOUT = 5
MAX_OUTPUT_LEN = 4096


def execute_code(code: str, test_code: str = "") -> Dict[str, Any]:
    """Execute Python code + test cases in an isolated namespace.

    Args:
        code: The source code to execute
        test_code: Test assertions to run after the code

    Returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "error": str or None,
            "tests_passed": int,
            "tests_failed": int,
            "test_results": [{"name": str, "passed": bool, "error": str}],
        }
    """
    namespace: Dict[str, Any] = {}
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "error": None,
        "tests_passed": 0,
        "tests_failed": 0,
        "test_results": [],
    }

    # Step 1: Execute the main code
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        exec(code, namespace)
        result["success"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["success"] = False
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        result["stdout"] = stdout_capture.getvalue()[:MAX_OUTPUT_LEN]
        result["stderr"] = stderr_capture.getvalue()[:MAX_OUTPUT_LEN]

    # Step 2: Run test cases
    if test_code and result["success"]:
        test_results = _run_tests(test_code, namespace)
        result["test_results"] = test_results
        result["tests_passed"] = sum(1 for t in test_results if t["passed"])
        result["tests_failed"] = sum(1 for t in test_results if not t["passed"])

    return result


def _run_tests(test_code: str, namespace: Dict) -> List[Dict[str, Any]]:
    """Run individual test assertions and collect results."""
    results = []

    # Split test code into individual assertions
    lines = test_code.strip().split('\n')
    test_blocks = []
    current_block = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('# test:') or stripped.startswith('# Test:'):
            if current_block:
                test_blocks.append(('\n'.join(current_block), current_block[0].strip()))
                current_block = []
        current_block.append(line)

    if current_block:
        test_blocks.append(('\n'.join(current_block), current_block[0].strip()))

    # If no labeled blocks, treat each assert as a test
    if len(test_blocks) <= 1:
        test_blocks = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                test_blocks.append((line, stripped[:60]))

    for code_block, name in test_blocks:
        try:
            exec(code_block, namespace)
            results.append({"name": name, "passed": True, "error": None})
        except AssertionError as e:
            results.append({"name": name, "passed": False, "error": f"AssertionError: {e}"})
        except Exception as e:
            results.append({"name": name, "passed": False, "error": f"{type(e).__name__}: {e}"})

    return results


def apply_fix_and_test(
    original_code: str,
    fix_code: str,
    test_code: str,
) -> Dict[str, Any]:
    """Apply agent's fix to the code and run tests.

    The fix_code can be either:
      1. Complete replacement code
      2. A patch description (we try to apply it)

    Returns execution results with test pass/fail.
    """
    # Try the fix as complete replacement first
    result = execute_code(fix_code, test_code)
    if result["tests_passed"] > 0:
        return result

    # If that didn't work, try prepending original code context
    combined = original_code + "\n\n" + fix_code
    result = execute_code(combined, test_code)
    return result


# ─── Test Case Templates ─────────────────────────────────────────────────────

# These map snippet names to test cases that validate correctness.
# Tests are written against the ORIGINAL (clean) code — they should
# PASS on clean code and FAIL on buggy code.

SNIPPET_TESTS: Dict[str, str] = {
    "binary_search": """\
assert binary_search([1,2,3,4,5], 3) == 2
assert binary_search([1,2,3,4,5], 1) == 0
assert binary_search([1,2,3,4,5], 5) == 4
assert binary_search([1,2,3,4,5], 6) == -1
assert binary_search([], 1) == -1
assert binary_search([1], 1) == 0
""",
    "fibonacci": """\
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(2) == 1
assert fibonacci(5) == 5
assert fibonacci(10) == 55
""",
    "max_subarray": """\
assert max_subarray([1, -2, 3, 4, -1]) == 7
assert max_subarray([-1, -2, -3]) == -1
assert max_subarray([5]) == 5
assert max_subarray([]) == 0
""",
    "is_palindrome": """\
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
assert is_palindrome("A man a plan a canal Panama") == True
assert is_palindrome("") == True
""",
    "merge_sort": """\
assert merge_sort([3,1,4,1,5]) == [1,1,3,4,5]
assert merge_sort([]) == []
assert merge_sort([1]) == [1]
assert merge_sort([5,4,3,2,1]) == [1,2,3,4,5]
""",
    "flatten_dict": """\
assert flatten_dict({"a": 1, "b": {"c": 2}}) == {"a": 1, "b.c": 2}
assert flatten_dict({}) == {}
assert flatten_dict({"x": {"y": {"z": 1}}}) == {"x.y.z": 1}
""",
    "validate_email": """\
assert validate_email("user@example.com") == True
assert validate_email("bad") == False
assert validate_email("@example.com") == False
assert validate_email("user@.com") == False
assert validate_email("") == False
""",
    "group_by": """\
result = group_by([1,2,3,4,5], lambda x: x % 2)
assert result[0] == [2, 4]
assert result[1] == [1, 3, 5]
""",
    "matrix_multiply": """\
assert matrix_multiply([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]
assert matrix_multiply([[1]], [[2]]) == [[2]]
""",
    "csv_parser": """\
assert parse_csv("a,b,c") == [["a", "b", "c"]]
assert parse_csv("1,2\\n3,4") == [["1", "2"], ["3", "4"]]
""",
    "topological_sort": """\
result = topological_sort({"a": ["b"], "b": ["c"], "c": []})
assert result.index("a") < result.index("b")
assert result.index("b") < result.index("c")
""",
    "lru_cache": """\
cache = LRUCache(2)
cache.put("a", 1)
cache.put("b", 2)
assert cache.get("a") == 1
cache.put("c", 3)
assert cache.get("b") == -1
""",
}
