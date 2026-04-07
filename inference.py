"""
CodeReviewEnv — Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if using from_docker_image() (also accepts LOCAL_IMAGE_NAME)

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=severity-labeling env=code-review-env model=openai/gpt-4o-mini
    [STEP] step=1 action=label_severity:high reward=0.50 done=false error=null
    [STEP] step=2 action=label_severity:critical reward=1.00 done=false error=null
    [STEP] step=3 action=label_severity:medium reward=0.80 done=true error=null
    [END] success=true steps=3 score=0.767 rewards=0.50,1.00,0.80
"""

import asyncio
import sys
import json
import os
import re
import textwrap
import inspect
from typing import Any, Dict, List, Optional

from openai import OpenAI

from models import CodeReviewAction, CodeReviewObservation, CodeReviewState
from client import CodeReviewEnv

# ─── .env loading (no extra dependency) ───────────────────────────────────────

def _load_dotenv(dotenv_path: str) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (without overriding)."""
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")  # tolerate simple quoting
                if key:
                    os.environ.setdefault(key, value)
    except FileNotFoundError:
        return


_load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ─── Configuration ────────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")  # If using from_docker_image()
# The platform injects API_BASE_URL and API_KEY — use them directly with OpenAI client.
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
BENCHMARK = "code-review-env"
TEMPERATURE = 0.0
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.3

# Debug: show which API config is active (stderr only)
print(f"[DEBUG] API_BASE_URL = {API_BASE_URL}", file=sys.stderr, flush=True)
print(f"[DEBUG] API_KEY value (last 8) = ...{API_KEY[-8:]}", file=sys.stderr, flush=True)
print(f"[DEBUG] MODEL_NAME = {MODEL_NAME}", file=sys.stderr, flush=True)




async def _maybe_await(value: Any) -> Any:
    """Await value if it's awaitable, else return it."""
    if inspect.isawaitable(value):
        return await value
    return value


# ─── Structured Logging (exact spec format) ──────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── LLM Interface ──────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
    """Call the LLM using OpenAI Client with retry. Returns response text."""
    model_candidates = [MODEL_NAME]
    if "/" in MODEL_NAME:
        model_candidates.append(MODEL_NAME.split("/", 1)[1])

    for attempt in range(max_retries):
        for model in model_candidates:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                return (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                print(
                    f"[DEBUG] Attempt {attempt+1}/{max_retries} failed (model={model}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                # Try next candidate model (if any) before sleeping/retrying.
                continue
        if attempt < max_retries - 1:
            import time
            time.sleep(2 ** attempt)
    return ""


def parse_json_response(response: str) -> Optional[Dict]:
    """Robustly parse JSON from LLM response, handling markdown code blocks."""
    if not response:
        return None
    response = response.strip()
    if response.startswith("```"):
        lines = [l for l in response.split("\n") if not l.strip().startswith("```")]
        response = "\n".join(lines).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ─── System Prompts ──────────────────────────────────────────────────────────

EASY_SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior software engineer performing code review.
    You will receive a pull request with a code diff. Assess the severity of any bugs present.

    Severity scale:
    - "critical": Security vulnerabilities (SQL injection, auth bypass, hardcoded secrets)
    - "high": Crashes or data corruption (null pointer dereference, race conditions)
    - "medium": Logic errors or missing error handling (off-by-one, uncaught exceptions)
    - "low": Performance issues (N+1 queries, unnecessary loops)
    - "none": Style-only changes, no bugs

    Respond ONLY with valid JSON:
    {"action_type": "label_severity", "severity": "<critical|high|medium|low|none>"}
""").strip()

MEDIUM_SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior software engineer managing a code review queue.
    You MUST order ALL the PR IDs by review priority (most urgent first).

    Priority rules (in strict order):
    1. Security vulnerabilities (SQL injection, auth bypass, hardcoded secrets,
       session issues) are ALWAYS highest priority, regardless of severity label.
    2. Higher severity bugs before lower: critical > high > medium > low > none
    3. Within same severity: junior developers need review first (most urgent),
       then mid-level, then senior.
    4. PRs without test coverage should be reviewed before those with tests.

    Bug severity guide:
    - critical: SQL injection, security vulnerabilities, auth bypass, hardcoded secrets
    - high: null pointer / None dereference, race conditions, data corruption
    - medium: logic errors, missing error handling, off-by-one bugs
    - low: performance issues (N+1 queries, unnecessary loops)
    - none: style-only changes, formatting, renaming

    IMPORTANT: You MUST include ALL PR IDs from the queue in your response.
    Do NOT omit any PR ID. Return the complete ordered list.

    Respond ONLY with valid JSON:
    {"action_type": "prioritize", "priority_order": ["PR-XXX", "PR-YYY", ...]}
""").strip()

HARD_COMMENT_PROMPT = textwrap.dedent("""
    You are a senior software engineer performing detailed code review.
    Your task is to add a SPECIFIC, ACTIONABLE review comment targeting a buggy line.

    Instructions:
    1. Read the diff carefully. Look for lines marked with BUG comments or
       common vulnerability patterns.
    2. Target the EXACT line number where the bug is (look at the diff line numbers).
    3. Your comment MUST:
       - Reference the specific bug type (e.g., "null pointer", "SQL injection",
         "race condition", "missing error handling", "logic error")
       - Include a concrete suggestion using words like: "use", "replace", "add",
         "remove", "consider", "should", "instead", "wrap", "avoid", "refactor"
    4. Set target_file to the exact filename from the diff header.
    5. Set target_line to the line number of the buggy code.

    Bug-specific keywords to use in comments:
    - Null pointer: "null", "None", "check", "guard"
    - SQL injection: "injection", "parameterize", "prepared statement", "sanitize"
    - Race condition: "race", "lock", "mutex", "atomic", "thread-safe"
    - Logic error: "off-by-one", "boundary", "condition", "edge case"
    - Missing error handling: "exception", "catch", "error", "handle", "try"
    - Security: "auth", "token", "encrypt", "hash", "secret", "leak"
    - Performance: "complexity", "cache", "optimize", "index", "N+1"

    Respond ONLY with valid JSON:
    {"action_type": "add_comment", "comment": "<specific feedback>", "target_file": "<filename>", "target_line": <line_number>}
""").strip()

HARD_DECISION_PROMPT = textwrap.dedent("""
    You are a senior software engineer completing a code review.
    You have already added review comments. Now make your final decision.

    Rules:
    - If the code has ANY bugs (security, null pointer, race condition, logic error,
      missing error handling): respond with request_changes
    - If the code is clean (style-only, formatting, renaming): respond with approve

    Respond ONLY with valid JSON:
    {"action_type": "request_changes"}
    or
    {"action_type": "approve"}
""").strip()


# ─── Observation Formatting ──────────────────────────────────────────────────

def format_obs_easy(obs: CodeReviewObservation) -> str:
    files_text = ""
    for f in obs.files:
        files_text += f"\n--- {f['filename']} ({f['language']}, {f['lines_changed']} lines) ---\n"
        files_text += f["diff"] + "\n"
    return (
        f"PR: {obs.pr_id} | {obs.title}\n{obs.description}\n"
        f"Author: {obs.author_experience}\n{files_text}\n"
        f"What is the severity?"
    )


def format_obs_medium(obs: CodeReviewObservation) -> str:
    """Format medium task observation showing all available context.

    Shows the visible PR's full details plus the queue IDs.
    The agent needs to prioritize all IDs in the queue.
    """
    pr_count = len(obs.review_queue)
    queue_text = f"Review Queue ({pr_count} PRs to prioritize):\n"
    for i, pr_id in enumerate(obs.review_queue, 1):
        queue_text += f"  {i}. {pr_id}\n"

    # Show the visible PR's full details to help with analysis
    visible_pr = (
        f"\nVisible PR Details (one of the PRs in the queue):\n"
        f"  PR ID: {obs.pr_id}\n"
        f"  Title: {obs.title}\n"
        f"  Description: {obs.description}\n"
        f"  Author Experience: {obs.author_experience}\n"
    )

    files_text = ""
    for f in obs.files:
        has_tests = f.get('has_tests', 'unknown')
        files_text += f"\n  --- {f['filename']} ({f['language']}, {f['lines_changed']} lines, tests: {has_tests}) ---\n"
        files_text += f"  {f['diff']}\n"

    return (
        f"Review Queue — Step {obs.step_number + 1}\n"
        f"{queue_text}"
        f"{visible_pr}"
        f"{files_text}\n"
        f"IMPORTANT: Order ALL {pr_count} PRs by priority. You MUST include every PR ID "
        f"listed above in your priority_order array.\n"
        f"Apply these rules: security PRs first, then by severity "
        f"(critical>high>medium>low>none), then junior authors before senior."
    )


def format_obs_hard(obs: CodeReviewObservation, phase: str = "comment") -> str:
    """Format hard task observation with line numbers and phase info."""
    files_text = ""
    for f in obs.files:
        files_text += f"\n--- {f['filename']} ({f['language']}, {f['lines_changed']} lines) ---\n"
        # Add line numbers to diff for precise targeting
        diff_lines = f["diff"].split("\n")
        numbered_diff = ""
        line_num = 1
        for dl in diff_lines:
            if dl.startswith("@@"):
                # Parse the @@ line to get starting line number
                try:
                    parts = dl.split("+")[1].split(",")[0].split(" ")[0]
                    line_num = int(parts)
                except (IndexError, ValueError):
                    pass
                numbered_diff += dl + "\n"
            elif dl.startswith("+") or dl.startswith(" ") or not dl.startswith("-"):
                numbered_diff += f"L{line_num}: {dl}\n"
                line_num += 1
            else:
                numbered_diff += f"     {dl}\n"  # removed lines don't get numbers
        files_text += numbered_diff

    comments = ""
    if obs.existing_comments:
        comments = f"\nYou have already made {len(obs.existing_comments)} comment(s) on this PR.\n"

    if phase == "comment":
        instruction = (
            f"\nFind a BUG in this code and add a specific comment targeting the buggy line.\n"
            f"Look for lines containing bug patterns: null checks, SQL injection, race conditions, "
            f"missing error handling, logic errors, security issues.\n"
            f"You MUST respond with add_comment action including target_file and target_line."
        )
    else:
        instruction = (
            f"\nYou have reviewed this PR and added comments. Now make your final decision.\n"
            f"If any bugs were found, respond with request_changes. If code is clean, respond with approve."
        )

    return (
        f"PR: {obs.pr_id} | {obs.title}\n"
        f"Description: {obs.description}\n"
        f"Author: {obs.author_experience}\n"
        f"{files_text}"
        f"{comments}"
        f"{instruction}"
    )


def action_to_str(action_dict: Dict) -> str:
    """Convert action dict to a compact string for logging."""
    at = action_dict.get("action_type", "unknown")
    if at == "label_severity":
        return f"label_severity:{action_dict.get('severity', '?')}"
    elif at == "prioritize":
        order = action_dict.get("priority_order", [])
        return f"prioritize:[{','.join(order)}]"
    elif at == "add_comment":
        comment = (action_dict.get("comment", ""))[:50]
        return f"add_comment:{comment}"
    else:
        return at


# ─── Task Runners ────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "task_name": "severity-labeling",
        "system_prompt": EASY_SYSTEM_PROMPT,
        "max_steps": 5,
        "format_obs": format_obs_easy,
        "default_action": lambda obs: {"action_type": "label_severity", "severity": "medium"},
    },
    "medium": {
        "task_name": "queue-prioritization",
        "system_prompt": MEDIUM_SYSTEM_PROMPT,
        "max_steps": 3,
        "format_obs": format_obs_medium,
        "default_action": lambda obs: {"action_type": "prioritize", "priority_order": list(obs.review_queue)},
    },
    "hard": {
        "task_name": "feedback-generation",
        "system_prompt": HARD_COMMENT_PROMPT,  # default prompt; run_task switches between comment/decision
        "max_steps": 18,
        "format_obs": lambda obs: format_obs_hard(obs, phase="comment"),
        "default_action": lambda obs: {
            "action_type": "add_comment",
            "comment": "Potential bug detected — please add error handling or validation.",
            "target_file": obs.files[0]["filename"] if obs.files else "unknown",
            "target_line": 10,
        },
    },
}


def _postprocess_medium_action(action_dict: Dict, obs: CodeReviewObservation) -> Dict:
    """Ensure the medium task response includes ALL PR IDs from the queue.

    If the LLM omits some PR IDs, we append them at the end.
    If the LLM includes IDs not in the queue, we remove them.
    """
    if action_dict.get("action_type") != "prioritize":
        return action_dict

    queue_ids = set(obs.review_queue)
    predicted = action_dict.get("priority_order", [])

    # Remove IDs not in the queue
    cleaned = [pr_id for pr_id in predicted if pr_id in queue_ids]
    included = set(cleaned)

    # Append missing IDs (maintain queue order for unknowns)
    for pr_id in obs.review_queue:
        if pr_id not in included:
            cleaned.append(pr_id)
            included.add(pr_id)

    action_dict["priority_order"] = cleaned
    return action_dict


def _build_hard_action(
    llm_client: OpenAI,
    obs: CodeReviewObservation,
    phase: str,
    comments_on_pr: int,
) -> Dict:
    """Build an action for the hard task based on the current phase.

    phase='comment': Generate an add_comment action targeting a specific bug.
    phase='decide':  Generate an approve/request_changes action.
    """
    if phase == "comment":
        system_prompt = HARD_COMMENT_PROMPT
        user_prompt = format_obs_hard(obs, phase="comment")
        if comments_on_pr > 0:
            user_prompt += f"\nYou have made {comments_on_pr} comment(s) so far. Target a DIFFERENT bug line."
    else:
        system_prompt = HARD_DECISION_PROMPT
        user_prompt = format_obs_hard(obs, phase="decide")

    try:
        response = call_llm(llm_client, system_prompt, user_prompt)
        parsed = parse_json_response(response)
    except Exception as e:
        print(f"[DEBUG] Hard task LLM call failed: {e}", file=sys.stderr, flush=True)
        parsed = None

    if parsed and parsed.get("action_type"):
        action_dict = parsed
    else:
        # Fallback
        if phase == "comment":
            filename = obs.files[0]["filename"] if obs.files else "unknown"
            action_dict = {
                "action_type": "add_comment",
                "comment": "Potential bug — consider adding error handling or input validation to prevent crashes.",
                "target_file": filename,
                "target_line": 15,
            }
        else:
            action_dict = {"action_type": "request_changes"}

    # Force correct action type for the phase
    if phase == "comment" and action_dict.get("action_type") not in ("add_comment",):
        filename = obs.files[0]["filename"] if obs.files else "unknown"
        action_dict = {
            "action_type": "add_comment",
            "comment": action_dict.get("comment", "Bug detected — should add proper validation."),
            "target_file": action_dict.get("target_file", filename),
            "target_line": action_dict.get("target_line", 15),
        }
    elif phase == "decide" and action_dict.get("action_type") not in ("approve", "request_changes"):
        action_dict = {"action_type": "request_changes"}

    return action_dict


async def run_task(env: CodeReviewEnv, llm_client: OpenAI, task: str) -> float:
    """Run a single task episode. Returns normalized score.
    
    NOTE: Caller is responsible for emitting [START] and [END] lines.
    This function only emits [STEP] lines.
    """
    config = TASK_CONFIGS[task]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01

    result = await _maybe_await(env.reset(seed=42, task=task))
    obs = result.observation

    # Hard task tracking
    hard_comments_on_pr = 0
    hard_current_pr = obs.pr_id if task == "hard" else None
    HARD_COMMENTS_PER_PR = 1  # 1 targeted comment then decide (maximizes mean reward)

    for step in range(1, config["max_steps"] + 1):
        if result.done:
            break

        if task == "hard":
            # ── Hard task: two-phase comment → decide workflow ──
            # Track PR transitions
            if obs.pr_id != hard_current_pr:
                hard_comments_on_pr = 0
                hard_current_pr = obs.pr_id

            # Decide phase
            if hard_comments_on_pr < HARD_COMMENTS_PER_PR:
                phase = "comment"
            else:
                phase = "decide"

            action_dict = _build_hard_action(
                llm_client, obs, phase, hard_comments_on_pr
            )

            if phase == "comment":
                hard_comments_on_pr += 1

        else:
            # ── Easy / Medium tasks: standard flow ──
            try:
                user_prompt = config["format_obs"](obs)
                response = call_llm(llm_client, config["system_prompt"], user_prompt)
                parsed = parse_json_response(response)
            except Exception as e:
                print(f"[DEBUG] LLM call failed at step {step}: {e}", file=sys.stderr, flush=True)
                parsed = None

            # Build action
            if parsed and parsed.get("action_type"):
                action_dict = parsed
            else:
                action_dict = config["default_action"](obs)

            # Post-process medium task to ensure all PR IDs are included
            if task == "medium":
                action_dict = _postprocess_medium_action(action_dict, obs)

        # Ensure valid action fields
        try:
            action = CodeReviewAction(**action_dict)
        except Exception as e:
            print(f"[DEBUG] Action validation failed: {e}", file=sys.stderr, flush=True)
            action_dict = config["default_action"](obs)
            action = CodeReviewAction(**action_dict)

        # Step
        try:
            result = await _maybe_await(env.step(action))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None
        except Exception as e:
            print(f"[DEBUG] env.step() failed: {e}", file=sys.stderr, flush=True)
            reward = 0.0
            done = True
            error = str(e)

        rewards.append(reward)
        steps_taken = step

        log_step(
            step=step,
            action=action_to_str(action_dict),
            reward=reward,
            done=done,
            error=error,
        )

        if done:
            break

    # Compute score: mean reward normalized to [0, 1]
    if rewards:
        score = sum(rewards) / len(rewards)
        score = min(max(score, 0.01), 0.99)

    return score, steps_taken, rewards


# ─── Main ────────────────────────────────────────────────────────────────────

async def main() -> int:
    # Initialize LLM client using the injected API_BASE_URL and API_KEY
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores = {}
    space_url = os.getenv("SPACE_URL", "https://ragavrida-code-review-env.hf.space")

    for task in ["easy", "medium", "hard"]:
        config = TASK_CONFIGS[task]
        task_name = config["task_name"]
        score = 0.0
        steps_taken = 0
        rewards: List[float] = []
        success = False
        env = None

        # Always emit [START] to stdout
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:

            # Fresh env per task to avoid reusing a closed ws connection.
            if IMAGE_NAME:
                print(f"[DEBUG] Connecting to Docker image: {IMAGE_NAME}", file=sys.stderr, flush=True)
                env = await CodeReviewEnv.from_docker_image(IMAGE_NAME)
            else:
                print(f"[DEBUG] Connecting to server: {space_url}", file=sys.stderr, flush=True)
                env = CodeReviewEnv(base_url=space_url)

            score, steps_taken, rewards = await run_task(env, llm_client, task)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)

        finally:
            # Always emit [END] to stdout — even on failure
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

            if env is not None:
                try:
                    close_result = env.close()
                    if inspect.isawaitable(close_result):
                        await close_result
                except Exception as e:
                    print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)

        scores[task] = score

    composite = sum(scores.values()) / max(len(scores), 1)
    print(f"\n[SUMMARY] composite={composite:.3f} easy={scores.get('easy',0):.3f} medium={scores.get('medium',0):.3f} hard={scores.get('hard',0):.3f}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
