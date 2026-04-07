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
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
BENCHMARK = "code-review-env"
TEMPERATURE = 0.0
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.3

def _maybe_disable_proxies() -> None:
    """
    OpenEnv's websocket client will honor HTTP(S)/SOCKS proxy env vars.
    For local runs, misconfigured proxies are a common source of connection failure.
    Set USE_PROXY=1 to keep proxy env vars enabled.
    """
    if os.getenv("USE_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    for k in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "SOCKS_PROXY",
        "SOCKS5_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "socks_proxy",
        "socks5_proxy",
    ):
        os.environ.pop(k, None)

    # Ensure local proxy bypass includes the default HF Space host.
    host = os.getenv("SPACE_URL", "https://ragavrida-code-review-env.hf.space")
    try:
        host = host.split("://", 1)[1].split("/", 1)[0]
    except Exception:
        host = "ragavrida-code-review-env.hf.space"
    for k in ("NO_PROXY", "no_proxy"):
        cur = os.getenv(k, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        if host not in parts:
            parts.append(host)
            os.environ[k] = ",".join(parts)


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
    Order the PRs by review priority (most urgent first).

    Priority rules:
    1. Security-related PRs are always highest priority
    2. Higher severity bugs before lower severity
    3. Junior developers need more urgent review
    4. PRs without tests should be reviewed earlier

    Respond ONLY with valid JSON:
    {"action_type": "prioritize", "priority_order": ["PR-XXX", "PR-YYY", ...]}
""").strip()

HARD_SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior software engineer performing detailed code review.
    You must: 1) Add specific, actionable review comments targeting buggy lines
              2) Then approve or request changes

    For comments:
    {"action_type": "add_comment", "comment": "<specific feedback>", "target_file": "<filename>", "target_line": <line>}

    When done reviewing:
    {"action_type": "request_changes"} if bugs found, or {"action_type": "approve"} if clean.

    Respond ONLY with valid JSON.
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
    queue_text = f"PRs to prioritize: {', '.join(obs.review_queue)}\n"
    files_text = ""
    for f in obs.files:
        files_text += f"\n--- {f['filename']} ({f['language']}) ---\n"
        files_text += (f["diff"][:300] + "...\n") if len(f["diff"]) > 300 else f["diff"] + "\n"
    return f"Review Queue — Step {obs.step_number + 1}\n{queue_text}{files_text}\nOrder by priority."


def format_obs_hard(obs: CodeReviewObservation) -> str:
    files_text = ""
    for f in obs.files:
        files_text += f"\n--- {f['filename']} ({f['language']}, {f['lines_changed']} lines) ---\n"
        files_text += f["diff"] + "\n"
    comments = ""
    if obs.existing_comments:
        comments = "\nYour previous comments:\n" + "\n".join(f"  - {c}" for c in obs.existing_comments) + "\n"
    return f"PR: {obs.pr_id} | {obs.title}\n{obs.description}\n{files_text}{comments}\nReview this code."


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
        "system_prompt": HARD_SYSTEM_PROMPT,
        "max_steps": 18,
        "format_obs": format_obs_hard,
        "default_action": lambda obs: {"action_type": "request_changes"},
    },
}


async def run_task(env: CodeReviewEnv, llm_client: OpenAI, task: str) -> float:
    """Run a single task episode. Returns normalized score.
    
    NOTE: Caller is responsible for emitting [START] and [END] lines.
    This function only emits [STEP] lines.
    """
    config = TASK_CONFIGS[task]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0

    result = await _maybe_await(env.reset(seed=42))
    obs = result.observation

    for step in range(1, config["max_steps"] + 1):
        if result.done:
            break

        # Get LLM response
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
        score = min(max(score, 0.0), 1.0)

    return score, steps_taken, rewards


# ─── Main ────────────────────────────────────────────────────────────────────

async def main() -> int:
    # Initialize LLM client — if missing, we still emit [START]/[END] per task
    llm_client = None
    try:
        if not API_KEY:
            print("[DEBUG] No API key found (HF_TOKEN / OPENAI_API_KEY / API_KEY)", file=sys.stderr, flush=True)
        else:
            llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] Failed to create LLM client: {e}", file=sys.stderr, flush=True)

    scores = {}
    _maybe_disable_proxies()
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
            if not llm_client:
                raise RuntimeError("No LLM client available (missing API key)")

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
