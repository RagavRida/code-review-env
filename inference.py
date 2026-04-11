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
    [START] task=code-review-easy env=code-review-env model=openai/gpt-4o-mini
    [STEP] step=1 action=review:3_issues reward=0.75 done=true error=null
    [END] success=true steps=1 score=0.750 rewards=0.75
"""

import asyncio
import sys
import json
import os
import re
import inspect
import time
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
                value = value.strip().strip("\"'")
                if key:
                    os.environ.setdefault(key, value)
    except FileNotFoundError:
        return


_load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ─── Configuration ────────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
BENCHMARK = "code-review-env"
TEMPERATURE = 0.0
MAX_TOKENS = 800
SUCCESS_SCORE_THRESHOLD = 0.3

if not API_BASE_URL:
    print("[FATAL] API_BASE_URL is not set. The platform must inject this.", file=sys.stderr, flush=True)
    sys.exit(1)
if not API_KEY:
    print("[FATAL] API_KEY is not set. The platform must inject this.", file=sys.stderr, flush=True)
    sys.exit(1)

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = API_BASE_URL

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
    """Call the LLM using OpenAI Client with retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            print(
                f"[DEBUG] LLM call attempt {attempt+1}/{max_retries} model={MODEL_NAME}",
                file=sys.stderr, flush=True,
            )
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            result = (completion.choices[0].message.content or "").strip()
            print(f"[DEBUG] LLM response length={len(result)}", file=sys.stderr, flush=True)
            return result
        except Exception as exc:
            last_error = exc
            print(f"[DEBUG] Attempt {attempt+1} failed: {exc}", file=sys.stderr, flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    print(f"[ERROR] All {max_retries} attempts failed: {last_error}", file=sys.stderr, flush=True)
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


# ─── System Prompt ───────────────────────────────────────────────────────────

REVIEW_SYSTEM_PROMPT = """\
You are a senior software engineer performing code review.
You will receive a code snippet that may contain bugs.

Your task:
1. Identify any bugs in the code (off-by-one errors, null dereferences, wrong operators, dead variables, logic inversions)
2. Report the exact line numbers where bugs are
3. Suggest a concrete fix
4. Write a helpful review comment explaining the issues

Bug types to look for:
- Off-by-one: wrong boundary conditions (< vs <=), incorrect range bounds
- Null dereference: missing null/None/nil guards before access
- Wrong operator: arithmetic (+/-/*) or comparison (==/!=) errors
- Dead variables: unused assignments that shadow live variables
- Logic inversion: flipped boolean conditions (and/or, True/False, ==/!=)

Respond ONLY with valid JSON:
{
  "issues": ["description of bug 1", "description of bug 2"],
  "flagged_lines": [3, 7],
  "suggestion": "Change < to <= on line 3 to fix the boundary condition",
  "comment": "Found an off-by-one error in the loop boundary that causes the last element to be skipped."
}
"""


# ─── Observation Formatting ──────────────────────────────────────────────────

def format_observation(obs: CodeReviewObservation) -> str:
    """Format observation for the LLM."""
    # Add line numbers to code
    lines = obs.code.split('\n')
    numbered = '\n'.join(f"L{i+1}: {line}" for i, line in enumerate(lines))

    return (
        f"Language: {obs.language}\n"
        f"Difficulty: {obs.difficulty}\n\n"
        f"Code to review:\n```\n{numbered}\n```\n\n"
        f"{obs.instructions}"
    )


def action_to_str(action_dict: Dict) -> str:
    """Convert action dict to a compact string for logging."""
    at = action_dict.get("action_type", "submit_review")
    if at == "flag_line":
        return f"flag_line:{action_dict.get('line', '?')}"
    elif at == "analyze":
        return "analyze"
    elif at == "request_hint":
        return "request_hint"
    else:
        n_issues = len(action_dict.get("issues", []))
        n_lines = len(action_dict.get("flagged_lines", []))
        return f"submit_review:{n_issues}_issues,{n_lines}_lines"


# ─── Task Runner ─────────────────────────────────────────────────────────────

async def run_task(env: CodeReviewEnv, llm_client: OpenAI, difficulty: str) -> tuple:
    """Run a multi-step episode. Agent: analyze → flag lines → submit review."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01

    result = await _maybe_await(env.reset(seed=42, task=difficulty))
    obs = result.observation

    # Step 1: Analyze the code (free action)
    try:
        analyze_action = CodeReviewAction(action_type="analyze")
        result = await _maybe_await(env.step(analyze_action))
        obs = result.observation
        rewards.append(result.reward or 0.0)
        steps_taken += 1
        log_step(step=steps_taken, action="analyze", reward=result.reward or 0.0,
                 done=result.done, error=None)
        if result.done:
            score = max(0.01, min(0.99, sum(rewards) / len(rewards) if rewards else 0.01))
            return score, steps_taken, rewards
    except Exception as e:
        print(f"[DEBUG] analyze failed: {e}", file=sys.stderr, flush=True)

    # Step 2: Use LLM to identify bugs and flag lines
    user_prompt = format_observation(obs)
    if obs.analysis:
        user_prompt += f"\n\nAnalysis: {obs.analysis}"

    try:
        response = call_llm(llm_client, REVIEW_SYSTEM_PROMPT, user_prompt)
        parsed = parse_json_response(response)
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)
        parsed = None

    flagged = []
    if parsed:
        flagged = parsed.get("flagged_lines", [])
        flagged = [int(x) for x in flagged if isinstance(x, (int, float))]

    # Step 2-3: Flag individual lines (intermediate feedback)
    for line in flagged[:2]:  # Flag up to 2 lines
        try:
            flag_action = CodeReviewAction(action_type="flag_line", line=line)
            result = await _maybe_await(env.step(flag_action))
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken += 1
            log_step(step=steps_taken, action=f"flag_line:{line}", reward=reward,
                     done=result.done, error=None)
            if result.done:
                score = max(0.01, min(0.99, sum(r for r in rewards if r > 0) / max(1, len([r for r in rewards if r > 0])) if rewards else 0.01))
                return score, steps_taken, rewards
        except Exception as e:
            print(f"[DEBUG] flag_line failed: {e}", file=sys.stderr, flush=True)

    # Final step: Submit full review
    if parsed:
        action_dict = {
            "action_type": "submit_review",
            "issues": parsed.get("issues", []),
            "flagged_lines": parsed.get("flagged_lines", []),
            "suggestion": parsed.get("suggestion", ""),
            "comment": parsed.get("comment", ""),
        }
    else:
        action_dict = {
            "action_type": "submit_review",
            "issues": [],
            "flagged_lines": [],
            "suggestion": "",
            "comment": "Unable to analyze the code.",
        }

    if not isinstance(action_dict.get("flagged_lines"), list):
        action_dict["flagged_lines"] = []
    action_dict["flagged_lines"] = [
        int(x) for x in action_dict["flagged_lines"] if isinstance(x, (int, float))
    ]

    try:
        action = CodeReviewAction(**action_dict)
    except Exception as e:
        print(f"[DEBUG] Action validation failed: {e}", file=sys.stderr, flush=True)
        action = CodeReviewAction(action_type="submit_review")

    try:
        result = await _maybe_await(env.step(action))
        obs = result.observation
        reward = result.reward or 0.01
        done = result.done
        error = None
    except Exception as e:
        print(f"[DEBUG] env.step() failed: {e}", file=sys.stderr, flush=True)
        reward = 0.01
        done = True
        error = str(e)

    rewards.append(reward)
    steps_taken += 1

    log_step(
        step=steps_taken,
        action=action_to_str(action_dict),
        reward=reward,
        done=done,
        error=error,
    )

    # Score is the final submit_review reward (the main grading)
    score = max(0.01, min(0.99, reward))
    return score, steps_taken, rewards


# ─── Main ────────────────────────────────────────────────────────────────────

async def main() -> int:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connectivity check
    try:
        print("[DEBUG] Testing LiteLLM proxy...", file=sys.stderr, flush=True)
        test = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5, temperature=0.0,
        )
        print(f"[DEBUG] Proxy OK — response={test.choices[0].message.content!r}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[WARNING] Proxy test failed: {e}. Continuing.", file=sys.stderr, flush=True)

    scores = {}
    space_url = os.getenv("SPACE_URL", "https://ragavrida-code-review-env.hf.space")

    # Run all three difficulty tiers
    for difficulty in ["easy", "medium", "hard"]:
        task_name = f"code-review-{difficulty}"
        score = 0.01
        steps_taken = 0
        rewards: List[float] = []
        success = False
        env = None

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            if IMAGE_NAME:
                print(f"[DEBUG] Docker image: {IMAGE_NAME}", file=sys.stderr, flush=True)
                env = await CodeReviewEnv.from_docker_image(IMAGE_NAME)
            else:
                print(f"[DEBUG] Server: {space_url}", file=sys.stderr, flush=True)
                env = CodeReviewEnv(base_url=space_url)

            score, steps_taken, rewards = await run_task(env, llm_client, difficulty)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as e:
            print(f"[ERROR] Task {difficulty} failed: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)

        finally:
            score = min(max(score, 0.01), 0.99)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

            if env is not None:
                try:
                    close_result = env.close()
                    if inspect.isawaitable(close_result):
                        await close_result
                except Exception as e:
                    print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)

        scores[difficulty] = score

    composite = sum(scores.values()) / max(len(scores), 1)
    print(
        f"\n[SUMMARY] composite={composite:.3f} "
        f"easy={scores.get('easy',0):.3f} "
        f"medium={scores.get('medium',0):.3f} "
        f"hard={scores.get('hard',0):.3f}",
        file=sys.stderr, flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
