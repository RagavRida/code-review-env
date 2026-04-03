#!/usr/bin/env python3
"""
CodeReviewEnv — Mandatory Inference Script
============================================
MANDATORY ENV VARS:
    API_BASE_URL    The API endpoint for the LLM.
    MODEL_NAME      The model identifier to use for inference.
    HF_TOKEN        Your Hugging Face / API key.

Uses OpenAI Client for all LLM calls.
Emits structured stdout logs: [START], [STEP], [END]  — per task.

Log format (strict, matches OpenEnv spec exactly):
    [START] {"task_id": "...", "task_description": "..."}
    [STEP]  {"step": N, "action": "...", "observation": "...", "reward": 0.0, "done": false}
    [END]   {"task_id": "...", "total_reward": 0.0, "steps": N, "success": false}
"""

import os
import re
import sys
import json
import time
import statistics
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 300


# ─── Structured Logging (spec-compliant) ────────────────────────────────────

def log_start(task_id: str, task_description: str):
    """Emit [START] structured log — one per task."""
    entry = {"task_id": task_id, "task_description": task_description}
    print(f"[START] {json.dumps(entry)}", flush=True)


def log_step(step: int, action: str, observation: str, reward: float, done: bool):
    """Emit [STEP] structured log — one per environment step."""
    entry = {"step": step, "action": action, "observation": observation, "reward": reward, "done": done}
    print(f"[STEP] {json.dumps(entry)}", flush=True)


def log_end(task_id: str, total_reward: float, steps: int, success: bool):
    """Emit [END] structured log — one per task."""
    entry = {"task_id": task_id, "total_reward": total_reward, "steps": steps, "success": success}
    print(f"[END] {json.dumps(entry)}", flush=True)


# ─── LLM Interface ──────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call the LLM using OpenAI Client. Returns response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
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
    for pattern in [r'\{[^{}]*\}', r'\{.*\}']:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ─── System Prompts ──────────────────────────────────────────────────────────

EASY_SYSTEM_PROMPT = """You are a senior software engineer performing code review.
You will receive a pull request with a code diff. Your job is to assess the severity of any bugs present.

Severity scale:
- "critical": Security vulnerabilities (SQL injection, auth bypass, hardcoded secrets, etc.)
- "high": Crashes or data corruption (null pointer dereference, race conditions, etc.)
- "medium": Logic errors or missing error handling (off-by-one, uncaught exceptions, etc.)
- "low": Performance issues (N+1 queries, unnecessary loops, etc.)
- "none": Style-only changes, no bugs

Respond ONLY with valid JSON. No explanation, no markdown, no code blocks.
Format: {"action_type": "label_severity", "severity": "<critical|high|medium|low|none>"}"""

MEDIUM_SYSTEM_PROMPT = """You are a senior software engineer managing a code review queue.
You will receive a list of pull requests. Order them by review priority (most urgent first).

Priority rules:
1. Security-related PRs (SQL injection, auth issues) are always highest priority
2. Higher severity bugs should be reviewed before lower severity ones
3. PRs from junior developers need more urgent review than senior ones
4. PRs without tests should be reviewed earlier

Respond ONLY with valid JSON. No explanation, no markdown, no code blocks.
Format: {"action_type": "prioritize", "priority_order": ["PR-XXX", "PR-YYY", ...]}"""

HARD_SYSTEM_PROMPT = """You are a senior software engineer performing detailed code review.
You will see a pull request with a code diff. You must:
1. Add specific, actionable review comments targeting buggy lines
2. Then approve or request changes

For comments, respond with JSON:
{"action_type": "add_comment", "comment": "<specific actionable feedback>", "target_file": "<filename>", "target_line": <line_number>}

When done reviewing, respond with:
{"action_type": "request_changes"} if there are bugs, or {"action_type": "approve"} if clean.

Use domain-specific keywords in comments (e.g. "null check", "parameterized query", "mutex lock").
Be specific about the bug and suggest a concrete fix.

Respond ONLY with valid JSON. No explanation, no markdown, no code blocks."""


# ─── Observation Formatting ──────────────────────────────────────────────────

def format_observation_easy(obs) -> str:
    files_text = ""
    for f in obs.files:
        files_text += f"\n--- {f.filename} ({f.language}, {f.lines_changed} lines changed"
        files_text += f", {'has' if f.has_tests else 'no'} tests) ---\n"
        files_text += f.diff + "\n"
    return (
        f"Pull Request: {obs.pr_id}\nTitle: {obs.title}\n"
        f"Description: {obs.description}\nAuthor experience: {obs.author_experience}\n"
        f"{files_text}\n"
        f"Step {obs.step_number + 1} of {obs.episode_budget + obs.step_number}. "
        f"What is the severity of any bugs in this PR?"
    )


def format_observation_medium(obs, queue_templates: List[Dict]) -> str:
    queue_text = ""
    for t in queue_templates:
        has_tests = "has tests" if t.get("has_tests") else "no tests"
        bug = t.get("bug_category", "unknown")
        queue_text += f"\n- {t['pr_id']}: \"{t['title']}\" (author: {t['author_experience']}, "
        queue_text += f"category: {bug}, {has_tests}, {t.get('lines_changed', '?')} lines changed)"
        diff_preview = t.get("diff", "")[:200]
        if diff_preview:
            queue_text += f"\n  Diff preview: {diff_preview.strip()[:150]}..."
    return (
        f"Review Queue — Step {obs.step_number + 1} of {obs.episode_budget + obs.step_number}\n\n"
        f"You have {len(queue_templates)} PRs to prioritize:{queue_text}\n\n"
        f"Order these PRs by review priority (most urgent first). Return ALL PR IDs."
    )


def format_observation_hard(obs) -> str:
    files_text = ""
    for f in obs.files:
        files_text += f"\n--- {f.filename} ({f.language}, {f.lines_changed} lines changed) ---\n"
        files_text += f.diff + "\n"
    comments_text = ""
    if obs.existing_comments:
        comments_text = "\nYour previous comments on this PR:\n"
        for c in obs.existing_comments:
            comments_text += f"  - {c}\n"
    return (
        f"Pull Request: {obs.pr_id}\nTitle: {obs.title}\n"
        f"Description: {obs.description}\nAuthor experience: {obs.author_experience}\n"
        f"Remaining PRs in queue: {', '.join(obs.review_queue) if obs.review_queue else 'none'}\n"
        f"{files_text}{comments_text}\n"
        f"Review this code. If you see bugs, add a specific comment targeting the buggy line.\n"
        f"If you've already commented on the main issues, use \"request_changes\" (if bugs) or \"approve\" (if clean)."
    )


def obs_summary(obs) -> str:
    """Create a short observation summary for the log."""
    return f"pr_id={obs.pr_id}, title={obs.title[:60]}, step={obs.step_number}"


# ─── Task Runners ────────────────────────────────────────────────────────────

def run_easy(client: OpenAI, seed: int) -> Tuple[float, List[Dict]]:
    """Run easy task episode. Returns (mean_reward, step_logs)."""
    from env.base import CodeReviewEnv
    from env.models import Action

    env = CodeReviewEnv(task="easy", seed=seed)
    obs = env.reset()

    log_start(
        task_id="easy",
        task_description="Classify PR severity (none/low/medium/high/critical)",
    )

    step_rewards = []
    total_steps = 0

    for step in range(5):
        prompt = format_observation_easy(obs)
        response = call_llm(client, EASY_SYSTEM_PROMPT, prompt)
        parsed = parse_json_response(response)

        if parsed and parsed.get("severity"):
            action = Action(action_type="label_severity", severity=parsed["severity"])
        else:
            action = Action(action_type="label_severity", severity="medium")

        obs, reward, done, info = env.step(action)
        total_steps = step + 1

        action_str = json.dumps({"action_type": action.action_type, "severity": action.severity})
        obs_str = obs_summary(obs)

        log_step(
            step=step + 1,
            action=action_str,
            observation=obs_str,
            reward=round(reward.value, 4),
            done=done,
        )
        step_rewards.append(reward.value)

        if done:
            break

    mean = statistics.mean(step_rewards) if step_rewards else 0.0
    total = sum(step_rewards)
    success = mean >= 0.6  # threshold: better than random

    log_end(
        task_id="easy",
        total_reward=round(total, 4),
        steps=total_steps,
        success=success,
    )

    return mean, step_rewards


def run_medium(client: OpenAI, seed: int) -> Tuple[float, List[Dict]]:
    """Run medium task episode. Returns (mean_reward, step_logs)."""
    from env.base import CodeReviewEnv
    from env.models import Action
    from tasks.task_medium import MediumTask

    env = CodeReviewEnv(task="medium", seed=seed)
    obs = env.reset()

    log_start(
        task_id="medium",
        task_description="Prioritize review queue by urgency",
    )

    step_rewards = []
    total_steps = 0

    for step in range(3):
        queue_templates = env.task.get_queue_templates(step)
        prompt = format_observation_medium(obs, queue_templates)
        response = call_llm(client, MEDIUM_SYSTEM_PROMPT, prompt)
        parsed = parse_json_response(response)

        queue_ids = [t["pr_id"] for t in queue_templates]
        if parsed and parsed.get("priority_order"):
            order = parsed["priority_order"]
            for qid in queue_ids:
                if qid not in order:
                    order.append(qid)
            order = [qid for qid in order if qid in queue_ids] or queue_ids
            action = Action(action_type="prioritize", priority_order=order)
        else:
            action = Action(action_type="prioritize", priority_order=queue_ids)

        obs, reward, done, info = env.step(action)
        total_steps = step + 1

        action_str = json.dumps({"action_type": "prioritize", "priority_order": action.priority_order})
        obs_str = obs_summary(obs)

        log_step(
            step=step + 1,
            action=action_str,
            observation=obs_str,
            reward=round(reward.value, 4),
            done=done,
        )
        step_rewards.append(reward.value)

        if done:
            break

    mean = statistics.mean(step_rewards) if step_rewards else 0.0
    total = sum(step_rewards)
    success = mean >= 0.5

    log_end(
        task_id="medium",
        total_reward=round(total, 4),
        steps=total_steps,
        success=success,
    )

    return mean, step_rewards


def run_hard(client: OpenAI, seed: int) -> Tuple[float, List[Dict]]:
    """Run hard task episode. Returns (mean_reward, step_logs)."""
    from env.base import CodeReviewEnv
    from env.models import Action

    env = CodeReviewEnv(task="hard", seed=seed)
    obs = env.reset()

    log_start(
        task_id="hard",
        task_description="Generate actionable review feedback for 3 PRs",
    )

    step_rewards = []
    total_steps = 0
    max_steps = 18  # 3 PRs × 6 actions max

    for step in range(max_steps):
        prompt = format_observation_hard(obs)
        response = call_llm(client, HARD_SYSTEM_PROMPT, prompt)
        parsed = parse_json_response(response)

        if parsed:
            action_type = parsed.get("action_type", "")
            if action_type == "add_comment":
                target_line = parsed.get("target_line", 1)
                if not isinstance(target_line, int):
                    try:
                        target_line = int(target_line)
                    except (ValueError, TypeError):
                        target_line = 1
                action = Action(
                    action_type="add_comment",
                    comment=parsed.get("comment", "Consider fixing this issue."),
                    target_file=parsed.get("target_file", "unknown.py"),
                    target_line=target_line,
                )
            elif action_type in ("approve", "request_changes"):
                action = Action(action_type=action_type)
            else:
                action = Action(action_type="request_changes")
        else:
            action = Action(action_type="request_changes")

        obs, reward, done, info = env.step(action)
        total_steps = step + 1

        action_dict = {"action_type": action.action_type}
        if action.action_type == "add_comment":
            action_dict["comment"] = (action.comment or "")[:100]
            action_dict["target_file"] = action.target_file
            action_dict["target_line"] = action.target_line
        action_str = json.dumps(action_dict)
        obs_str = obs_summary(obs)

        log_step(
            step=step + 1,
            action=action_str,
            observation=obs_str,
            reward=round(reward.value, 4),
            done=done,
        )
        step_rewards.append(reward.value)

        if done:
            break

    # PR-level scoring: filter out comment acks (0.05)
    pr_rewards = [r for r in step_rewards if abs(r - 0.05) > 0.01]
    mean = statistics.mean(pr_rewards) if pr_rewards else 0.0
    total = sum(step_rewards)
    success = mean >= 0.3

    log_end(
        task_id="hard",
        total_reward=round(total, 4),
        steps=total_steps,
        success=success,
    )

    return mean, step_rewards


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not HF_TOKEN:
        print("ERROR: No API key. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    start_time = time.time()
    all_results = {}

    # ── Easy Task ─────────────────────────────────────────────────────
    easy_scores = []
    for ep in range(3):
        ep_seed = SEED + ep
        score, logs = run_easy(client, ep_seed)
        easy_scores.append(score)
    easy_mean = statistics.mean(easy_scores)
    easy_std = statistics.stdev(easy_scores) if len(easy_scores) > 1 else 0.0
    all_results["easy"] = {"mean": round(easy_mean, 4), "std": round(easy_std, 4), "scores": [round(s, 4) for s in easy_scores]}

    # ── Medium Task ───────────────────────────────────────────────────
    medium_scores = []
    for ep in range(3):
        ep_seed = SEED + ep
        score, logs = run_medium(client, ep_seed)
        medium_scores.append(score)
    medium_mean = statistics.mean(medium_scores)
    medium_std = statistics.stdev(medium_scores) if len(medium_scores) > 1 else 0.0
    all_results["medium"] = {"mean": round(medium_mean, 4), "std": round(medium_std, 4), "scores": [round(s, 4) for s in medium_scores]}

    # ── Hard Task ─────────────────────────────────────────────────────
    hard_scores = []
    for ep in range(3):
        ep_seed = SEED + ep
        score, logs = run_hard(client, ep_seed)
        hard_scores.append(score)
    hard_mean = statistics.mean(hard_scores)
    hard_std = statistics.stdev(hard_scores) if len(hard_scores) > 1 else 0.0
    all_results["hard"] = {"mean": round(hard_mean, 4), "std": round(hard_std, 4), "scores": [round(s, 4) for s in hard_scores]}

    # ── Save results ──────────────────────────────────────────────────
    elapsed = time.time() - start_time
    composite = round((easy_mean + medium_mean + hard_mean) / 3, 4)

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline", "results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    out = {
        "model": MODEL_NAME,
        "composite": composite,
        "seed": SEED,
        **all_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(results_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}", flush=True)
    print(f"INFERENCE COMPLETE — {MODEL_NAME}", flush=True)
    print(f"  Composite: {composite:.4f}", flush=True)
    print(f"  Easy:   {easy_mean:.4f} ± {easy_std:.4f}", flush=True)
    print(f"  Medium: {medium_mean:.4f} ± {medium_std:.4f}", flush=True)
    print(f"  Hard:   {hard_mean:.4f} ± {hard_std:.4f}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
