#!/usr/bin/env python3
"""
CodeReviewEnv — Real-Time Live Evaluation (WebSocket)
======================================================
Drives the LIVE HF Space endpoint using WebSocket sessions for stateful
interaction. Uses real OpenRouter / OpenAI-compatible LLM calls.

The WebSocket endpoint (/ws) creates a persistent environment session,
preserving state across reset() and step() calls within a single
connection. This is critical — the HTTP endpoints (/reset, /step) are
stateless and create fresh environments on every call.

Usage:
    OPENAI_API_KEY=sk-or-... python3 eval_live.py [--model MODEL] [--task all|easy|medium|hard]

Environment variables:
    OPENAI_API_KEY   OpenRouter or OpenAI key  (required)
    API_BASE_URL     LLM gateway base URL       (default: https://openrouter.ai/api/v1)
    MODEL_NAME       Model to use               (default: openai/gpt-4o-mini)
    ENV_BASE_URL     Live Space URL             (default: https://ragavrida-code-review-env.hf.space)
    SEED             Episode seed               (default: 42)
"""

import asyncio
import os
import re
import sys
import json
import time
import argparse
import statistics
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI
import websockets

# ─── Configuration ────────────────────────────────────────────────────────────

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://ragavrida-code-review-env.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
SEED         = int(os.getenv("SEED", "42"))
TEMPERATURE  = 0.0
MAX_TOKENS   = 512
DEBUG        = os.getenv("DEBUG", "false").lower() in ("true", "1")

TASKS = ["easy", "medium", "hard"]

# ─── WebSocket helpers ────────────────────────────────────────────────────────

def _ws_url() -> str:
    """Convert HTTP URL to WebSocket URL."""
    base = ENV_BASE_URL.rstrip("/")
    if base.startswith("https://"):
        return base.replace("https://", "wss://") + "/ws"
    elif base.startswith("http://"):
        return base.replace("http://", "ws://") + "/ws"
    return base + "/ws"


async def ws_episode(task: str, seed: int, step_fn) -> List[Dict]:
    """
    Run a full episode over a single WebSocket session.

    Opens a WS connection, sends reset, then repeatedly calls step_fn
    to get the next action and sends it. Returns the list of step results.

    Args:
        task: "easy" | "medium" | "hard"
        seed: episode seed
        step_fn: callable(obs_data: Dict, step: int) -> Dict (action dict)
                 Returns None to stop the episode.

    Returns:
        List of (obs, reward, done, action) dicts for each step.
    """
    ws_url = _ws_url()
    results = []

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=30,
            ping_timeout=60,
            close_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB max message
        ) as ws:
            # ── Reset ────────────────────────────────────────────────
            reset_msg = json.dumps({
                "type": "reset",
                "data": {"task": task, "seed": seed},
            })
            await ws.send(reset_msg)
            reset_resp = json.loads(await ws.recv())

            if reset_resp.get("type") == "error":
                print(f"  [ERROR] Reset failed: {reset_resp.get('data', {}).get('message', 'unknown')}")
                return results

            # Parse reset response
            resp_data = reset_resp.get("data", {})
            obs = resp_data.get("observation", resp_data)
            reward = resp_data.get("reward") or 0.0
            done = resp_data.get("done", False)

            # ── Step loop ────────────────────────────────────────────
            step = 0
            max_steps = {"easy": 5, "medium": 3, "hard": 21}.get(task, 10)

            while not done and step < max_steps:
                action = step_fn(obs, step)
                if action is None:
                    break

                step_msg = json.dumps({
                    "type": "step",
                    "data": action,
                })
                await ws.send(step_msg)
                step_resp = json.loads(await ws.recv())

                if step_resp.get("type") == "error":
                    print(f"  [ERROR] Step {step} failed: {step_resp.get('data', {}).get('message', 'unknown')}")
                    break

                resp_data = step_resp.get("data", {})
                obs = resp_data.get("observation", resp_data)
                reward = resp_data.get("reward") or 0.0
                done = resp_data.get("done", False)

                results.append({
                    "step": step,
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "action": action,
                })

                step += 1

            # ── Close ────────────────────────────────────────────────
            try:
                close_msg = json.dumps({"type": "close"})
                await ws.send(close_msg)
            except Exception:
                pass

    except Exception as exc:
        print(f"  [ERROR] WebSocket error: {exc}")

    return results


# ─── HTTP health check (still use HTTP for simple health) ─────────────────────

def env_health() -> bool:
    """Ping /health and return True if up."""
    try:
        r = requests.get(f"{ENV_BASE_URL.rstrip('/')}/health", timeout=10)
        return r.status_code == 200 and r.json().get("status") == "healthy"
    except Exception as exc:
        print(f"  [ERROR] Health check failed: {exc}")
        return False


# ─── LLM helpers ─────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system: str, user: str) -> str:
    """Call the configured model and return raw text (empty string on error)."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")
        return ""


def parse_json(text: str) -> Optional[Dict]:
    """Robustly extract JSON from a response that may contain markdown."""
    text = text.strip()
    # Strip ``` fences
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find outermost {...}
    for pattern in [r'\{.*\}', r'\{[^{}]*\}']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ─── System prompts ───────────────────────────────────────────────────────────

PROMPTS = {
    "easy": """\
You are a senior software engineer performing code review.
Assess the severity of bugs in the pull request shown.

Severity scale:
- "critical": Security vulnerabilities (SQL injection, auth bypass, hardcoded secrets)
- "high":     Crashes or data corruption (null pointer, race condition)
- "medium":   Logic errors or missing error handling
- "low":      Performance issues (N+1 queries, unnecessary loops)
- "none":     Style-only, no bugs

Respond ONLY with valid JSON — no prose, no markdown.
Format: {"action_type": "label_severity", "severity": "<critical|high|medium|low|none>"}""",

    "medium": """\
You are a senior software engineer managing a code review queue.
Order the given PRs by review priority (most urgent first).

Priority rules:
1. Security PRs (SQL injection, auth) top priority
2. Higher severity bugs before lower severity
3. Junior-author PRs need earlier review
4. PRs without tests should be reviewed sooner

Respond ONLY with valid JSON — no prose, no markdown.
Format: {"action_type": "prioritize", "priority_order": ["PR-XXX", "PR-YYY", ...]}""",

    "hard": """\
You are a senior software engineer performing detailed code review.
You must:
1. Add specific, actionable comments targeting buggy lines
2. Then approve or request changes

For each comment:
{"action_type": "add_comment", "comment": "<specific fix suggestion>",
 "target_file": "<filename>", "target_line": <line_number>}

When done:
{"action_type": "request_changes"}  — if there are bugs
{"action_type": "approve"}          — if clean

Use domain-specific keywords (null check, parameterized query, mutex lock etc.).
Respond ONLY with valid JSON — no prose, no markdown.""",
}


# ─── Observation formatters ───────────────────────────────────────────────────

def fmt_easy(obs: Dict) -> str:
    files_text = ""
    for f in obs.get("files", []):
        files_text += (
            f"\n--- {f.get('filename','?')} "
            f"({f.get('language','?')}, {f.get('lines_changed','?')} lines, "
            f"{'has tests' if f.get('has_tests') else 'no tests'}) ---\n"
            f"{f.get('diff','')}\n"
        )
    return (
        f"PR: {obs.get('pr_id','?')}\n"
        f"Title: {obs.get('title','?')}\n"
        f"Description: {obs.get('description','?')}\n"
        f"Author: {obs.get('author_experience','?')}\n"
        f"{files_text}\n"
        f"Step {obs.get('step_number',0)+1} — What is the bug severity?"
    )


def fmt_medium(obs: Dict) -> str:
    queue = obs.get("review_queue", [])
    pr_lines = ""
    for pr_id in queue:
        pr_lines += f"\n  - {pr_id}"
    return (
        f"Current PR: {obs.get('pr_id','?')} | Queue: {queue}\n"
        f"Title: {obs.get('title','?')} | Author: {obs.get('author_experience','?')}\n"
        f"\nPRs to prioritize (most urgent first):{pr_lines or ' (none listed)'}\n"
        f"\nOrder ALL {len(queue)} PR IDs by review urgency."
    )


def fmt_hard(obs: Dict) -> str:
    files_text = ""
    for f in obs.get("files", []):
        files_text += (
            f"\n--- {f.get('filename','?')} ---\n"
            f"{f.get('diff','')}\n"
        )
    prev = obs.get("existing_comments", [])
    prev_text = ""
    if prev:
        prev_text = "\nYour previous comments:\n" + "\n".join(f"  - {c}" for c in prev)
    return (
        f"PR: {obs.get('pr_id','?')}\n"
        f"Title: {obs.get('title','?')}\n"
        f"Description: {obs.get('description','?')}\n"
        f"Author: {obs.get('author_experience','?')}\n"
        f"{files_text}{prev_text}\n"
        "Review this code. Add comments for bugs. Then approve or request_changes."
    )


# ─── Task runners ─────────────────────────────────────────────────────────────

def run_easy(client: OpenAI, seed: int) -> Tuple[float, List[float], List[Dict]]:
    print(f"\n  [EASY] Running episode with seed={seed} via WebSocket...")

    step_rewards, log = [], []

    def step_fn(obs: Dict, step: int) -> Optional[Dict]:
        prompt  = fmt_easy(obs)
        raw     = call_llm(client, PROMPTS["easy"], prompt)
        parsed  = parse_json(raw)

        if parsed and parsed.get("severity"):
            action = {"action_type": "label_severity", "severity": parsed["severity"]}
        else:
            action = {"action_type": "label_severity", "severity": "medium"}

        return action

    results = asyncio.run(ws_episode("easy", seed, step_fn))

    for r in results:
        reward = r["reward"]
        step = r["step"]
        obs = r["obs"]
        action = r["action"]

        # Extract truth from the observation's info field
        info = obs.get("info") or {}
        truth = (info.get("ground_truth") or {}).get("severity", "?")
        predicted = action.get("severity", "?")

        step_rewards.append(reward)
        log.append({
            "step": step + 1,
            "predicted": predicted,
            "truth": truth,
            "reward": round(reward, 4),
        })

        if DEBUG:
            row = log[-1]
            print(f"    step={row['step']} pred={row['predicted']:8s} "
                  f"truth={row['truth']:8s} r={row['reward']:.3f}")

    mean = statistics.mean(step_rewards) if step_rewards else 0.0
    return mean, step_rewards, log


def run_medium(client: OpenAI, seed: int) -> Tuple[float, List[float], List[Dict]]:
    print(f"\n  [MEDIUM] Running episode with seed={seed} via WebSocket...")

    step_rewards, log = [], []

    def step_fn(obs: Dict, step: int) -> Optional[Dict]:
        queue = obs.get("review_queue", [])
        if not queue:
            queue = [obs.get("pr_id", "PR-001")]

        prompt = fmt_medium(obs)
        raw    = call_llm(client, PROMPTS["medium"], prompt)
        parsed = parse_json(raw)

        if parsed and parsed.get("priority_order"):
            order = parsed["priority_order"]
            # Fill any missing IDs at the end
            for qid in queue:
                if qid not in order:
                    order.append(qid)
            order = [q for q in order if q in queue] or queue
            action = {"action_type": "prioritize", "priority_order": order}
        else:
            action = {"action_type": "prioritize", "priority_order": queue}

        return action

    results = asyncio.run(ws_episode("medium", seed, step_fn))

    for r in results:
        reward = r["reward"]
        step = r["step"]
        obs = r["obs"]
        action = r["action"]

        info = obs.get("info") or {}
        truth_order = (
            info.get("ground_truth_order")
            or (info.get("ground_truth") or {}).get("priority_order")
            or []
        )

        step_rewards.append(reward)
        log.append({
            "step": step + 1,
            "predicted_order": action.get("priority_order", []),
            "truth_order": truth_order,
            "kendall_tau": info.get("kendall_tau", "?"),
            "reward": round(reward, 4),
        })

        if DEBUG:
            row = log[-1]
            print(f"    step={row['step']} tau={row['kendall_tau']} pred={row['predicted_order']} "
                  f"truth={row['truth_order']} r={row['reward']:.3f}")

    mean = statistics.mean(step_rewards) if step_rewards else 0.0
    return mean, step_rewards, log


def run_hard(client: OpenAI, seed: int) -> Tuple[float, List[float], List[Dict]]:
    print(f"\n  [HARD] Running episode with seed={seed} via WebSocket...")

    step_rewards, log = [], []
    comments_sent: Dict[str, int] = {}
    pr_finalized: set = set()
    MAX_COMMENTS_PER_PR = 3

    def step_fn(obs: Dict, step: int) -> Optional[Dict]:
        current_pr = obs.get("pr_id", "?")

        # If this PR is already finalized, the server stuck — stop
        if current_pr in pr_finalized:
            if DEBUG:
                print(f"    [WARN] Already finalized {current_pr}, server stuck — stopping")
            return None

        comments_sent.setdefault(current_pr, 0)
        n_comments = comments_sent[current_pr]

        prompt = fmt_hard(obs)
        raw    = call_llm(client, PROMPTS["hard"], prompt)
        parsed = parse_json(raw)

        # Build action — force decision if we've already sent enough comments
        if n_comments >= MAX_COMMENTS_PER_PR:
            action = {"action_type": "request_changes"}
        elif parsed:
            atype = parsed.get("action_type", "")
            if atype == "add_comment":
                action = {
                    "action_type": "add_comment",
                    "comment":     parsed.get("comment", "Consider fixing this issue."),
                    "target_file": parsed.get("target_file", "unknown"),
                    "target_line": int(parsed.get("target_line") or 1),
                }
            elif atype in ("approve", "request_changes"):
                action = {"action_type": atype}
            else:
                action = {"action_type": "request_changes"}
        else:
            action = {"action_type": "request_changes"}

        # Track comment count
        if action["action_type"] == "add_comment":
            comments_sent[current_pr] += 1
        else:
            pr_finalized.add(current_pr)

        return action

    results = asyncio.run(ws_episode("hard", seed, step_fn))

    for r in results:
        reward = r["reward"]
        step = r["step"]
        action = r["action"]
        obs = r["obs"]
        current_pr = obs.get("pr_id", "?")

        step_rewards.append(reward)
        log.append({
            "step":   step + 1,
            "pr":     current_pr,
            "action": action.get("action_type", "?"),
            "comments_sent": comments_sent.get(current_pr, 0),
            "reward": round(reward, 4),
        })

        if DEBUG:
            row = log[-1]
            print(f"    step={row['step']} pr={row['pr']} "
                  f"action={row['action']:20s} comments={row['comments_sent']} r={row['reward']:.3f}")

    # Score on PR-level rewards only (skip comment ack 0.05s)
    pr_rewards = [r for r in step_rewards if abs(r - 0.05) > 0.01]
    mean = statistics.mean(pr_rewards) if pr_rewards else 0.0
    return mean, step_rewards, log


# ─── Report ───────────────────────────────────────────────────────────────────

def print_step_table(task: str, logs: List[Dict]):
    """Print a human-readable per-step breakdown."""
    print(f"\n  {'Step':<6}", end="")
    if task == "easy":
        print(f"{'Predicted':>10} {'Truth':>10} {'Reward':>8}")
        print("  " + "-" * 36)
        for row in logs:
            match = "✓" if row.get("predicted") == row.get("truth") else "✗"
            print(f"  {row['step']:<6} {row.get('predicted','?'):>10} "
                  f"{row.get('truth','?'):>10} {row.get('reward',0):>8.3f}  {match}")
    elif task == "medium":
        print(f"{'Tau':>6} {'Reward':>8}")
        print("  " + "-" * 26)
        for row in logs:
            tau = row.get('kendall_tau', '?')
            tau_str = f"{tau:.3f}" if isinstance(tau, float) else str(tau)
            print(f"  {row['step']:<6} {tau_str:>6} {row.get('reward',0):>8.3f}")
            # Print the orderings on the next line
            pred = row.get('predicted_order', [])
            truth = row.get('truth_order', [])
            if truth:
                print(f"         pred : {pred}")
                print(f"         truth: {truth}")
    else:
        print(f"{'PR':>8} {'Comments':>9} {'Action':>20} {'Reward':>8}")
        print("  " + "-" * 52)
        for row in logs:
            print(f"  {row['step']:<6} {row.get('pr','?'):>8} "
                  f"{row.get('comments_sent',0):>9} "
                  f"{row.get('action','?'):>20} {row.get('reward',0):>8.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global MODEL_NAME, SEED, DEBUG  # allow args to override module-level defaults

    parser = argparse.ArgumentParser(description="CodeReviewEnv live evaluation")
    parser.add_argument("--task",   default="all",  choices=["all", "easy", "medium", "hard"])
    parser.add_argument("--model",  default=MODEL_NAME)
    parser.add_argument("--seed",   default=SEED,   type=int)
    parser.add_argument("--eps",    default=3,      type=int, help="Episodes per task")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    MODEL_NAME = args.model
    SEED       = args.seed
    DEBUG      = args.debug or DEBUG

    tasks = TASKS if args.task == "all" else [args.task]

    # ── Banner ────────────────────────────────────────────────────────
    print("=" * 64)
    print("  CodeReviewEnv — Live Evaluation (WebSocket + Real LLM)")
    print("=" * 64)
    print(f"  Space URL : {ENV_BASE_URL}")
    print(f"  WS URL    : {_ws_url()}")
    print(f"  LLM API   : {API_BASE_URL}")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  Tasks     : {', '.join(tasks)}")
    print(f"  Episodes  : {args.eps} per task  |  Seed: {SEED}")
    print("=" * 64)

    # ── Guard-rails ───────────────────────────────────────────────────
    if not API_KEY:
        print("\n[ERROR] No API key — set OPENAI_API_KEY (or HF_TOKEN / API_KEY)")
        sys.exit(1)

    print("\n[1/3] Checking live Space health...")
    if not env_health():
        print("  [FAIL] Space is not healthy. Check https://hf.co/spaces/ragavrida/code-review-env")
        sys.exit(1)
    print("  [OK] Space is healthy ✓")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results: Dict[str, Any] = {}
    start   = time.time()

    runners = {"easy": run_easy, "medium": run_medium, "hard": run_hard}

    for task in tasks:
        label = task.upper()
        runner = runners[task]
        print(f"\n{'─'*64}")
        print(f"  Task: {label}")
        print(f"{'─'*64}")

        scores, all_logs = [], []
        for ep in range(args.eps):
            ep_seed = SEED + ep
            score, steps, log = runner(client, ep_seed)
            scores.append(score)
            all_logs.append(log)
            print(f"\n  Episode {ep+1} (seed={ep_seed}) → mean reward = {score:.4f}")
            print_step_table(task, log)

        mean = statistics.mean(scores) if scores else 0.0
        std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"\n  ── {label} Summary: {mean:.4f} ± {std:.4f} ──")
        results[task] = {"mean": mean, "std": std, "scores": scores, "logs": all_logs}

    # ── Final summary ─────────────────────────────────────────────────
    elapsed   = time.time() - start
    all_means = [results[t]["mean"] for t in tasks]
    composite = statistics.mean(all_means) if all_means else 0.0

    print(f"\n{'='*64}")
    print("  FINAL RESULTS")
    print(f"{'='*64}")
    print(f"  {'Task':<12} {'Score':>8} {'Std':>8}")
    print("  " + "-" * 30)
    for task in tasks:
        r = results[task]
        print(f"  {task.capitalize():<12} {r['mean']:>8.4f} {r['std']:>8.4f}")
    if len(tasks) > 1:
        print("  " + "-" * 30)
        print(f"  {'Composite':<12} {composite:>8.4f}")
    print(f"\n  Model: {MODEL_NAME}  |  Time: {elapsed:.1f}s")
    print(f"{'='*64}")

    # ── Save results ──────────────────────────────────────────────────
    out = {
        "model":     MODEL_NAME,
        "space_url": ENV_BASE_URL,
        "seed":      SEED,
        "episodes":  args.eps,
        "tasks":     {t: {"mean": results[t]["mean"], "std": results[t]["std"],
                           "scores": results[t]["scores"]} for t in tasks},
        "composite": composite,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "baseline", "live_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
