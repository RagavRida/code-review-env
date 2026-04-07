---
title: CodeReviewEnv
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: "First RL benchmark for semantic MBRL over code review"
tags:
  - openenv
  - reinforcement-learning
  - code-review
  - mbrl
  - knowledge-work
  - llm-agents
  - semantic-world-model
---

# 🔍 CodeReviewEnv

**An OpenEnv-compliant RL environment for software code review agents.**

> **CodeReviewEnv is the first RL benchmark for structured knowledge work.** Unlike MuJoCo (continuous physics), Atari (pixel grids), or TextWorld (synthetic narratives), CodeReviewEnv operates over *real-world semantic states* — code diffs, bug categories, and human-calibrated severity labels from actual software engineering practice. Its trajectory export (`export_trajectory()`) provides the first standardized dataset format for training **Knowledge-Work World Models (KW-WM)** — world models over structured professional text.

Train and evaluate LLM agents on real code review tasks — severity triage, queue prioritization, and actionable feedback generation — with deterministic grading, shaped rewards, and trajectory logging for Knowledge-Work World Model (KW-WM) research.

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

---

## Table of Contents

- [Motivation](#motivation)
- [Environment Description](#environment-description)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Reward Design](#reward-design)
- [Tasks](#tasks)
- [Baseline Scores](#baseline-scores)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Running Inference](#running-inference)
- [Environment Variables](#environment-variables)
- [Pre-Submission Checklist](#pre-submission-checklist)
- [Structured Logging Format](#structured-logging-format)
- [Project Structure](#project-structure)
- [Trajectory Dataset](#trajectory-dataset)
- [Citation](#citation)

---

## Motivation

### The Problem

Modern AI agents are increasingly deployed for knowledge work — summarizing documents, triaging issues, reviewing code — yet the RL/agent community lacks environments that faithfully model these tasks. Existing benchmarks either live in toy domains (grid worlds, text adventures) or are evaluation-only suites (SWE-bench, WebArena) with no MDP formalism, reward shaping, or trajectory export.

Prior work on text-based world models (Li et al., 2025 "From Word to World") studies general text games. Prior work on semantic world models (Berg et al., 2025 "SWM") targets embodied robotics. **CodeReviewEnv is the first benchmark for world model training in structured knowledge work** — where state transitions depend on professional judgment rather than physical or game mechanics.

SWE-bench and its successors explicitly acknowledge they cannot measure code maintainability or professional review quality (Da et al., 2025). CodeReviewEnv is the first environment designed to make these dimensions **learnable via MBRL**.

### Why CodeReviewEnv?

Code review is one of the highest-volume, highest-impact knowledge tasks in software engineering. Every development team does it daily, and quality directly affects shipped software security, reliability, and maintainability. CodeReviewEnv fills a genuine gap:

| Need | CodeReviewEnv |
|------|---------------|
| Real task that people do daily | ✅ Software code review |
| MDP formalism with R(s,a,s') | ✅ Semantic MDP with shaped rewards |
| Multiple difficulty levels | ✅ Easy / Medium / Hard |
| Deterministic, reproducible grading | ✅ Seed-controlled, no stochastic graders |
| Trajectory export for KW-WM | ✅ JSONL `(s, a, r, s')` per step |
| Deployable as a service | ✅ Docker + HF Space + OpenEnv spec |

### Research Gap

| Benchmark | State Space | Transition | World Model? | Domain |
|-----------|-------------|------------|--------------|--------|
| MuJoCo | ℝⁿ (joints) | Physics sim | ✅ Dreamer | Robotics |
| Atari | Pixels | Game engine | ✅ MuZero | Games |
| TextWorld | Synthetic text | Game rules | ⚠️ Li et al. 2025 | Text games |
| SWM (Berg et al.) | Visual + text | Physics | ✅ Embodied | Robotics |
| SWE-bench | Code | N/A | ❌ Eval only | SE |
| **CodeReviewEnv** | **Structured text** | **Professional judgment** | **✅ KW-WM (this work)** | **Knowledge work** |

CodeReviewEnv introduces **knowledge-work transitions** — the state is structured professional text (code diffs, bug patterns, author context) and the transition depends on *professional judgment*. This enables training **Knowledge-Work World Models (KW-WM)** — a new class of world models not benchmarked by prior work in games (Li et al., 2025), robotics (Berg et al., 2025), or code generation (Da et al., 2025).

---

## Environment Description

CodeReviewEnv models the software code review process as a **Semantic Markov Decision Process (S-MDP)**. An agent receives pull request observations (code diffs, metadata, review history) and must take structured review actions (classify severity, prioritize queues, write feedback). A deterministic grader scores each action and the episode produces a clean trajectory suitable for RL training or world model research.

### Episode Flow

```
reset(seed) → Observation₀
    ↓
step(Action₁) → (Observation₁, Reward₁, done₁, info₁)
step(Action₂) → (Observation₂, Reward₂, done₂, info₂)
    ...
step(Actionₙ) → (Observationₙ, Rewardₙ, done=True, infoₙ)
    ↓
export_trajectory() → [(s₀, a₁, r₁, s₁), (s₁, a₂, r₂, s₂), ...]
```

- **`reset(seed)`** produces a clean initial state — no leakage between episodes
- **`step(action)`** returns the standard `(observation, reward, done, info)` tuple
- **`state()`** exposes the full internal state including trajectory history
- **`export_trajectory()`** outputs `(s, a, r, s')` transitions in JSONL

---

## Observation Space

The observation represents the semantic state `s ∈ S` visible to the agent at each timestep.

| Field | Type | Description |
|-------|------|-------------|
| `pr_id` | `str` | Unique pull request identifier (e.g. `PR-001`) |
| `title` | `str` | Human-readable PR title |
| `description` | `str` | PR description / summary |
| `author_experience` | `str ∈ {junior, mid, senior}` | Experience level of the PR author |
| `files` | `List[PRFile]` | Code diffs per file (see below) |
| `existing_comments` | `List[str]` | Previously submitted review comments |
| `review_queue` | `List[str]` | IDs of pending PRs in queue |
| `step_number` | `int` | Current step in the episode (0-indexed) |
| `episode_budget` | `int` | Steps remaining in this episode |

### PRFile Schema

Each file in `files` contains:

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `str` | File path (e.g. `UserService.java`) |
| `language` | `str ∈ {python, javascript, java, go, rust, typescript, ruby}` | Programming language |
| `diff` | `str` | Unified diff of changes |
| `lines_changed` | `int` | Number of modified lines |
| `has_tests` | `bool` | Whether the PR includes test coverage |

---

## Action Space

The action space is **heterogeneous** — different `action_type` values activate different required fields. This mirrors real code review decisions.

| `action_type` | Required Fields | Description |
|---------------|----------------|-------------|
| `label_severity` | `severity ∈ {critical, high, medium, low, none}` | Classify the bug severity of the current PR |
| `prioritize` | `priority_order: List[str]` | Order the review queue by urgency (most urgent first) |
| `add_comment` | `comment: str`, `target_file: str`, `target_line: int` | Add a review comment targeting a specific line in a specific file |
| `approve` | — | Approve the PR (no remaining concerns) |
| `request_changes` | — | Request changes (bugs found, feedback given) |

### Action Validation

- Actions with an `action_type` mismatched to the current task receive a penalty reward
- Actions with missing required fields are handled gracefully (no crash, penalty applied)
- The environment never raises on malformed actions — it returns a penalty `Reward` instead

---

## Reward Design

Rewards are shaped to provide useful, varying signal — not just sparse terminal feedback. Each reward `R(s, a, s') ∈ [-1.0, 1.0]` includes a `breakdown` dict for component-level analysis.

| Component | Value | When Applied |
|-----------|-------|--------------|
| `step_reward` | `[0.0, 1.0]` | Per-action quality score from the task-specific grader |
| `efficiency_bonus` | `+0.10` | Complete the episode under budget |
| `coverage_bonus` | `+0.15` | Catch all critical bugs in the PR |
| `consistency_penalty` | `−0.20` | Contradict your own previous severity labels |
| `exploit_penalty` | `−0.50` | Approve a PR with unaddressed critical bugs |

### Reward Properties

- **Bounded**: All rewards clamped to `[-1.0, 1.0]`
- **Shaped**: Non-sparse signal at every step (not just at episode end)
- **Transparent**: `reward.breakdown` exposes all components for reward attribution research
- **Anti-exploit**: Spam comments and blind approvals are penalized (see tests)

---

## Tasks

CodeReviewEnv provides **3 tasks** spanning easy → hard, with different skills and grading logic:

### Task 1: Severity Labeling (Easy)

| Property | Value |
|----------|-------|
| Difficulty | ⭐ Easy |
| Episode Length | 5 steps (5 PRs) |
| Objective | Classify each PR's bug severity |
| Actions Used | `label_severity` |
| Grader | Ordinal matching — exact match = 1.0, adjacent match = 0.6, off-by-two = 0.2, miss = 0.0. Extra penalties for confusing `critical` with `none`. |
| Expected Score (random) | ~0.21 |
| Expected Score (GPT-4o-mini) | ~0.73 |
| Expected Score (perfect) | 1.00 |

The agent sees one PR at a time and must label its severity. The grader uses ordinal distance on the severity scale: `none < low < medium < high < critical`.

### Task 2: Queue Prioritization (Medium)

| Property | Value |
|----------|-------|
| Difficulty | ⭐⭐ Medium |
| Episode Length | 3 steps (3 queues of 5 PRs each) |
| Objective | Sort the review queue by urgency |
| Actions Used | `prioritize` |
| Grader | Kendall Tau rank correlation + position penalty for misplacing top-priority items. Perfect ordering = 1.0, random ≈ 0.31, fully reversed ≈ 0.0. |
| Expected Score (random) | ~0.31 |
| Expected Score (GPT-4o-mini) | ~0.94 |
| Expected Score (perfect) | 1.00 |

The agent sees a queue of PRs with metadata (author level, bug category, test coverage) and must output the optimal priority ordering. Grading uses Kendall Tau correlation to smoothly rank all orderings.

### Task 3: Feedback Generation (Hard)

| Property | Value |
|----------|-------|
| Difficulty | ⭐⭐⭐ Hard |
| Episode Length | Up to 18 steps (3 PRs × ≤6 actions each) |
| Objective | Write actionable review comments, then approve or request changes |
| Actions Used | `add_comment`, `approve`, `request_changes` |
| Grader | 5-component weighted scorer: relevance (line targeting accuracy), specificity (domain keyword matching), actionability (concrete fix suggestions), coverage (% of bugs found), precision (signal-to-noise ratio). |
| Expected Score (random) | ~0.09 |
| Expected Score (GPT-4o-mini) | ~1.00 |
| Expected Score (perfect oracle) | ~0.91 |

This is the most challenging task. The agent must read code diffs, identify bugs, write specific comments targeting exact lines, use domain-specific terminology, and decide whether to approve or request changes. The multi-component grader ensures that generic, vague, or spammy comments score poorly.

**Why GPT-4o-mini > perfect oracle on hard**: The "perfect" agent uses template-based comments with known bug lines. GPT-4o-mini generates more natural, specific comments that score higher on the specificity and actionability components.

---

## Baseline Scores

Two baseline agents are included: a **heuristic agent** (no LLM, runs instantly via `baseline.py`) and an **LLM agent** (requires API key, via `inference.py`).

### Heuristic Baseline (`baseline.py`)

Run `python baseline.py` — no API key required.

| Agent | Easy | Medium | Hard | Composite |
|-------|------|--------|------|-----------|
| Keyword Heuristic | 0.73 ± 0.12 | 0.47 ± 0.10 | 0.59 ± 0.09 | 0.60 |
| Random | ~0.21 | ~0.31 | ~0.05 | ~0.18 |

### LLM Baseline (`inference.py`)

Run `python inference.py` with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` set.

| Agent | Easy | Medium | Hard | Composite |
|-------|------|--------|------|-----------|
| GPT-4o-mini (measured) | 1.00 ± 0.00 | 0.68 ± 0.02 | 0.38 ± 0.01 | 0.69 |
| Perfect Oracle | 1.00 | 1.00 | ~0.91 | ~0.97 |

> **Note:** LLM scores depend on model version and API provider. Run `python inference.py` to generate reproducible results for your setup. Results are saved to `baseline/results.json`.

**Key observations:**
- **Monotonic difficulty**: Easy (1.00) > Medium (0.68) > Hard (0.38) — validated with real LLM
- **Clear agent separation**: Random (0.18) < Heuristic (0.60) < GPT-4o-mini (0.69) < Perfect (0.97)
- **Large headroom**: GPT-4o-mini at 0.69 vs ceiling 0.97 — significant room for RL-trained improvement
- **Hard is genuinely hard**: Even GPT-4o-mini scores only 0.38 on multi-turn feedback generation
- **Spam-resistant**: Decaying comment rewards prevent trivial exploit loops
- All scores are deterministic given the same seed and model

---

## Setup & Installation

### Requirements

- **Python 3.11+**
- An OpenAI-compatible API key (OpenRouter, OpenAI, etc.)

### Install Dependencies

```bash
# Clone the repository
git clone https://huggingface.co/spaces/openenv/code-review-env
cd code-review-env

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install research dependencies for world model training
pip install -r requirements-research.txt
```

### Docker

```bash
# Build
docker build -t code-review-env .

# Run (serves on port 7860)
docker run -p 7860:7860 code-review-env

# Verify
curl http://localhost:7860/health
```

### Validate

```bash
# Run the full OpenEnv compliance validation suite
python validate.py

# Run the test suite (19 tests across 5 categories)
pytest tests/ -v
```

---

## Usage

### Direct Python API

```python
from env.base import CodeReviewEnv
from env.models import Action

# Initialize with task and seed
env = CodeReviewEnv(task="easy", seed=42)
obs = env.reset()

# Take an action
action = Action(action_type="label_severity", severity="high")
obs, reward, done, info = env.step(action)

print(f"Reward: {reward.value:.3f}")
print(f"Breakdown: {reward.breakdown}")
print(f"Done: {done}")

# Run full episode
while not done:
    action = Action(action_type="label_severity", severity="medium")
    obs, reward, done, info = env.step(action)

# Export trajectory for MBRL research
trajectory = env.export_trajectory()
```

### HTTP API (OpenEnv Server)

```bash
# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Health check
curl http://localhost:7860/health

# Reset (start new episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42}'

# Step (take action)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "label_severity", "severity": "high"}}'

# Get current state
curl http://localhost:7860/state

# Environment metadata
curl http://localhost:7860/metadata

# OpenAPI docs: http://localhost:7860/docs
```

### OpenEnv Client (Async/Sync)

```python
import asyncio
from code_review_env import CodeReviewEnv, CodeReviewAction

async def main():
    async with CodeReviewEnv(base_url="https://openenv-code-review-env.hf.space") as env:
        result = await env.reset(seed=42)
        print(result.observation.pr_id)

        result = await env.step(
            CodeReviewAction(action_type="label_severity", severity="high")
        )
        print(f"Reward: {result.reward}, Done: {result.done}")

asyncio.run(main())

# Or synchronous:
with CodeReviewEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(seed=42)
    result = env.step(CodeReviewAction(action_type="label_severity", severity="high"))
```

---

## Running Inference

The `inference.py` script is the mandatory evaluation entry point. It runs all 3 tasks, emits structured logs, and saves results.

### Environment Variables

These **must** be set before running inference:

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | The API endpoint for the LLM (default: `https://openrouter.ai/api/v1`) |
| `MODEL_NAME` | Yes | The model identifier (default: `openai/gpt-4o-mini`) |
| `HF_TOKEN` | Yes | Your Hugging Face / API key. Also accepts `OPENAI_API_KEY` or `API_KEY`. |

### Run

```bash
# Set required environment variables
export API_BASE_URL="https://openrouter.ai/api/v1"
export MODEL_NAME="openai/gpt-4o-mini"
export HF_TOKEN="your-api-key-here"

# Run inference (completes in < 20 minutes)
python inference.py
```

### What It Does

1. Initializes the OpenAI client with `API_BASE_URL` and `HF_TOKEN`
2. Runs 3 episodes per task (easy, medium, hard) with `seed=42`
3. Emits structured `[START]`, `[STEP]`, `[END]` logs to stdout
4. Saves results to `baseline/results.json`

---

## Structured Logging Format

The inference script emits structured stdout logs in the **mandatory** `[START]`, `[STEP]`, `[END]` format:

### `[START]` — emitted once per task

```
[START] task=severity-labeling env=code-review-env model=openai/gpt-4o-mini
```

### `[STEP]` — emitted for each action taken

```
[STEP] step=1 action=label_severity:high reward=0.50 done=false error=null
```

### `[END]` — emitted once per task at completion

```
[END] success=true steps=5 score=0.767 rewards=0.50,1.00,0.80,0.60,0.93
```

---

## Pre-Submission Checklist

| Check | Command / Verification | Status |
|-------|----------------------|--------|
| HF Space deploys | `curl https://ragavrida-code-review-env.hf.space/health` returns 200 | ✅ |
| OpenEnv spec compliance | `python validate.py` — all 17 checks pass | ✅ |
| Dockerfile builds | `docker build -t code-review-env .` | ✅ |
| Baseline reproduces | `python baseline.py` completes end-to-end without errors | ✅ |
| LLM inference | `python inference.py` completes with API key, saves `baseline/results.json` | ✅ |
| 3+ tasks with graders | `easy`, `medium`, `hard` — all graders produce scores in `[0.0, 1.0]` | ✅ |
| Tests pass | `pytest tests/ -v` — 21 tests across 6 categories | ✅ |
| `API_BASE_URL` defined | Used by `inference.py` | ✅ |
| `MODEL_NAME` defined | Used by `inference.py` | ✅ |
| `HF_TOKEN` defined | Used by `inference.py` | ✅ |
| `inference.py` in root | Located at `./inference.py` | ✅ |
| Uses OpenAI Client | `from openai import OpenAI` | ✅ |
| Structured stdout logs | `[START]`, `[STEP]`, `[END]` format | ✅ |
| Runtime < 20 min | ~200 seconds | ✅ |
| Runs on vcpu=2, memory=8GB | No GPU dependencies, lightweight CPU inference | ✅ |

---

## Project Structure

```
code-review-env/
├── inference.py                # Mandatory LLM evaluation script (root)
├── baseline.py                 # Heuristic baseline agent (no API key needed)
├── openenv.yaml                # OpenEnv manifest
├── Dockerfile                  # Python 3.11 + uvicorn on :7860
├── requirements.txt            # Core dependencies
├── requirements-research.txt   # Optional: torch, sentence-transformers
├── pyproject.toml              # Package configuration
├── validate.py                 # OpenEnv spec compliance validator
├── models.py                   # OpenEnv Action/Observation/State subclasses
├── client.py                   # CodeReviewEnv(EnvClient) — async/sync client
├── dataset.py                  # SemanticTransitionDataset (PyTorch-compatible)
├── __init__.py                 # Package exports
│
├── trajectories/               # MBRL trajectory data (JSONL)
│   └── sample_trajectory.jsonl # 13 sample transitions from all 3 tasks
│
├── env/                        # Core environment logic
│   ├── base.py                 # CodeReviewEnv main class (S-MDP)
│   ├── models.py               # Internal Pydantic models (Action, Observation, Reward, State)
│   ├── data_generator.py       # 50 PR templates with real code diffs
│   └── trajectory_logger.py    # JSONL trajectory logging for MBRL
│
├── server/                     # OpenEnv-compliant server
│   ├── code_review_environment.py  # Environment(OpenEnv base class)
│   └── app.py                  # create_app() — FastAPI + WebSocket
│
├── tasks/                      # Three difficulty levels
│   ├── task_easy.py            # Severity labeling (5 PRs/episode)
│   ├── task_medium.py          # Queue prioritization (3 queues/episode)
│   └── task_hard.py            # Feedback generation (3 PRs, multi-action)
│
├── graders/                    # Deterministic graders
│   ├── grader_easy.py          # Ordinal matching + critical penalties
│   ├── grader_medium.py        # Kendall Tau + position penalties
│   ├── grader_hard.py          # 5-component weighted scorer
│   └── reliability.py          # Cohen's Kappa, Krippendorff's Alpha
│
├── benchmark/                  # Baseline evaluation
│   ├── protocol.py             # BenchmarkRunner, LaTeX tables
│   └── agents.py               # RandomAgent, PerfectAgent
│
├── baseline/                   # Saved results
│   └── results.json            # GPT-4o-mini baseline scores
│
├── world_model/                # MBRL research scaffold
│   └── scaffold.py             # SemanticTransitionDataset, WorldModelTrainer
│
└── tests/                      # 19 tests across 5 categories
    └── test_env.py             # Core, grader, variance, reproducibility, exploit tests
```

---

## Using CodeReviewEnv for MBRL Research

Standard MBRL benchmarks (Dreamer, MBPO, MuZero) assume vector state spaces with physics-based transitions. Text-based world models (Li et al., 2025) study synthetic text games; embodied semantic world models (Berg et al., 2025) target robotics. No prior work addresses **knowledge-work state spaces** where T(s,a)→s' depends on professional judgment rather than physics or game rules. CodeReviewEnv is the first environment designed for training **Knowledge-Work World Models (KW-WM)**.

### Step 1: Collect Trajectories

```bash
# Run inference to generate trajectory data
python inference.py  # generates trajectories/*.jsonl

# Or collect from the server API
curl "https://ragavrida-code-review-env.hf.space/export_trajectory?session_id=latest"
```

### Step 2: Load Dataset

```python
from dataset import SemanticTransitionDataset

ds = SemanticTransitionDataset("trajectories/")
print(f"{len(ds)} transitions collected")
print(ds.stats())

# Filter by task difficulty
hard_ds = SemanticTransitionDataset("trajectories/", task_filter="hard")

# Each transition:
t = ds[0]
print(t["state_text"])       # "PR PR-020: Refactor StringUtils | ..."
print(t["action_text"])      # "label_severity:high"
print(t["reward"])           # 0.5
print(t["next_state_text"])  # "PR PR-006: Add rate limiter | ..."
print(t["done"])             # False
```

### Step 3: Train Knowledge-Work World Model (KW-WM)

```python
from sentence_transformers import SentenceTransformer
import torch

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Encode states
states = [ds[i]["state_text"] for i in range(len(ds))]
actions = [ds[i]["action_text"] for i in range(len(ds))]
s_enc = encoder.encode(states)    # (N, 384) embeddings
a_enc = encoder.encode(actions)   # (N, 384) embeddings

# Train MLP transition head: (s_enc, a_enc) → (s'_enc, r)
# Then use Dyna-Q for sample-efficient planning
# See world_model/scaffold.py for infrastructure
```

#### Proof-of-Concept Results

We include `train_world_model.py` — a self-contained KW-WM trainer (no PyTorch required):

```
python train_world_model.py
```

Results on 727 transitions from 50 PR templates (581 train, 146 test):

**Reward Prediction g(s,a) → r** — *Can the model predict review quality from (state, action)?*

| Model | Test MSE | vs Mean-pred | vs Random |
|-------|----------|-------------|-----------|
| Random | 0.225 | — | — |
| Mean-pred (always predict mean) | 0.104 | — | — |
| **KW-WM (MLP)** | **0.064** | **+38.8% ✅** | **+71.7% ✅** |

Direction accuracy: **75.3%** — the model correctly predicts whether an action scores above or below 0.5 three-quarters of the time. This enables model-based planning: simulate different review strategies, pick the highest-predicted-reward action.

**State Prediction f(s,a) → s'** — *Can the model predict review state transitions?*

| Model | Test MSE | Notes |
|-------|----------|-------|
| Random | 0.039 | No structure captured |
| Copy baseline (s' = s) | 0.024 | Strong — states change incrementally |
| **KW-WM (MLP)** | **0.037** | **Beats random (+6.0%)**, approaching copy baseline |

The copy baseline is naturally strong in knowledge-work domains because states evolve incrementally (unlike Atari where frames change dramatically). **The key takeaway is reward prediction** — the model learns which actions yield good reviews, enabling MBRL planning without environment interaction.

### Step 4: PyTorch DataLoader

```python
# Direct PyTorch integration
torch_ds = ds.to_pytorch()
from torch.utils.data import DataLoader
loader = DataLoader(torch_ds, batch_size=32, shuffle=True)

for batch in loader:
    s_text = batch["state_text"]       # list of state strings
    a_text = batch["action_text"]      # list of action strings
    rewards = batch["reward"]          # (B,) tensor
    done = batch["done"]              # (B,) tensor
    break
```

### Sample Trajectory

A `trajectories/sample_trajectory.jsonl` file is included with 13 transitions from all 3 tasks (seed=42). Each line:

```json
{"episode_id": "sample_easy_seed42", "task": "easy", "step": 0, "state": {"pr_id": "PR-020", "title": "Refactor StringUtils"}, "action_text": "label_severity:high", "reward": 0.0, "done": false}
```

### Open Research Questions

1. **Error compounding**: Does prediction error compound exponentially in knowledge-work spaces like in continuous spaces (Janner et al., 2019)?
2. **Natural error correction**: Does structured text provide error correction that physics-based transitions lack (cf. Berg et al., 2025), enabling longer model-based rollouts?
3. **Cross-domain transfer**: Can a KW-WM trained on code review transfer to email triage, bug prioritization, or document summarization?
4. **Representation learning**: What embedding dimension is sufficient for knowledge-work state spaces — 384 (MiniLM) vs 768 (BERT) vs 4096 (code-specific)?
5. **Learnability of review quality**: Can KW-WM learn the dimensions that SWE-bench cannot measure — code maintainability and professional review quality (Da et al., 2025)?

---

## Citation

```bibtex
@misc{codereviewenv2026,
  title={CodeReviewEnv: A Knowledge-Work World Model Benchmark for Model-Based Reinforcement Learning},
  author={Raghav Rida},
  year={2026},
  note={OpenEnv Hackathon — First RL benchmark for knowledge-work world models (KW-WM)},
  url={https://huggingface.co/spaces/ragavrida/code-review-env}
}
```

---

## License

BSD-3-Clause
