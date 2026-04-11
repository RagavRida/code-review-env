---
title: CodeReviewEnv
emoji: "\U0001F50D"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
short_description: "RL benchmark for automated code review"
tags:
  - openenv
  - reinforcement-learning
  - code-review
  - mbrl
  - semantic-mdp
  - llm-agents
---

# CodeReviewEnv

**An OpenEnv-compliant RL environment for automated code review.**

CodeReviewEnv formalizes software code review as a Semantic Markov Decision Process (S-MDP). Agents receive buggy code snippets, identify bugs, flag line numbers, suggest fixes, and write review comments. A 5-signal shaped reward provides rich training signal for reinforcement learning.

## Why Code Review as RL?

Code review is a core knowledge-work task that requires understanding program semantics, identifying patterns across languages, and producing actionable feedback. Unlike toy text environments, CodeReviewEnv uses real code patterns and deterministic grading --- no LLM-in-the-loop evaluation.

| Benchmark | Domain | State Space | Reward | World Model? |
|-----------|--------|-------------|--------|--------------|
| MuJoCo | Robotics | R^n | Physics | Yes |
| Atari | Games | Pixels | Score | Yes |
| SWE-bench | Code | Text | Pass/Fail | No |
| **CodeReviewEnv** | **Code Review** | **Source code** | **5-signal shaped** | **Yes** |

## MDP Formulation

| Component | Description |
|-----------|-------------|
| **State (S)** | Buggy source code + language + difficulty metadata |
| **Action (A)** | `{issues, flagged_lines, suggestion, comment}` |
| **Transition (T)** | Deterministic: episode ends after review submission |
| **Reward (R)** | Weighted combination of 5 normalized signals |
| **Discount** | 1.0 (single-step episodes) |

## Reward Breakdown

| Signal | Weight | Description |
|--------|--------|-------------|
| `bug_detection` | 0.40 | Fraction of gold bugs identified via keyword matching |
| `fix_quality` | 0.25 | Similarity of suggested fix to gold fix (difflib + keyword overlap) |
| `line_precision` | 0.15 | F1 score of flagged lines vs gold lines (+-3 line tolerance) |
| `comment_quality` | 0.10 | Length, actionability keywords, specificity heuristics |
| `efficiency` | 0.10 | Penalty for excessive steps and hint usage |

Hard difficulty gives a 1.5x multiplier on `bug_detection`.

## Difficulty Tiers

| Tier | Bugs | Snippet Size | Bug Types |
|------|------|-------------|-----------|
| `easy` | 1 | < 20 lines | Common patterns |
| `medium` | 1-2 | 20-50 lines | Logic errors, mixed |
| `hard` | 2-3 | 50+ lines | Interacting bugs, subtle |

## Procedural Generation

Every `reset()` produces a unique episode:
1. Picks a clean snippet from a bank of 35+ functions (Python/JS/Go)
2. Applies 1-3 bug injectors based on difficulty
3. Stores gold answers in State (never exposed to agent)

### Bug Injectors

| Injector | What it does |
|----------|-------------|
| `off_by_one` | Flips `<` to `<=`, adjusts `range()` bounds |
| `null_deref` | Removes a null/None/nil guard |
| `wrong_operator` | Swaps `+` / `-`, `*` / `/` |
| `unused_var` | Inserts dead variable shadowing a live one |
| `logic_inversion` | Flips `True`/`False`, `and`/`or`, `==`/`!=` |

## Quick Start

```bash
# Install
cd code-review-env
pip install -r requirements.txt

# Run baseline (no API key needed)
python baseline.py

# Run tests
pytest tests/ -v

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Usage

```python
from server.code_review_environment import CodeReviewEnvironment
from models import CodeReviewAction

env = CodeReviewEnvironment()
obs = env.reset(seed=42, difficulty="easy")

print(obs.code)        # Buggy code to review
print(obs.language)    # python, javascript, or go
print(obs.difficulty)  # easy, medium, or hard

action = CodeReviewAction(
    issues=["Off-by-one error in loop boundary"],
    flagged_lines=[3],
    suggestion="Change < to <= on line 3",
    comment="The loop exits too early, missing the last element.",
)

result = env.step(action)
print(f"Reward: {result.reward:.3f}")
print(f"Breakdown: {result.reward_breakdown}")
```

## MCP Tools

For tool-calling agents, CodeReviewEnv exposes MCP endpoints:

```
POST /mcp/reset            - Start a new episode
POST /mcp/get_code_snippet - Get current buggy code
POST /mcp/submit_review    - Submit review, get reward
POST /mcp/request_hint     - Get a hint (-0.05 reward penalty)
GET  /mcp/get_state        - Get episode state summary
```

## Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

## Inference

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export API_KEY="your-key"
export MODEL_NAME="openai/gpt-4o-mini"
python inference.py
```

## Project Structure

```
code-review-env/
  models.py                  # Pydantic Action/Observation/State
  snippet_bank.py            # 35+ snippets + 5 bug injectors
  reward.py                  # 5-signal shaped reward
  client.py                  # EnvClient for WebSocket
  inference.py               # LLM evaluation (mandatory)
  baseline.py                # Heuristic agent (no API key)
  server/
    app.py                   # FastAPI + MCP endpoints
    code_review_environment.py  # Core Environment class
  tests/
    test_code_review_env.py
  openenv.yaml               # OpenEnv manifest
  Dockerfile                 # Production container
  requirements.txt
  README.md
```

## Citation

```bibtex
@software{codereviewenv2024,
  title={CodeReviewEnv: A Semantic MDP for Automated Code Review},
  author={CodeReviewEnv Team},
  year={2024},
  url={https://github.com/ragavrida/code-review-env}
}
```
