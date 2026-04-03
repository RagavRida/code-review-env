# Why CodeReviewEnv Matters for MBRL Research

## The Gap

Every Model-Based RL benchmark today assumes the world is physical:
MuJoCo (robotic joints), Atari (pixel frames), DMControl (physics simulation).
The transition function f(s,a) → s' is always a deterministic physics engine.
The world model just learns to approximate something with a mathematical ground truth.

Agent benchmarks (AgentBench, WebArena, SWE-bench) go further — they operate on
real-world tasks. But they measure only success/failure. There is no MDP formalism,
no trajectory dataset, no path to learning a transition model. They are evaluation
suites, not RL environments.

## The Formal Problem: Semantic MDP (S-MDP)

We define a **Semantic Markov Decision Process (S-MDP)** as a tuple:

    (S, A, T, R, γ)

Where:
- **S** — semantic state space: structured text + metadata (not R^n)
- **A** — structured action space: typed decisions over semantic entities
- **T** — semantic transition function: T(s, a) → s' where s,s' ∈ S
  - T is **not expressible as a closed-form equation**
  - T must be **learned from trajectory data**
- **R** — shaped reward: R(s, a, s') → [-1, 1] with trajectory-level components
- **γ** — discount factor: 0.95 (standard)

This is distinct from:
- **POMDPs**: partial observability, not semantic transitions
- **Text games** (Jericho, TextWorld): synthetic game worlds, not real-world tasks
- **LLM agent benchmarks**: measure success/failure, no MDP formalism, no trajectories
- **Standard MBRL benchmarks**: continuous vector state, physics-based transitions

## What Changes Here

In CodeReviewEnv, the transition depends on **meaning**:

- **State**: a pull request with code diffs, bug patterns, author context, review queue
- **Action**: a review decision (label severity, prioritize queue, add comment)
- **Next state**: updated queue, new PRs, changed review context

There is no equation for this. You cannot derive T analytically.
The transition is **semantic** — it depends on understanding code,
recognizing bug patterns, and making judgment calls.

## The New Model Class: Semantic World Model

A semantic world model must learn:

    M(state_t, action_t) → (state_{t+1}, reward_t)

Where state is **structured text**, not a vector. This requires encoding
semantic relationships — bug severity, code quality, reviewer judgment —
into a learnable transition function.

This model class does not exist in the literature. No benchmark supports it.
CodeReviewEnv is the first environment designed for its study.

## How to Use This Env for MBRL Research

1. **Collect trajectories**: Run agents to generate (s, a, r, s') data via `export_trajectory()`
2. **Build dataset**: Load JSONL files into `SemanticTransitionDataset`
3. **Encode states**: Use sentence-transformers or fine-tuned LLM encoder
4. **Train transition model**: Fine-tuned encoder + MLP head for (next_state, reward)
5. **Plan with model**: Imagine rollouts without real env
6. **Dyna-Q over language**: Sample-efficient learning for knowledge-work agents

## Key Research Questions

1. **Error compounding**: Does model error grow exponentially with rollout horizon H
   in semantic spaces, as it does in continuous spaces? Or does the structured nature
   of text provide error correction?

2. **State representation**: What encoding produces the best transition model?
   Sentence embeddings? Fine-tuned LLM hidden states? Structured feature vectors?

3. **Transfer**: Can a world model trained on code review transfer to other
   knowledge-work domains (email triage, document review, bug prioritization)?

4. **Planning horizon**: How far ahead can a semantic world model reliably plan?
   Is H=3 useful? H=10? Does the answer differ from physical world models?

## Why This Is Novel

| Benchmark | Domain | State Space | Transition | World Model? |
|-----------|--------|-------------|------------|-------------|
| MuJoCo | Robotics | R^n (joints) | Physics | Yes (Dreamer, MBPO) |
| Atari | Games | Pixels | Engine | Yes (MuZero) |
| DMControl | Physics | R^n | Simulation | Yes (DreamerV3) |
| AgentBench | Tasks | Text | N/A | No (eval only) |
| WebArena | Web | DOM | N/A | No (eval only) |
| SWE-bench | Code | Text | N/A | No (eval only) |
| **CodeReviewEnv** | **Code Review** | **Semantic Text** | **Semantic** | **Yes (this work)** |

CodeReviewEnv is the first benchmark designed for semantic MBRL from the ground up.
