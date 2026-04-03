# CodeReviewEnv: A Semantic MDP Benchmark for Model-Based RL on Knowledge-Work

## Abstract

*Target venue: NeurIPS/ICLR workshop, full paper later*

We introduce **CodeReviewEnv**, the first Semantic MDP benchmark designed for
model-based reinforcement learning on knowledge-work tasks. Unlike existing
MBRL benchmarks which assume continuous vector state spaces governed by
physics, CodeReviewEnv operates over structured text states with semantic
transitions — modeling real software code review. We formalize the **Semantic MDP**
(S-MDP) class, provide three tasks of increasing difficulty with deterministic graders,
and include a world model training scaffold enabling researchers to study
semantic transition model learning. Baseline results with GPT-4o-mini show
composite score 0.66, with significant headroom demonstrating benchmark utility.

---

## 1. Introduction

The success of Model-Based RL in physical domains — Dreamer on Atari,
MBPO on MuJoCo, MuZero on board games — rests on a key assumption:
the environment has a transition function that can be approximated by
a neural network operating on continuous vectors or pixel arrays.

Meanwhile, LLM agent benchmarks (AgentBench, WebArena, SWE-bench) have
demonstrated impressive agent capabilities on real-world tasks. But these
benchmarks are **evaluation suites**, not RL environments. They lack:
- Standard MDP formalism (state/action/reward/transition)
- Trajectory logging in (s, a, r, s') format
- Infrastructure for learning transition models
- Controlled difficulty levels with deterministic grading

This leaves a gap: **there is no benchmark for studying model-based RL
in semantic domains** — domains where the state is structured text and
transitions depend on meaning rather than physics.

CodeReviewEnv fills this gap. Key contributions:
1. **S-MDP formalism**: formal definition of Semantic MDPs
2. **Three-difficulty benchmark**: easy/medium/hard tasks with deterministic graders
3. **World model scaffold**: training infrastructure for semantic transition models
4. **Trajectory datasets**: JSONL logging for (s, a, r, s') transitions

## 2. Related Work

### MBRL Benchmarks
- **MuJoCo** (Todorov et al.): continuous joints, physics transition
- **DMControl** (Tassa et al.): physics simulation, pixel/state observations
- **Atari / ALE** (Bellemare et al.): pixel states, game engine transition
- **ProcGen** (Cobbe et al.): procedurally generated game levels

All operate on continuous vector or pixel state spaces with physics-based transitions.

### LLM Agent Benchmarks
- **AgentBench** (Liu et al.): multi-task, measures success/fail
- **WebArena** (Zhou et al.): web browsing tasks, no MDP formalism
- **SWE-bench** (Jimenez et al.): software engineering, pass/fail grading
- **LATM** (Cai et al.): LLMs as tool makers, planning focus

None provide MDP formalism, trajectory logging, or world model support.

### Text-Based RL
- **Jericho** (Hausknecht et al.): text adventure games
- **TextWorld** (Côté et al.): synthetic text environments
- **ALFWorld** (Shridhar et al.): embodied instruction following

These are synthetic game worlds, not real-world knowledge-work tasks.

## 3. The Semantic MDP Formalism

**Definition.** A Semantic MDP (S-MDP) is a tuple (S, A, T, R, γ) where:
- S is a semantic state space: each s ∈ S is structured text with metadata
- A is a structured action space: typed decisions with heterogeneous fields
- T: S × A → S is a semantic transition function not expressible in closed form
- R: S × A × S → [-1, 1] is a shaped reward with trajectory-level components
- γ ∈ (0, 1) is the discount factor

**Key property:** T cannot be derived analytically. It must be learned from data.
This distinguishes S-MDPs from physics-based MDPs where T approximates a known equation.

## 4. CodeReviewEnv

### 4.1 Environment Design
- 20 hand-crafted PRs across Python, JavaScript, Java, Go
- 8 bug categories with ground truth severity labels
- 3 author experience levels affecting bug probability
- Pre-annotated human labels for grader validity

### 4.2 Tasks
1. **Easy — Severity Labeling**: classify PR bug severity (5 steps)
2. **Medium — Queue Prioritization**: order PRs by review urgency (3 steps)
3. **Hard — Feedback Generation**: add targeted comments + decision (3 PRs)

### 4.3 Grader Design
All graders are fully deterministic (no LLM calls, no randomness):
- Easy: ordinal matching with asymmetric critical penalties
- Medium: Kendall Tau rank correlation with position constraints
- Hard: 5-component weighted score (relevance, specificity, actionability, coverage, precision)

### 4.4 Reward Shaping
Trajectory-level components beyond per-step reward:
- Efficiency bonus (+0.1): complete under budget
- Coverage bonus (+0.15): catch all critical bugs
- Consistency penalty (-0.2): contradicting own labels
- Exploit penalty (-0.5): approve with unaddressed critical bug

## 5. Experiments

### 5.1 Baseline Results

| Agent | Easy | Medium | Hard | Composite |
|-------|------|--------|------|-----------|
| Random | 0.21 ± 0.09 | 0.31 ± 0.11 | 0.09 ± 0.05 | 0.18 |
| GPT-4o-mini | 0.85 ± 0.07 | 0.37 ± 0.00 | 0.78 ± 0.06 | 0.66 |
| Perfect | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.91 ± 0.03 | 0.97 |

### 5.2 Key Findings
- Significant headroom between GPT-4o-mini and perfect agent (0.66 vs 0.97)
- Medium task is hardest for LLMs: GPT-4o-mini scores only 0.37 (Kendall tau ≈ 0.0–0.6)
- Easy task nearly solved: 0.85 accuracy on severity labeling
- Hard task surprisingly strong at 0.78: LLMs write effective code review comments
- Random agent well below all trained agents (floor = 0.18)
- Graders show substantial agreement with human labels (κ > 0.7)

### 5.3 Failure Mode Analysis
- Easy: GPT-4o-mini over-labels low as high (systematic over-severity bias)
- Medium: security PRs correctly prioritized, but within-severity ordering is near-random
- Hard: high decision quality (approve/reject) but variable comment precision across PRs

## 6. World Model Experiments (Future Work)

### 6.1 Proposed Architecture
- State encoder: fine-tuned sentence-transformer
- Action encoder: one-hot + severity embedding
- Transition model: MLP head predicting (next_state_embedding, reward)

### 6.2 Research Questions
- Error compounding rate in semantic vs physical spaces
- Optimal state encoding for transition prediction
- Planning horizon limits for semantic world models
- Transfer potential across knowledge-work domains

## 7. Conclusion and Future Work

CodeReviewEnv is the first benchmark designed for studying model-based RL
on semantic environments. Future directions:
- **Causal semantic world models**: learning causal structure in code review decisions
- **Multi-agent code review**: multiple reviewers with different expertise
- **Cross-domain transfer**: code review → email triage → document review
- **Curriculum learning**: progressive difficulty within episodes

---

## References

*To be populated with full citations for camera-ready version.*
