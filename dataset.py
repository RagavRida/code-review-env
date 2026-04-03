"""
SemanticTransitionDataset — PyTorch-compatible dataset for MBRL research.

Loads trajectory JSONL files exported from CodeReviewEnv episodes and
provides (state, action, reward, next_state, done) transitions for
training Knowledge-Work World Models (KW-WM).

Usage:
    from dataset import SemanticTransitionDataset

    ds = SemanticTransitionDataset("trajectories/")
    print(f"{len(ds)} transitions collected")

    # Each item is a dict with keys:
    #   state_text:      str — serialized observation (PR diff, context)
    #   action_text:     str — serialized action (type + params)
    #   reward:          float — grader reward for this transition
    #   next_state_text: str — serialized next observation
    #   done:            bool — whether episode ended
    #   task:            str — easy|medium|hard
    #   step:            int — step number in episode

    # For embedding-based world models:
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    s_enc = encoder.encode(ds[0]["state_text"])
    # Train: MLP(s_enc, a_enc) → (s'_enc, r_pred)

Research context:
    Standard MBRL benchmarks (Dreamer, MBPO, MuZero) assume vector state
    spaces with physics-based transitions. Text-based world models (Li et al.,
    2025) study synthetic text games; embodied SWMs (Berg et al., 2025) target
    robotics. CodeReviewEnv enables training the first **Knowledge-Work World
    Models (KW-WM)** — world models over structured professional text where
    T(s,a)→s' depends on professional judgment rather than physics or game rules.

    Open questions this dataset enables:
    1. Does prediction error compound exponentially in knowledge-work spaces?
    2. Does structured text provide natural error correction vs. continuous?
    3. Can a KW-WM transfer across knowledge-work domains?
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class SemanticTransitionDataset:
    """
    Loads trajectory JSONL files for training a Knowledge-Work World Model.

    Compatible with PyTorch Dataset interface (implements __len__ and __getitem__).
    Each trajectory file is a JSONL where each line is a transition dict.

    Args:
        trajectory_dir: Path to directory containing .jsonl trajectory files
        task_filter: Optional — only load trajectories for this task (easy|medium|hard)
        max_transitions: Optional — cap total transitions loaded (for memory)
    """

    def __init__(
        self,
        trajectory_dir: str,
        task_filter: Optional[str] = None,
        max_transitions: Optional[int] = None,
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.transitions: List[Dict[str, Any]] = []
        self._load(task_filter, max_transitions)

    def _load(self, task_filter: Optional[str], max_transitions: Optional[int]) -> None:
        """Load all .jsonl files from the trajectory directory."""
        if not self.trajectory_dir.exists():
            return

        for fpath in sorted(self.trajectory_dir.glob("*.jsonl")):
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        transition = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Apply task filter if specified
                    if task_filter and transition.get("task") != task_filter:
                        continue

                    self.transitions.append(transition)

                    if max_transitions and len(self.transitions) >= max_transitions:
                        return

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single transition as a dict.

        Keys:
            state_text (str): Serialized observation text
            action_text (str): Serialized action string
            reward (float): Step reward
            next_state_text (str): Serialized next observation
            done (bool): Whether episode ended
            task (str): Task difficulty level
            step (int): Step number in episode
        """
        t = self.transitions[idx]
        return {
            "state_text": t.get("state_text", json.dumps(t.get("state", {}))),
            "action_text": t.get("action_text", json.dumps(t.get("action", {}))),
            "reward": self._extract_reward(t),
            "next_state_text": t.get("next_state_text", json.dumps(t.get("next_state", {}))),
            "done": bool(t.get("done", False)),
            "task": t.get("task", "unknown"),
            "step": int(t.get("step", 0)),
        }

    @staticmethod
    def _extract_reward(t: Dict) -> float:
        """Extract reward as float, handling dict or float formats."""
        r = t.get("reward", 0.0)
        if isinstance(r, dict):
            return float(r.get("value", 0.0))
        try:
            return float(r)
        except (TypeError, ValueError):
            return 0.0

    def get_episode(self, episode_id: str) -> List[Dict[str, Any]]:
        """Get all transitions from a specific episode."""
        return [t for t in self.transitions if t.get("episode_id") == episode_id]

    def get_episodes(self) -> List[str]:
        """Get all unique episode IDs."""
        return list(set(t.get("episode_id", "unknown") for t in self.transitions))

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for the loaded dataset."""
        episodes = self.get_episodes()
        rewards = [self._extract_reward(t) for t in self.transitions]
        tasks = {}
        for t in self.transitions:
            task = t.get("task", "unknown")
            tasks[task] = tasks.get(task, 0) + 1

        return {
            "total_transitions": len(self.transitions),
            "total_episodes": len(episodes),
            "task_distribution": tasks,
            "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "reward_min": min(rewards) if rewards else 0.0,
            "reward_max": max(rewards) if rewards else 0.0,
        }

    def to_pytorch(self):
        """Convert to a PyTorch-compatible dataset (requires torch)."""
        try:
            import torch
            from torch.utils.data import Dataset as TorchDataset

            parent = self

            class _TorchWrapper(TorchDataset):
                def __len__(self):
                    return len(parent)

                def __getitem__(self, idx):
                    item = parent[idx]
                    return {
                        "state_text": item["state_text"],
                        "action_text": item["action_text"],
                        "reward": torch.tensor(item["reward"], dtype=torch.float32),
                        "next_state_text": item["next_state_text"],
                        "done": torch.tensor(item["done"], dtype=torch.bool),
                    }

            return _TorchWrapper()
        except ImportError:
            raise ImportError("PyTorch is required for to_pytorch(). Install with: pip install torch")
