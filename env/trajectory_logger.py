"""
Trajectory Logger — MBRL Dataset Hook

Each trajectory file is a valid training dataset for a semantic world model.
Format: JSONL where each line is one (s, a, r, s') transition.

To train a world model:
  1. Load trajectories from trajectories/ directory
  2. Encode states with an LLM encoder (e.g. sentence-transformers)
  3. Train transition model f(s_t, a_t) -> (s_{t+1}, r_t)
  4. Use model for planning without real env — Dyna-Q over language state space

This is the first step toward model-based planning over knowledge-work
environments. No existing MBRL benchmark provides this for semantic state spaces.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from env.models import Observation, Action, Reward


class TrajectoryLogger:
    """
    Logs (state, action, reward, next_state) transitions in JSONL format.

    Each episode produces one JSONL file, each line is one transition.
    This format is directly consumable by dataset loaders for semantic
    world model training — see world_model/scaffold.py.

    Research motivation:
      Standard MBRL benchmarks (MuJoCo, Atari) log transitions as
      numerical vectors. SemanticTransitionDataset wraps these JSONL
      files and provides encoding hooks for structured text states.
    """

    def __init__(self, output_dir: str = "trajectories"):
        self.output_dir = output_dir
        self.transitions: List[Dict] = []
        self.episode_id: Optional[str] = None
        self.task: Optional[str] = None
        os.makedirs(self.output_dir, exist_ok=True)

    def start_episode(self, episode_id: str, task: str) -> None:
        """Begin a new episode, clearing any existing transition buffer."""
        self.episode_id = episode_id
        self.task = task
        self.transitions = []

    def log_transition(
        self,
        step: int,
        state: Observation,
        action: Action,
        reward: Reward,
        next_state: Observation,
        done: bool,
    ) -> None:
        """
        Log a single (s, a, r, s') transition.

        Each transition is a complete snapshot suitable for world model
        training: given (state, action), predict (next_state, reward).
        """
        transition = {
            "step": step,
            "state": state.model_dump(),
            "action": action.model_dump(),
            "reward": reward.model_dump(),
            "next_state": next_state.model_dump(),
            "done": done,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode_id": self.episode_id,
            "task": self.task,
        }
        self.transitions.append(transition)

    def save(self) -> str:
        """
        Save episode trajectory to JSONL file.

        Returns the filepath of the saved trajectory.
        Format: trajectories/{task}_{episode_id}.jsonl
        """
        if not self.transitions:
            return ""

        filename = f"{self.task}_{self.episode_id}.jsonl"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            for transition in self.transitions:
                f.write(json.dumps(transition, default=str) + "\n")

        return filepath

    def export(self) -> List[Dict]:
        """
        Return full episode as list of dicts.

        Clean JSONL-ready format for world model training dataset.
        Each dict has keys: step, state, action, reward, next_state, done, timestamp.
        """
        return list(self.transitions)

    def reset(self) -> None:
        """Clear transition buffer for new episode."""
        self.transitions = []
        self.episode_id = None
        self.task = None
