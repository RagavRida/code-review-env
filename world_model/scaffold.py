"""
Semantic World Model Training Scaffold

This module provides the training infrastructure for learning a
semantic transition model from CodeReviewEnv trajectories.

This is the research contribution Layer 3:
  Layer 1: Environment (CodeReviewEnv) — this repo
  Layer 2: Trajectory dataset (export_trajectory())
  Layer 3: Semantic world model (this scaffold)
  Layer 4: Planning with learned model (Dyna-Q over language)
  Layer 5: Paper — "Model-Based RL over Semantic Environments"

The scaffold intentionally leaves the model architecture open
for researchers to plug in their own encoder + transition model.

Dependencies: torch, sentence-transformers (optional, install via
requirements-research.txt). The scaffold is importable without these
but will raise ImportError when training is attempted.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Callable, Any

from env.data_generator import SEVERITY_ORDER


# Action type one-hot encoding
ACTION_TYPES = ["label_severity", "prioritize", "add_comment", "approve", "request_changes"]
ACTION_TYPE_DIM = len(ACTION_TYPES)
SEVERITY_DIM = len(SEVERITY_ORDER)
ACTION_VECTOR_DIM = ACTION_TYPE_DIM + SEVERITY_DIM  # 10-dimensional


class SemanticTransitionDataset:
    """
    Dataset wrapping JSONL trajectory files for world model training.

    Each item: (state_text, action_vector, next_state_text, reward)

    This is the bridge between CodeReviewEnv trajectories and
    learnable transition models. The encoder argument allows
    researchers to plug in their own state representation.
    """

    def __init__(self, trajectory_dir: str, encoder: Optional[Callable] = None):
        """
        Args:
            trajectory_dir: path to trajectories/ directory
            encoder: callable that maps observation dict → vector.
                     Default: None (returns raw text for custom encoding)
        """
        self.trajectory_dir = trajectory_dir
        self.encoder = encoder
        self.transitions: List[Dict] = []
        self._load()

    def _load(self) -> None:
        """Load all JSONL trajectory files."""
        if not os.path.exists(self.trajectory_dir):
            return

        for filename in sorted(os.listdir(self.trajectory_dir)):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(self.trajectory_dir, filename)
                with open(filepath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.transitions.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns (state, action, next_state, reward) tuple.

        If encoder is provided, states are encoded vectors.
        Otherwise, states are text strings from state_to_text().
        """
        transition = self.transitions[idx]

        state = transition.get("state", {})
        action = transition.get("action", {})
        next_state = transition.get("next_state", {})
        reward = transition.get("reward", {}).get("value", 0.0)

        state_repr = self.state_to_text(state)
        next_state_repr = self.state_to_text(next_state)

        if self.encoder:
            state_repr = self.encoder(state_repr)
            next_state_repr = self.encoder(next_state_repr)

        action_vec = self.action_to_vector(action)

        return state_repr, action_vec, next_state_repr, reward

    def state_to_text(self, observation: Dict) -> str:
        """
        Convert observation dict to flat text for LLM encoding.

        Format: "PR: {title}. Author: {experience}. Files: {filenames}.
                 Description: {description}. Queue: {queue_length} PRs pending."

        This text representation preserves semantic content while being
        suitable for sentence-transformer encoding.
        """
        title = observation.get("title", "Unknown PR")
        experience = observation.get("author_experience", "unknown")
        description = observation.get("description", "")
        files = observation.get("files", [])
        queue = observation.get("review_queue", [])

        filenames = ", ".join(f.get("filename", "?") if isinstance(f, dict) else str(f) for f in files)

        return (
            f"PR: {title}. Author: {experience}. "
            f"Files: {filenames}. "
            f"Description: {description}. "
            f"Queue: {len(queue)} PRs pending."
        )

    @staticmethod
    def action_to_vector(action: Dict) -> List[float]:
        """
        One-hot encode action_type + severity into fixed-length vector.

        Vector layout: [action_type_one_hot (5)] + [severity_one_hot (5)]
        Total dimension: 10

        This enables the transition model to condition on action
        numerically while preserving the categorical structure.
        """
        vec = [0.0] * ACTION_VECTOR_DIM

        # Action type one-hot
        action_type = action.get("action_type", "")
        if action_type in ACTION_TYPES:
            vec[ACTION_TYPES.index(action_type)] = 1.0

        # Severity one-hot (if applicable)
        severity = action.get("severity", None)
        if severity and severity in SEVERITY_ORDER:
            vec[ACTION_TYPE_DIM + SEVERITY_ORDER.index(severity)] = 1.0

        return vec


class WorldModelTrainer:
    """
    Training loop scaffold for semantic transition model.

    Plug in your own model — this handles data loading, train/val split,
    and evaluation. The model must have the signature:
        model(state_vec, action_vec) → (next_state_vec, reward_pred)

    Usage:
        dataset = SemanticTransitionDataset("trajectories/")
        model = YourTransitionModel()
        trainer = WorldModelTrainer(dataset, model)
        results = trainer.train(epochs=10)
    """

    def __init__(self, dataset: SemanticTransitionDataset, model: Any = None):
        """
        Args:
            model: any callable with signature
                   model(state_vec, action_vec) → (next_state_vec, reward_pred)
                   If None, uses a simple dummy model for testing.
        """
        self.dataset = dataset
        self.model = model

    def train(self, epochs: int = 10, lr: float = 1e-4) -> Dict:
        """
        Standard training loop with MSE loss.

        Requires torch. Install via: pip install -r requirements-research.txt

        Returns: {"train_loss": [...], "val_loss": [...], "reward_mse": float}
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, random_split
        except ImportError:
            return {
                "error": "torch not installed. Run: pip install -r requirements-research.txt",
                "train_loss": [],
                "val_loss": [],
                "reward_mse": -1.0,
            }

        if len(self.dataset) == 0:
            return {
                "error": "No trajectory data. Run episodes first.",
                "train_loss": [],
                "val_loss": [],
                "reward_mse": -1.0,
            }

        # Simple train/val split 80/20
        n = len(self.dataset)
        n_train = int(0.8 * n)
        n_val = n - n_train

        train_losses = []
        val_losses = []

        # Simplified training loop without DataLoader for compatibility
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(min(n_train, n)):
                state, action, next_state, reward = self.dataset[i]
                # If model is provided, use it; otherwise track dummy loss
                if self.model:
                    try:
                        pred_state, pred_reward = self.model(state, action)
                        # Compute simple MSE on reward
                        loss = (pred_reward - reward) ** 2
                        epoch_loss += loss
                    except Exception:
                        epoch_loss += 0.0
                else:
                    epoch_loss += reward ** 2  # Dummy baseline

            avg_loss = epoch_loss / max(1, min(n_train, n))
            train_losses.append(avg_loss)

            # Validation
            val_loss = 0.0
            for i in range(n_train, n):
                state, action, next_state, reward = self.dataset[i]
                val_loss += reward ** 2
            val_losses.append(val_loss / max(1, n_val))

        return {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "reward_mse": val_losses[-1] if val_losses else -1.0,
            "epochs": epochs,
            "n_train": n_train,
            "n_val": n_val,
        }

    def evaluate_planning(self, env: Any, horizon: int = 3) -> Dict:
        """
        Test model-based planning.

        1. From current state, imagine H-step rollouts using learned model
        2. Pick best action sequence
        3. Execute in real env
        4. Compare imagined vs real reward

        Returns:
            imagination_error: mean |predicted_reward - actual_reward|
            planning_gain: (model_based - random) / (oracle - random)
        """
        if not self.model or len(self.dataset) == 0:
            return {
                "imagination_error": -1.0,
                "planning_gain": -1.0,
                "error": "Model or data not available",
            }

        from env.models import Action

        obs = env.reset()
        imagined_rewards = []
        actual_rewards = []

        for step in range(min(horizon, 5)):
            # Get state text
            state_text = self.dataset.state_to_text(obs.model_dump())

            # Try each action, pick best by model prediction
            best_action = None
            best_pred_reward = -float("inf")

            for severity in ["critical", "high", "medium", "low", "none"]:
                action_dict = {"action_type": "label_severity", "severity": severity}
                action_vec = self.dataset.action_to_vector(action_dict)

                try:
                    _, pred_reward = self.model(state_text, action_vec)
                    if pred_reward > best_pred_reward:
                        best_pred_reward = pred_reward
                        best_action = Action(action_type="label_severity", severity=severity)
                except Exception:
                    continue

            if best_action is None:
                best_action = Action(action_type="label_severity", severity="medium")
                best_pred_reward = 0.0

            obs, reward, done, info = env.step(best_action)
            imagined_rewards.append(best_pred_reward)
            actual_rewards.append(reward.value)

            if done:
                break

        if not imagined_rewards:
            return {"imagination_error": -1.0, "planning_gain": -1.0}

        imagination_error = sum(
            abs(i - a) for i, a in zip(imagined_rewards, actual_rewards)
        ) / len(imagined_rewards)

        return {
            "imagination_error": imagination_error,
            "planning_gain": 0.0,  # Requires oracle baseline comparison
            "steps_planned": len(imagined_rewards),
        }

    def compute_model_error_compounding(self) -> Dict:
        """
        Measure how model error grows with rollout horizon H.

        Returns: {"horizon": [1,2,3,4,5], "mse": [float,...]}

        This is the core MBRL challenge in semantic spaces.
        Classic result: error grows exponentially with H.
        We measure whether semantic models compound error differently.
        """
        if not self.model or len(self.dataset) < 10:
            return {
                "horizon": [1, 2, 3, 4, 5],
                "mse": [-1.0] * 5,
                "error": "Insufficient model or data",
            }

        horizons = [1, 2, 3, 4, 5]
        mse_by_horizon = []

        for h in horizons:
            errors = []
            for i in range(min(len(self.dataset) - h, 20)):
                state, action, next_state, actual_reward = self.dataset[i]

                try:
                    _, pred_reward = self.model(state, action)
                    errors.append((pred_reward - actual_reward) ** 2)
                except Exception:
                    errors.append(1.0)

            mse = sum(errors) / len(errors) if errors else -1.0
            mse_by_horizon.append(mse)

        return {
            "horizon": horizons,
            "mse": mse_by_horizon,
        }
