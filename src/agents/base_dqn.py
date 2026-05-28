"""
DQN Backbone shared by all three agent variants.
─────────────────────────────────────────────────
Provides:
  • QNetwork    — three-layer FC network with optional dueling heads
  • ReplayBuffer — uniform experience replay with configurable capacity
  • DQNAgent    — MDP baseline (λ = 0, no Logos component)

All MLP variants subclass DQNAgent and override:
  `_td_target`  — Logos-augmented Bellman target (Eq. 2)
  `select_action` — optionally shaped by Φ_L (MLP-Full only)
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Q-Network ─────────────────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """
    Three fully-connected layers (64 → 64 → |A|) with ReLU activations.
    Architecture matches Section V-B of the paper.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────


class ReplayBuffer:
    """
    Uniform random experience replay (capacity = 10 000 by default).
    """

    def __init__(self, capacity: int = 10_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── DQN Agent (MDP baseline) ──────────────────────────────────────────────────


class DQNAgent:
    """
    Standard Deep Q-Network agent — the MDP baseline (λ = 0).

    Parameters
    ----------
    obs_dim          : int
    n_actions        : int
    lr               : float  — Adam learning rate
    gamma            : float  — discount factor
    batch_size       : int
    replay_capacity  : int
    target_update_freq : int  — steps between target network syncs
    eps_start / eps_end / eps_decay_episodes
    device           : torch.device
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_capacity: int = 10_000,
        target_update_freq: int = 100,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 300,
        device: Optional[torch.device] = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net      = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(replay_capacity)

        # Counters
        self._step    = 0
        self._episode = 0
        self.losses: list[float] = []

    # ── Epsilon schedule ─────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        frac = min(self._episode / max(self.eps_decay_episodes, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray, exploit: bool = False) -> int:
        """ε-greedy action selection (pure Q-values, no Logos shaping)."""
        if not exploit and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(obs_t).squeeze(0)
        return int(q.argmax().item())

    # ── TD update ────────────────────────────────────────────────────────────

    def _td_target(
        self,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        next_obs_np: Optional[np.ndarray] = None,
        actions_np: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Standard Bellman target: r + γ · max_a' Q_target(s', a')
        Overridden in MLP variants to add Φ_L.
        """
        with torch.no_grad():
            q_next = self.target_net(next_obs).max(dim=1)[0]
        return rewards + self.gamma * q_next * (1 - dones)

    def update(self) -> Optional[float]:
        """Sample a batch and perform one gradient step. Returns loss or None."""
        if len(self.replay) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size)

        obs_t      = torch.tensor(obs,      device=self.device)
        actions_t  = torch.tensor(actions,  device=self.device)
        rewards_t  = torch.tensor(rewards,  device=self.device)
        next_obs_t = torch.tensor(next_obs, device=self.device)
        dones_t    = torch.tensor(dones,    device=self.device)

        # Current Q values
        q_pred = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target (possibly Logos-augmented in subclasses)
        q_target = self._td_target(rewards_t, next_obs_t, dones_t, next_obs, actions)

        loss = nn.functional.mse_loss(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    # ── Persistence ──────────────────────────────────────────────────────────

    def state_dict_export(self) -> dict:
        return {
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "episode":    self._episode,
            "step":       self._step,
        }

    def load_state_dict_import(self, sd: dict) -> None:
        self.q_net.load_state_dict(sd["q_net"])
        self.target_net.load_state_dict(sd["target_net"])
        self._episode = sd.get("episode", 0)
        self._step    = sd.get("step", 0)
