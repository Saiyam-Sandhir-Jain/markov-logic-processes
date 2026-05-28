"""
MLP-Bellman and MLP-Full Agent Variants
────────────────────────────────────────
MLP-Bellman: Logos shaping only in the TD target (Eq. 2).
MLP-Full:    Logos shaping in both the TD target AND action selection.
             π(s) = argmax_a [ Q(s, a) + λ·Φ_L(s, a) ]

Both variants share the DQNAgent backbone and override only the
methods that differ from the baseline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base_dqn import DQNAgent
from ..logos.potential import Logos


class MLPBellmanAgent(DQNAgent):
    """
    MLP-Bellman ablation: Logos-augmented Bellman target only.
    Action selection remains pure ε-greedy on Q(s, a).
    """

    def __init__(self, *args, logos: Optional[Logos] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logos = logos or Logos(n_actions=self.n_actions)

    # ── Experience routing ───────────────────────────────────────────────────

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Push experience to both replay buffer and Logos window."""
        self.replay.push(obs, action, reward, next_obs, done)
        self.logos.add_experience(obs, action, reward)

    # ── Logos-augmented Bellman target (Eq. 2) ───────────────────────────────

    def _td_target(
        self,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        next_obs_np: Optional[np.ndarray] = None,
        actions_np: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            q_next = self.target_net(next_obs).max(dim=1)[0]

        # Φ_L shaping on next state (greedy action)
        phi_bonus = torch.zeros_like(rewards)
        if next_obs_np is not None:
            for i, nobs in enumerate(next_obs_np):
                best_a = int(q_next[i].item())  # greedy w.r.t. target Q
                phi_bonus[i] = self.logos.phi(nobs, best_a)

        return rewards + self.gamma * (q_next + phi_bonus) * (1 - dones)

    # ── Model export ─────────────────────────────────────────────────────────

    def state_dict_export(self) -> dict:
        sd = super().state_dict_export()
        sd["logos_rules"] = self.logos.miner.rules
        sd["logos_UL"]    = self.logos.UL
        return sd


class MLPFullAgent(MLPBellmanAgent):
    """
    MLP-Full (proposed): Logos shaping in TD target AND action selection.
    During exploitation: π(s) = argmax_a [ Q(s,a) + λ·Φ_L(s,a) ]
    """

    def select_action(self, obs: np.ndarray, exploit: bool = False) -> int:
        """
        ε-greedy with Logos-augmented exploitation.

        During exploration (random): pure random, no Logos overhead.
        During exploitation: argmax [ Q(s, a) + λ·Φ_L(s, a) ]
        With deductive override when U_L > 0.9.
        """
        import random

        if not exploit and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(obs_t).squeeze(0).cpu().numpy()

        # Deductive override (Fix 4)
        q_vals = self.logos.deduction_override_value(obs, q_vals)

        # Add Φ_L for each action
        phi_arr = self.logos.phi_all_actions(obs)
        combined = q_vals + phi_arr

        return int(combined.argmax())

    def select_action_with_explanation(
        self, obs: np.ndarray, exploit: bool = False
    ):
        """
        Like select_action but also returns:
            (action, q_values, phi_values, matched_rules_for_chosen_action)
        Used by the video generation notebook.
        """
        import random

        if not exploit and random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
            return action, None, None, []

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(obs_t).squeeze(0).cpu().numpy()

        q_vals = self.logos.deduction_override_value(obs, q_vals)
        phi_arr = self.logos.phi_all_actions(obs)
        combined = q_vals + phi_arr
        action = int(combined.argmax())

        # Explain the chosen action
        _, matched_rules = self.logos.explain(obs, action)

        return action, q_vals, phi_arr, matched_rules
