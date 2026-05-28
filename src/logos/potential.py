"""
Logic Potential Φ_L and Logos Module Orchestrator
──────────────────────────────────────────────────
Implements Eq. (6) from the MLP paper:

    Φ_L(s, a) = U_L · Σ_r match(r, s, a) · κ_r · min(ρ(r), 5)

and the deductive-mode override (Fix 4):
    When U_L > 0.9, replace Q(s,a) entirely with Φ_L for exploitation.

This module also exposes `explain(obs, action)` which returns the set of
actively-matched rules — used by the video generation notebook (03_video.ipynb)
to render the "inference used" overlay.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from .discretiser import FeatureDiscretiser
from .miner import AsyncAprioriMiner, Rule


# ── Logos orchestrator ────────────────────────────────────────────────────────


class Logos:
    """
    The complete Logos module L = (D, B, F, τ) from Definition 1.

    Wraps the FeatureDiscretiser (B) and AsyncAprioriMiner (F / τ),
    exposes:
        add_experience(obs, action, reward) → updates D, triggers τ
        phi(obs, action)                    → scalar Φ_L(s, a)
        explain(obs, action)                → (phi, list[matched_rule_info])
        deduction_override(q_values)        → rule-derived value if U_L > 0.9

    Parameters
    ----------
    lambda_scale         : float  — λ scaling factor
    window_size          : int    — W
    mine_interval        : int    — Δ
    min_support          : float  — σ
    min_confidence       : float  — κ
    persistence_threshold: int    — ρ_min for "persistent" classification
    persistence_cap      : int    — min(ρ, cap)
    deduction_threshold  : float  — U_L threshold for deductive override
    n_actions            : int    — |A|
    """

    def __init__(
        self,
        lambda_scale: float = 1.5,
        window_size: int = 2_000,
        mine_interval: int = 2_000,
        min_support: float = 0.20,
        min_confidence: float = 0.80,
        persistence_threshold: int = 3,
        persistence_cap: int = 5,
        deduction_threshold: float = 0.90,
        n_actions: int = 4,
    ):
        self.lambda_scale = lambda_scale
        self.persistence_threshold = persistence_threshold
        self.persistence_cap = persistence_cap
        self.deduction_threshold = deduction_threshold
        self.n_actions = n_actions

        self.discretiser = FeatureDiscretiser()
        self.miner = AsyncAprioriMiner(
            window_size=window_size,
            mine_interval=mine_interval,
            min_support=min_support,
            min_confidence=min_confidence,
        )

    # ── Experience ingestion ─────────────────────────────────────────────────

    def add_experience(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
    ) -> None:
        """
        Encode (o_t, a_t, r_t) and push to the miner's sliding window.
        Also updates the discretiser's running statistics.
        """
        self.discretiser.partial_fit(observation)
        item_set = self.discretiser.build_item_set(observation, action, reward)
        self.miner.add_experience(item_set)

    # ── Logic potential ──────────────────────────────────────────────────────

    def phi(self, observation: np.ndarray, action: int) -> float:
        """
        Compute Φ_L(s, a) — the scalar logic potential for a state-action pair.

        Returns lambda_scale * Φ_L.
        """
        phi_raw, _ = self._compute(observation, action)
        return self.lambda_scale * phi_raw

    def explain(
        self,
        observation: np.ndarray,
        action: int,
    ) -> Tuple[float, List[Dict]]:
        """
        Compute Φ_L(s, a) and return the list of matched rules.

        Returns
        -------
        (phi_scaled, matched_rules)
        matched_rules is sorted by contribution descending; each entry:
            {
                "antecedents_str": "x:Med ∧ angle:High",
                "consequents_str": "action:2",
                "confidence":      0.89,
                "persistence":     4,
                "contribution":    3.56,
            }
        """
        phi_raw, matched = self._compute(observation, action, return_rules=True)
        return self.lambda_scale * phi_raw, matched

    def _compute(
        self,
        observation: np.ndarray,
        action: int,
        return_rules: bool = False,
    ) -> Tuple[float, List[Dict]]:
        rules = self.miner.rules
        UL = self.miner.UL

        if not rules:
            return 0.0, []

        # Build item set for this (s, a) — no reward included
        items = set(self.discretiser.build_item_set(observation, action))

        phi_raw = 0.0
        matched: List[Dict] = []

        for rule in rules:
            ant: frozenset = rule["antecedents"]
            # Remove reward: items from antecedent before matching (state-only)
            ant_state = frozenset(i for i in ant if not i.startswith("reward:"))
            if ant_state.issubset(items):
                kappa = rule["confidence"]
                rho   = min(rule["persistence"], self.persistence_cap)
                contribution = kappa * rho
                phi_raw += contribution

                if return_rules:
                    ant_str = " ∧ ".join(sorted(ant_state)) or "(empty)"
                    con_str = " ∨ ".join(sorted(rule["consequents"]))
                    matched.append({
                        "antecedents_str": ant_str,
                        "consequents_str": con_str,
                        "confidence":      kappa,
                        "persistence":     rule["persistence"],
                        "contribution":    contribution,
                    })

        phi_raw *= UL

        if return_rules:
            matched.sort(key=lambda x: x["contribution"], reverse=True)
            matched = matched[:5]    # top-5 rules for overlay

        return phi_raw, matched

    # ── All-action potential (for action selection in MLP-Full) ──────────────

    def phi_all_actions(self, observation: np.ndarray) -> np.ndarray:
        """
        Return Φ_L(s, a) for all actions as a numpy array of shape (n_actions,).
        Used by MLP-Full during exploitation: π(s) = argmax[Q(s,a) + λΦ_L(s,a)].
        """
        return np.array([self.phi(observation, a) for a in range(self.n_actions)])

    # ── Deductive-mode override (Fix 4) ─────────────────────────────────────

    def deduction_override_value(
        self,
        observation: np.ndarray,
        q_values: np.ndarray,
    ) -> np.ndarray:
        """
        When U_L > deduction_threshold, replace Q values with Φ_L.
        Returns potentially-modified q_values array.
        """
        if self.miner.UL > self.deduction_threshold:
            phi_arr = self.phi_all_actions(observation)
            return phi_arr
        return q_values

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def UL(self) -> float:
        return self.miner.UL

    @property
    def n_rules(self) -> int:
        return len(self.miner.rules)

    def shutdown(self) -> None:
        self.miner.shutdown()
