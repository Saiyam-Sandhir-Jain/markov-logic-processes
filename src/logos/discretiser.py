"""
Feature Discretisation (B)
──────────────────────────
Maps continuous LunarLander observations to Low/Med/High bins using
the hybrid Equal-Width + Equal-Frequency scheme from Section IV-B
of the MLP paper.

Feature layout (LunarLander-v3, 8-dim observation):
  idx 0: x position
  idx 1: y position
  idx 2: x velocity
  idx 3: y velocity
  idx 4: angle (theta)
  idx 5: angular velocity
  idx 6: left leg contact  (binary — excluded from discretisation)
  idx 7: right leg contact (binary — excluded from discretisation)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

# Human-readable names for overlay / debugging
FEATURE_NAMES = ["x", "y", "x_vel", "y_vel", "angle", "ang_vel"]
BIN_LABELS = ["Low", "Med", "High"]
CONTINUOUS_IDX = [0, 1, 2, 3, 4, 5]   # indices of continuous features


class FeatureDiscretiser:
    """
    Hybrid EW+EF discretiser that adapts to both uniform and skewed
    marginal distributions, as described in Eq. (3–5) of the paper.

    Parameters
    ----------
    n_bins : int
        Number of bins (default 3 → Low / Med / High).
    continuous_idx : list[int]
        Indices of continuous features to discretise.
    eps : float
        Small constant for numerical stability in EW scheme.
    """

    def __init__(
        self,
        n_bins: int = 3,
        continuous_idx: List[int] | None = None,
        eps: float = 1e-8,
    ):
        self.n_bins = n_bins
        self.continuous_idx = continuous_idx if continuous_idx is not None else CONTINUOUS_IDX
        self.eps = eps

        # Running statistics for adaptive quantile computation
        self._obs_buffer: List[np.ndarray] = []
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._q33: np.ndarray | None = None
        self._q66: np.ndarray | None = None
        self._fitted = False

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, observations: np.ndarray) -> "FeatureDiscretiser":
        """
        Compute statistics from a batch of observations.

        Parameters
        ----------
        observations : np.ndarray  shape (N, D)
        """
        obs = np.asarray(observations)
        cont = obs[:, self.continuous_idx]
        self._min = cont.min(axis=0)
        self._max = cont.max(axis=0)
        self._q33 = np.percentile(cont, 33, axis=0)
        self._q66 = np.percentile(cont, 66, axis=0)
        self._fitted = True
        return self

    def partial_fit(self, observation: np.ndarray) -> "FeatureDiscretiser":
        """Incrementally update statistics with a single observation."""
        self._obs_buffer.append(np.asarray(observation)[self.continuous_idx])
        if len(self._obs_buffer) >= 200:        # refit every 200 samples
            arr = np.stack(self._obs_buffer)
            self._min = arr.min(axis=0) if self._min is None else np.minimum(self._min, arr.min(axis=0))
            self._max = arr.max(axis=0) if self._max is None else np.maximum(self._max, arr.max(axis=0))
            self._q33 = np.percentile(arr, 33, axis=0)
            self._q66 = np.percentile(arr, 66, axis=0)
            self._fitted = True
            self._obs_buffer.clear()
        return self

    # ── Discretisation ───────────────────────────────────────────────────────

    def _ew_bin(self, values: np.ndarray) -> np.ndarray:
        """Equal-Width binning → Eq. (3)."""
        norm = (values - self._min) / (self._max - self._min + self.eps)
        return np.clip((norm * self.n_bins).astype(int), 0, self.n_bins - 1)

    def _ef_bin(self, values: np.ndarray) -> np.ndarray:
        """Equal-Frequency (quantile) binning → Eq. (4)."""
        bins = np.zeros(len(values), dtype=int)
        bins[values >= self._q33] = 1
        bins[values >= self._q66] = 2
        return bins

    def transform(self, observation: np.ndarray) -> List[Tuple[str, str]]:
        """
        Discretise a single observation.

        Returns
        -------
        list of (feature_name, bin_label) pairs, e.g.:
            [("x", "Med"), ("y", "Low"), ("angle", "High"), ...]
        Only continuous features are returned.
        """
        obs = np.asarray(observation)
        cont = obs[self.continuous_idx]

        if not self._fitted:
            # Fall back to equal-width with heuristic LunarLander ranges
            self._min = np.array([-1.5, -0.5, -2.5, -2.5, -3.14, -5.0])
            self._max = np.array([ 1.5,  1.5,  2.5,  2.5,  3.14,  5.0])
            self._q33 = self._min + (self._max - self._min) / 3
            self._q66 = self._min + 2 * (self._max - self._min) / 3
            self._fitted = True

        ew = self._ew_bin(cont)
        ef = self._ef_bin(cont)
        # Hybrid blend → Eq. (5)
        blended = np.round((ew + ef) / 2).astype(int)
        blended = np.clip(blended, 0, self.n_bins - 1)

        return [
            (FEATURE_NAMES[i], BIN_LABELS[b])
            for i, b in enumerate(blended)
        ]

    def discretise_reward(self, reward: float) -> str:
        """Bin a scalar reward into Low / Med / High."""
        if reward < -50:
            return "Low"
        elif reward <= 50:
            return "Med"
        else:
            return "High"

    def build_item_set(
        self,
        observation: np.ndarray,
        action: int,
        reward: float | None = None,
    ) -> List[str]:
        """
        Build the full item set for one experience tuple, suitable for
        passing to the Apriori miner.

        Format: ["x:Med", "y:Low", ..., "action:2", "reward:High"]
        """
        items = [f"{name}:{label}" for name, label in self.transform(observation)]
        items.append(f"action:{action}")
        if reward is not None:
            items.append(f"reward:{self.discretise_reward(reward)}")
        return items
