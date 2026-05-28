"""
Unit tests for the Logos module components.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_obs(seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, 8)
    obs[6] = float(rng.integers(0, 2))   # binary leg contacts
    obs[7] = float(rng.integers(0, 2))
    return obs


# ── FeatureDiscretiser ────────────────────────────────────────────────────────

class TestDiscretiser:
    def setup_method(self):
        import sys; sys.path.insert(0, "src")
        from logos.discretiser import FeatureDiscretiser
        self.disc = FeatureDiscretiser()

    def test_transform_returns_six_pairs(self):
        obs = make_obs()
        result = self.disc.transform(obs)
        assert len(result) == 6

    def test_bin_labels_valid(self):
        obs = make_obs()
        result = self.disc.transform(obs)
        for name, label in result:
            assert label in ("Low", "Med", "High"), f"Bad label: {label}"

    def test_itemset_includes_action(self):
        obs = make_obs(); action = 2
        items = self.disc.build_item_set(obs, action)
        assert "action:2" in items

    def test_itemset_includes_reward(self):
        obs = make_obs()
        items = self.disc.build_item_set(obs, 0, reward=100.0)
        assert "reward:High" in items

    def test_reward_bins(self):
        disc = self.disc
        assert disc.discretise_reward(-100) == "Low"
        assert disc.discretise_reward(0)    == "Med"
        assert disc.discretise_reward(200)  == "High"

    def test_partial_fit_does_not_crash(self):
        for i in range(250):
            self.disc.partial_fit(make_obs(i))


# ── Logos module (smoke tests — no real mining needed) ────────────────────────

class TestLogosSmoke:
    def setup_method(self):
        import sys; sys.path.insert(0, "src")
        from logos.potential import Logos
        self.logos = Logos(window_size=100, mine_interval=50)

    def teardown_method(self):
        self.logos.shutdown()

    def test_phi_zero_before_mining(self):
        obs = make_obs(); phi = self.logos.phi(obs, 0)
        assert isinstance(phi, float)

    def test_phi_all_correct_shape(self):
        obs = make_obs(); phi_arr = self.logos.phi_all_actions(obs)
        assert phi_arr.shape == (4,)

    def test_explain_returns_tuple(self):
        obs = make_obs(); result = self.logos.explain(obs, 0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_add_experience_increments_window(self):
        obs = make_obs()
        for i in range(10):
            self.logos.add_experience(obs, i % 4, float(i * 10 - 30))


# ── Stats utilities ───────────────────────────────────────────────────────────

class TestStats:
    def setup_method(self):
        import sys; sys.path.insert(0, "src")
        from utils.stats import (summary_stats, welch_ttest, compute_ma,
                                   episodes_to_solve, post_convergence_stats)
        self.ss  = summary_stats
        self.wt  = welch_ttest
        self.cma = compute_ma
        self.ets = episodes_to_solve
        self.pcs = post_convergence_stats

    def test_summary_stats_basic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = self.ss(vals)
        assert abs(s["mean"] - 3.0) < 1e-6
        assert s["min"] == 1.0

    def test_welch_stars_significant(self):
        a = [100.0] * 10; b = [200.0] * 10
        res = self.wt(a, b)
        assert res["stars"] in ("*", "**", "***")

    def test_compute_ma_length(self):
        rewards = list(range(200))
        ma = self.cma(rewards, window=100)
        assert len(ma) == 101   # 200 - 100 + 1

    def test_episodes_to_solve_found(self):
        rewards = [-100.0] * 100 + [220.0] * 200
        ep = self.ets(rewards)
        assert np.isfinite(ep)

    def test_episodes_to_solve_not_found(self):
        rewards = [100.0] * 500   # below threshold
        ep = self.ets(rewards)
        assert ep == float("inf")

    def test_post_conv_stats_returns_none_if_unsolved(self):
        rewards = [50.0] * 300
        m, s, mn = self.pcs(rewards)
        assert m is None
