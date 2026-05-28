"""
Asynchronous Apriori Rule Miner (F / τ)
────────────────────────────────────────
Implements the background-thread mining component of the Logos module
(Section IV-C of the MLP paper).

Key design decisions mirrored from the paper:
  • Sliding window W = 2 000 experience tuples
  • Mining every Δ = 2 000 environment steps on a single background thread
  • min_support σ = 0.20, min_confidence κ = 0.80
  • Three post-processing fixes:
      Fix 1 — binary leg-contact features excluded upstream (in discretiser)
      Fix 2 — rules with reward-only antecedents are discarded
      Fix 3 — rules whose consequents contain neither action: nor reward: are removed
  • Rule persistence tracking (Fix 4 threshold used in potential.py)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# ── Type alias ────────────────────────────────────────────────────────────────
Rule = Dict  # keys: antecedents (frozenset), consequents (frozenset),
              #       confidence (float), support (float), persistence (int)


class AsyncAprioriMiner:
    """
    Background Apriori miner that updates the rule set without blocking
    the main DQN training loop.

    Parameters
    ----------
    window_size    : int   — sliding experience window W
    mine_interval  : int   — steps between mining runs Δ
    min_support    : float — Apriori σ
    min_confidence : float — Apriori κ
    """

    def __init__(
        self,
        window_size: int = 2_000,
        mine_interval: int = 2_000,
        min_support: float = 0.20,
        min_confidence: float = 0.80,
    ):
        self.window_size = window_size
        self.mine_interval = mine_interval
        self.min_support = min_support
        self.min_confidence = min_confidence

        # Sliding-window experience database D
        self._window: deque[List[str]] = deque(maxlen=window_size)
        self._step_counter: int = 0

        # Thread-safe rule storage
        self._rules: List[Rule] = []
        self._persistence: Dict[str, int] = {}   # rule_key → count
        self._UL: float = 0.0                    # global confidence estimate
        self._lock = threading.Lock()

        # Background executor
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._mining_future = None
        self._mining_active = False

    # ── Experience ingestion ─────────────────────────────────────────────────

    def add_experience(self, item_set: List[str]) -> None:
        """
        Add one transaction (discretised experience tuple) to the window.
        Triggers an asynchronous mine every Δ steps.
        """
        self._window.append(item_set)
        self._step_counter += 1

        if (
            self._step_counter % self.mine_interval == 0
            and len(self._window) >= 50          # need enough data
            and not self._mining_active
        ):
            self._mining_active = True
            self._mining_future = self._executor.submit(self._run_mining)

    # ── Mining logic ─────────────────────────────────────────────────────────

    def _run_mining(self) -> None:
        """
        Background thread: run Apriori on a snapshot of the window,
        post-process rules, and update persistence counts.
        """
        try:
            snapshot = list(self._window)   # thread-safe snapshot
            new_rules = self._mine(snapshot)
            self._update_rules(new_rules)
        except Exception:
            pass
        finally:
            self._mining_active = False

    def _mine(self, transactions: List[List[str]]) -> List[Rule]:
        """Run Apriori + post-processing filters on `transactions`."""
        if len(transactions) < 20:
            return []

        # Encode transactions
        te = TransactionEncoder()
        try:
            te_array = te.fit_transform(transactions)
        except Exception:
            return []

        df = pd.DataFrame(te_array, columns=te.columns_)

        # Apriori frequent itemsets
        try:
            freq = apriori(
                df,
                min_support=self.min_support,
                use_colnames=True,
                verbose=0,
            )
        except Exception:
            return []

        if freq.empty:
            return []

        # Association rules
        try:
            rules_df = association_rules(
                freq,
                metric="confidence",
                min_threshold=self.min_confidence,
            )
        except Exception:
            return []

        if rules_df.empty:
            return []

        # ── Post-processing filters ───────────────────────────────────────

        valid: List[Rule] = []
        for _, row in rules_df.iterrows():
            ant: frozenset = row["antecedents"]
            con: frozenset = row["consequents"]

            # Fix 2: discard rules whose antecedents contain ONLY reward: items
            ant_non_reward = {i for i in ant if not i.startswith("reward:")}
            if not ant_non_reward:
                continue

            # Fix 3: keep only rules whose consequent includes action: or reward:
            con_action  = any(i.startswith("action:")  for i in con)
            con_reward  = any(i.startswith("reward:")  for i in con)
            if not (con_action or con_reward):
                continue

            valid.append({
                "antecedents":  ant,
                "consequents":  con,
                "confidence":   float(row["confidence"]),
                "support":      float(row["support"]),
                "lift":         float(row.get("lift", 1.0)),
                "persistence":  0,     # filled in by _update_rules
            })

        return valid

    def _rule_key(self, rule: Rule) -> str:
        """Stable string key for a rule (for persistence tracking)."""
        ant = ",".join(sorted(rule["antecedents"]))
        con = ",".join(sorted(rule["consequents"]))
        return f"{ant}=>{con}"

    def _update_rules(self, new_rules: List[Rule]) -> None:
        """
        Merge new rules with persistence counts and compute U_L.
        Acquires the write lock only for the final assignment.
        """
        new_keys = {self._rule_key(r): r for r in new_rules}

        # Update persistence
        updated_persistence: Dict[str, int] = {}
        for key, rule in new_keys.items():
            prev = self._persistence.get(key, 0)
            updated_persistence[key] = prev + 1
            rule["persistence"] = updated_persistence[key]

        # U_L: fraction of new rules that are persistent (ρ ≥ threshold)
        if new_rules:
            persistent_count = sum(
                1 for k in new_keys if updated_persistence[k] >= 3
            )
            ul = persistent_count / len(new_rules)
        else:
            ul = 0.0

        with self._lock:
            self._rules = list(new_keys.values())
            self._persistence = updated_persistence
            self._UL = ul

    # ── Public accessors ─────────────────────────────────────────────────────

    @property
    def rules(self) -> List[Rule]:
        with self._lock:
            return list(self._rules)

    @property
    def UL(self) -> float:
        with self._lock:
            return self._UL

    @property
    def step_count(self) -> int:
        return self._step_counter

    def shutdown(self) -> None:
        """Gracefully shut down the background executor."""
        self._executor.shutdown(wait=False)
