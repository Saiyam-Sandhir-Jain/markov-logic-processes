"""
Checkpoint & Session Management
────────────────────────────────
Kaggle-aware checkpointing system.  All persistent state lives under
CKPT_DIR (defaults to checkpoints/ inside the Kaggle working directory).

Structure
─────────
  checkpoints/
    seed_ckpts/{exp_key}__{agent}__{seed}.json   ← per-seed result
    models/{agent}_{env}_{seed}.pt               ← Q-network weights
    run_log.jsonl                                 ← append-only audit log
    results_raw.json                              ← aggregated (rebuilt on load)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ── Paths ─────────────────────────────────────────────────────────────────────

def get_base_dir() -> Path:
    kaggle = Path("/kaggle/working")
    return kaggle if kaggle.exists() else Path(".")


def make_dirs(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    base = base_dir or get_base_dir()
    paths = {
        "ckpt":   base / "checkpoints",
        "seeds":  base / "checkpoints" / "seed_ckpts",
        "models": base / "checkpoints" / "models",
        "figs":   base / "checkpoints" / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (frozenset, set)):
        return sorted(list(obj))
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj).__name__}")


def _seed_ckpt_path(paths: Dict[str, Path], exp_key: str, agent: str, seed: int) -> Path:
    safe_agent = agent.replace("-", "_")
    return paths["seeds"] / f"{exp_key}__{safe_agent}__seed{seed}.json"


# ── Save / load individual seed results ──────────────────────────────────────

def save_seed_result(
    paths: Dict[str, Path],
    exp_key: str,
    agent: str,
    seed: int,
    result: Dict,
) -> None:
    path = _seed_ckpt_path(paths, exp_key, agent, seed)
    with open(path, "w") as f:
        json.dump(result, f, default=_json_default)

    # Append to run log
    entry = {
        "ts":             time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exp_key":        exp_key,
        "agent":          agent,
        "seed":           seed,
        "final_ma100":    result.get("final_ma100"),
        "eval_mean":      result.get("eval_mean"),
        "ep_to_solve":    result.get("ep_to_solve"),
        "train_time_min": result.get("train_time_min"),
    }
    log_path = paths["ckpt"] / "run_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=_json_default) + "\n")


def load_all_seed_results(paths: Dict[str, Path]) -> Dict:
    """Scan seed_ckpts/ and rebuild the full results dict."""
    all_results: Dict = {}
    for p in sorted(paths["seeds"].glob("*.json")):
        name = p.stem                     # e.g. ll_std__MLP_Full__seed42
        parts = name.split("__")
        if len(parts) != 3:
            continue
        exp_key, safe_agent, seed_part = parts
        agent = safe_agent.replace("_", "-")
        seed  = int(seed_part.replace("seed", ""))

        with open(p) as f:
            result = json.load(f)

        all_results.setdefault(exp_key, {}).setdefault(agent, {})[seed] = result

    return all_results


def is_done(
    paths: Dict[str, Path],
    exp_key: str,
    agent: str,
    seed: int,
) -> bool:
    return _seed_ckpt_path(paths, exp_key, agent, seed).exists()


# ── Model weight persistence ──────────────────────────────────────────────────

def save_model(
    paths: Dict[str, Path],
    state_dict: Dict,
    agent: str,
    env_key: str,
    seed: int,
    tag: str = "",
) -> Path:
    """Save Q-network weights to checkpoints/models/."""
    safe_agent = agent.replace("-", "_")
    suffix = f"_{tag}" if tag else ""
    filename = f"{safe_agent}_{env_key}_seed{seed}{suffix}.pt"
    out = paths["models"] / filename
    torch.save(state_dict, out)
    return out


def save_best_model(
    paths: Dict[str, Path],
    state_dict: Dict,
    agent: str,
    env_key: str,
) -> Path:
    """Save the best-seed model as {agent}_{env}_best.pt."""
    safe_agent = agent.replace("-", "_")
    filename = f"{safe_agent}_{env_key}_best.pt"
    out = paths["models"] / filename
    torch.save(state_dict, out)
    return out


def load_model(path: Path, device: Optional[torch.device] = None) -> Dict:
    device = device or torch.device("cpu")
    return torch.load(path, map_location=device, weights_only=True)


# ── Progress dashboard ────────────────────────────────────────────────────────

def print_progress(
    all_results: Dict,
    experiment_keys: List[str],
    agent_types: List[str],
    seeds: List[int],
) -> None:
    total = len(experiment_keys) * len(agent_types) * len(seeds)
    done  = sum(
        1
        for ek in experiment_keys
        for at in agent_types
        for s  in seeds
        if s in all_results.get(ek, {}).get(at, {})
    )
    print(f"Progress: {done}/{total} seed runs complete")
    print()
    for ek in experiment_keys:
        print(f"  {ek}:")
        for at in agent_types:
            completed_seeds = sorted(all_results.get(ek, {}).get(at, {}).keys())
            remaining = [s for s in seeds if s not in completed_seeds]
            status = "✅" if not remaining else f"⏳ remaining: {remaining}"
            print(f"    {at:<15} {status}")
    print()
