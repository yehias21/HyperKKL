#!/usr/bin/env python
"""
Hyperparameter search for HyperKKL.

Supports grid search and random search over training hyperparameters.
Results are saved in structured directories with configs for reproducibility.

Usage:
    python -m scripts.sweep --system duffing --method lora --n_trials 10
    python -m scripts.sweep --system duffing --sweep_config configs/sweep_example.yaml
"""

import sys
import argparse
import itertools
import json
import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, load_config, save_config
from src.systems import create_system
from src.dataset import generate_phase2_data
from src.training import train_phase1, train_augmented, train_dynamic
from src.evaluation import evaluate_method
from src.logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE = {
    "phase2.lr": [1e-4, 5e-4, 1e-3, 2e-3],
    "phase2.lora_rank": [2, 4, 8],
    "phase2.rnn_hidden": [32, 64, 128],
    "phase2.latent_dim": [16, 32, 64],
    "phase1.lr": [5e-4, 1e-3, 2e-3],
    "phase1.lambda_pde": [0.5, 1.0, 2.0],
}


def load_search_space(path: str = None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return DEFAULT_SEARCH_SPACE


# ---------------------------------------------------------------------------
# Apply flat overrides to config
# ---------------------------------------------------------------------------

def _apply_flat_override(cfg_dict: dict, key: str, value):
    """Apply a dot-separated key override to a nested dict."""
    parts = key.split(".")
    d = cfg_dict
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def grid_search(search_space: dict):
    """Generate all combinations from the search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def random_search(search_space: dict, n_trials: int, seed: int = 42):
    """Randomly sample from the search space."""
    rng = np.random.RandomState(seed)
    for _ in range(n_trials):
        trial = {}
        for key, values in search_space.items():
            trial[key] = values[rng.randint(len(values))]
        yield trial


# ---------------------------------------------------------------------------
# Run one trial
# ---------------------------------------------------------------------------

def run_trial(trial_id: int, hp_overrides: dict, base_cfg: ExperimentConfig,
              sys_name: str, method: str, out_dir: Path, device: torch.device):
    """Run a single hyperparameter trial."""
    # Apply overrides
    cfg_dict = asdict(base_cfg)
    for key, value in hp_overrides.items():
        _apply_flat_override(cfg_dict, key, value)

    cfg = load_config(sys_name, overrides=cfg_dict)

    trial_dir = out_dir / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save trial config
    save_config(cfg, trial_dir / "config.yaml")
    with open(trial_dir / "hp_overrides.json", "w") as f:
        json.dump(hp_overrides, f, indent=2)

    logger = ExperimentLogger(str(trial_dir / "logs"), use_tb=True, use_wandb=False)

    sys_config = cfg.system
    system = create_system(sys_config)

    # Phase 1
    T, T_inv, p1_loss = train_phase1(system, sys_config, cfg, device, logger)
    torch.save({"model": T.state_dict()}, trial_dir / "T_encoder.pt")
    torch.save({"model": T_inv.state_dict()}, trial_dir / "T_inv_decoder.pt")

    # Phase 2
    if method != "autonomous":
        train_data = generate_phase2_data(
            sys_config, sys_config.natural_inputs,
            cfg.phase2.n_train_traj, cfg.phase2.window_size, cfg.seed)

        T_copy = copy.deepcopy(T)
        T_inv_copy = copy.deepcopy(T_inv)

        if method == "augmented":
            enc, phi, p2_loss = train_augmented(T_copy, T_inv_copy, sys_config, train_data,
                                                cfg, device, logger)
            torch.save({"encoder": enc.state_dict(), "phi": phi.state_dict()},
                       trial_dir / f"{method}.pt")
            models = {"type": "augmented", "T_inv": T_inv_copy, "encoder": enc, "phi": phi}
        else:
            hypernet, T_mod, Tinv_mod, p2_loss = train_dynamic(
                T_copy, T_inv_copy, sys_config, train_data, cfg, device, method, logger)
            torch.save({"hypernet": hypernet.state_dict()}, trial_dir / f"{method}.pt")
            mtype = "lora" if method == "lora" else "full"
            models = {"type": mtype, "hypernet": hypernet, "T_base": T_mod, "T_inv_base": Tinv_mod}
    else:
        models = {"type": "autonomous", "T_inv": T_inv}
        p2_loss = []

    # Evaluate
    eval_res, _ = evaluate_method(
        system, sys_config, models, method,
        sys_config.natural_inputs, cfg.evaluation.n_trials, cfg.seed, device,
        cfg.phase2.window_size)

    # Compute aggregate score (mean RMSE across all signal types)
    all_rmse = [v.get("rmse_steady", float("inf"))
                for v in eval_res.values() if isinstance(v, dict)]
    mean_rmse = np.mean(all_rmse) if all_rmse else float("inf")

    result = {
        "trial_id": trial_id,
        "hp_overrides": hp_overrides,
        "mean_rmse": float(mean_rmse),
        "eval_results": eval_res,
        "phase1_final_loss": p1_loss["encoder"][-1] if p1_loss.get("encoder") else None,
        "phase2_final_loss": p2_loss[-1] if p2_loss else None,
    }

    with open(trial_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.close()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HyperKKL Hyperparameter Search")
    parser.add_argument("--system", default="duffing")
    parser.add_argument("--method", default="lora")
    parser.add_argument("--search", default="random", choices=["grid", "random"])
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--sweep_config", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./results/sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_cfg = load_config(args.system)
    search_space = load_search_space(args.sweep_config)

    out_dir = Path(args.out_dir) / args.system / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save search space
    with open(out_dir / "search_space.yaml", "w") as f:
        yaml.dump(search_space, f)

    if args.search == "grid":
        trials = list(grid_search(search_space))
        print(f"Grid search: {len(trials)} trials")
    else:
        trials = list(random_search(search_space, args.n_trials, args.seed))
        print(f"Random search: {len(trials)} trials")

    all_results = []
    best_rmse = float("inf")
    best_trial = None

    for i, hp in enumerate(trials):
        print(f"\n{'=' * 60}")
        print(f"TRIAL {i + 1}/{len(trials)}: {hp}")
        print(f"{'=' * 60}")

        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)

        result = run_trial(i, hp, base_cfg, args.system, args.method, out_dir, device)
        all_results.append(result)

        if result["mean_rmse"] < best_rmse:
            best_rmse = result["mean_rmse"]
            best_trial = result
            print(f"  *** New best! RMSE = {best_rmse:.6f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total trials: {len(all_results)}")
    print(f"Best trial: #{best_trial['trial_id']}")
    print(f"Best RMSE: {best_rmse:.6f}")
    print(f"Best HPs: {best_trial['hp_overrides']}")

    # Save summary
    sorted_results = sorted(all_results, key=lambda r: r["mean_rmse"])
    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump({
            "best_trial": best_trial,
            "all_trials": sorted_results,
        }, f, indent=2)

    # Leaderboard
    print(f"\nLeaderboard (top 5):")
    print(f"{'Rank':<6} {'Trial':<8} {'RMSE':<12} {'HPs'}")
    for rank, r in enumerate(sorted_results[:5], 1):
        print(f"{rank:<6} {r['trial_id']:<8} {r['mean_rmse']:<12.6f} {r['hp_overrides']}")


if __name__ == "__main__":
    main()
