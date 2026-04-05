#!/usr/bin/env python
"""
Main HyperKKL training pipeline.

Runs Phase 1 (autonomous KKL) and Phase 2 (non-autonomous methods)
for a given system, saving structured results with configs.

Usage:
    python -m scripts.run_pipeline --system duffing --method all
    python -m scripts.run_pipeline --system lorenz --method lora_lstm
    python -m scripts.run_pipeline --all_systems --parallel
"""

import sys
import os
import argparse
import copy
import json
import multiprocessing
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig, load_config, save_config
from src.systems import create_system
from src.dataset import generate_phase2_data
from src.training import train_phase1, train_augmented, train_curriculum, train_dynamic
from src.evaluation import evaluate_method, get_plot_trajectories
from src.plotting import (plot_loss_history, plot_time_series, plot_attractor,
                          plot_density, plot_boxplot, plot_boxplot_stacked)
from src.logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Per-method train + eval
# ---------------------------------------------------------------------------

ALL_METHODS = ["autonomous", "curriculum", "augmented", "full", "lora"]

DEFAULT_METHODS = ["autonomous", "augmented", "full", "lora"]


def _train_and_eval_method(method, T_net, T_inv_net, system, sys_config, train_data,
                           cfg, device, seed, run_dir, window_size, logger):
    """Train and evaluate a single method.

    Returns (method, results, loss_history, trajectories, per_trial).
    """
    T_copy = copy.deepcopy(T_net)
    T_inv_copy = copy.deepcopy(T_inv_net)

    loss_history = None
    trajectories = None
    per_trial = None
    eval_inputs = sys_config.natural_inputs

    try:
        if method == "curriculum":
            T_curr, T_inv_curr, loss_history = train_curriculum(
                T_copy, T_inv_copy, system, sys_config, cfg, device, logger)
            torch.save({"T_encoder": T_curr.state_dict(), "T_inv_decoder": T_inv_curr.state_dict()},
                       run_dir / f"{method}.pt")
            models = {"type": "autonomous", "T_inv": T_inv_curr}
            eval_res, per_trial = evaluate_method(system, sys_config, models, method,
                                                  eval_inputs, cfg.evaluation.n_trials, seed, device, window_size)
            trajectories = get_plot_trajectories(system, sys_config, models, method,
                                                 device, window_size, seed)

        elif method == "augmented":
            enc, phi, loss_history = train_augmented(
                T_copy, T_inv_copy, sys_config, train_data, cfg, device, logger)
            torch.save({"encoder": enc.state_dict(), "phi": phi.state_dict()},
                       run_dir / f"{method}.pt")
            models = {"type": "augmented", "T_inv": T_inv_copy, "encoder": enc, "phi": phi}
            eval_res, per_trial = evaluate_method(system, sys_config, models, method,
                                                  eval_inputs, cfg.evaluation.n_trials, seed, device, window_size)
            trajectories = get_plot_trajectories(system, sys_config, models, method,
                                                 device, window_size, seed)

        elif method in ("full", "lora"):
            hypernet, T_mod, Tinv_mod, loss_history = train_dynamic(
                T_copy, T_inv_copy, sys_config, train_data, cfg, device, method, logger)
            torch.save({"hypernet": hypernet.state_dict()}, run_dir / f"{method}.pt")
            mtype = "lora" if method == "lora" else "full"
            models = {"type": mtype, "hypernet": hypernet, "T_base": T_mod, "T_inv_base": Tinv_mod}
            eval_res, per_trial = evaluate_method(system, sys_config, models, method,
                                                  eval_inputs, cfg.evaluation.n_trials, seed, device, window_size)
            trajectories = get_plot_trajectories(system, sys_config, models, method,
                                                 device, window_size, seed)
        else:
            return method, {}, None, None, None

        print(f"  [{method}] Results: {eval_res}")
        return method, eval_res, loss_history, trajectories, per_trial

    except Exception as e:
        import traceback
        print(f"  [{method}] FAILED: {e}")
        traceback.print_exc()
        return method, {}, None, None, None


# ---------------------------------------------------------------------------
# Versioned output directory
# ---------------------------------------------------------------------------

def _next_version_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()],
        key=lambda d: int(d.name[1:]),
    )
    next_num = int(existing[-1].name[1:]) + 1 if existing else 1
    return base / f"v{next_num}"


# ---------------------------------------------------------------------------
# Run experiment for one system
# ---------------------------------------------------------------------------

def run_system_experiment(sys_name: str, methods: list, cfg: ExperimentConfig,
                          out_dir: Path, seed: int, device_id: int = 0):
    """Run all experiments for a single system."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#' * 80}")
    print(f"# SYSTEM: {sys_name.upper()} on {device}")
    print(f"{'#' * 80}")

    sys_config = cfg.system
    system = create_system(sys_config)

    # Versioned output
    sys_base = out_dir / sys_name
    run_dir = _next_version_dir(sys_base)
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    print(f"  Results: {run_dir}")

    # Save config with results
    save_config(cfg, run_dir / "config.yaml")

    # Setup logger
    logger = ExperimentLogger(
        log_dir=str(run_dir / "logs"),
        use_tb=True, use_wandb=False)
    logger.log_config(asdict(cfg))

    results = {}
    all_losses = {}
    all_trajectories = {}

    # Phase 1
    T_net, T_inv_net, auto_loss = train_phase1(system, sys_config, cfg, device, logger)
    torch.save({"model": T_net.state_dict()}, run_dir / "T_encoder.pt")
    torch.save({"model": T_inv_net.state_dict()}, run_dir / "T_inv_decoder.pt")

    all_per_trial = {}

    # Evaluate autonomous
    auto_models = {"type": "autonomous", "T_inv": T_inv_net}
    auto_res, auto_pt = evaluate_method(system, sys_config, auto_models, "autonomous",
                                        sys_config.natural_inputs, cfg.evaluation.n_trials, seed, device,
                                        cfg.phase2.window_size)
    results["autonomous"] = auto_res
    all_losses["autonomous"] = auto_loss
    all_per_trial["autonomous"] = auto_pt
    auto_traj = get_plot_trajectories(system, sys_config, auto_models, "autonomous",
                                      device, cfg.phase2.window_size, seed)
    all_trajectories["autonomous"] = auto_traj
    print(f"  [autonomous] Results: {auto_res}")

    # Generate Phase 2 data
    needs_data = any(m != "autonomous" and m != "curriculum" for m in methods)
    train_data = None
    if needs_data:
        train_data = generate_phase2_data(
            sys_config, sys_config.natural_inputs,
            cfg.phase2.n_train_traj, cfg.phase2.window_size, seed)

    # Phase 2 methods
    for method in [m for m in methods if m != "autonomous"]:
        method, eval_res, loss_hist, traj, per_trial = _train_and_eval_method(
            method, T_net, T_inv_net, system, sys_config, train_data,
            cfg, device, seed, run_dir, cfg.phase2.window_size, logger)

        if eval_res:
            results[method] = eval_res
        if loss_hist is not None:
            all_losses[method] = loss_hist
        if traj is not None:
            all_trajectories[method] = traj
        if per_trial is not None:
            all_per_trial[method] = per_trial

    # Generate plots
    print(f"\nGenerating plots for {sys_name}...")
    is_traffic = sys_name == "highway_traffic"

    for mname in results:
        if mname in all_losses:
            plot_loss_history(all_losses[mname], sys_name, mname, plot_dir / f"{mname}_loss.png")

    for input_type in sys_config.natural_inputs:
        sig_dir = plot_dir / input_type
        sig_dir.mkdir(exist_ok=True)
        for mname in results:
            if mname not in all_trajectories or input_type not in all_trajectories[mname]:
                continue
            x_true, x_hat, t = all_trajectories[mname][input_type]
            plot_time_series(x_true, x_hat, t, sys_name, f"{mname} ({input_type})",
                             sig_dir / f"{mname}_timeseries.png")
            plot_attractor(x_true, x_hat, sys_name, f"{mname} ({input_type})",
                           sig_dir / f"{mname}_attractor.png")
            if is_traffic:
                plot_density(x_true, x_hat, t, sys_name, f"{mname} ({input_type})",
                             sig_dir / f"{mname}_density.png")

    # Boxplots
    if all_per_trial:
        for metric in ["rmse_steady", "rmse_total", "max_error"]:
            plot_boxplot(all_per_trial, sys_name, plot_dir / f"boxplot_{metric}.png", metric=metric)
        plot_boxplot_stacked(all_per_trial, sys_name, plot_dir / "boxplot_rmse_smape.png")

    # Save results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    serializable_loss = {}
    for mname, loss_data in all_losses.items():
        if isinstance(loss_data, dict):
            serializable_loss[mname] = {k: v for k, v in loss_data.items()}
        else:
            serializable_loss[mname] = loss_data
    with open(run_dir / "loss_histories.json", "w") as f:
        json.dump(serializable_loss, f, indent=2)

    logger.close()
    return sys_name, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HyperKKL Training Pipeline")
    parser.add_argument("--system", type=str, default="duffing",
                        choices=["duffing", "vdp", "lorenz", "rossler", "fhn", "highway_traffic", "all"])
    parser.add_argument("--method", type=str, default="all",
                        choices=ALL_METHODS + ["all", "default"])
    parser.add_argument("--all_systems", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--config_dir", type=str, default=None)

    # Override common hyperparameters
    parser.add_argument("--epochs_phase1", type=int, default=None)
    parser.add_argument("--epochs_phase2", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    systems = (["duffing", "vdp", "lorenz", "rossler", "fhn", "highway_traffic"]
               if args.all_systems or args.system == "all" else [args.system])

    if args.method == "all":
        methods = DEFAULT_METHODS
    elif args.method == "default":
        methods = DEFAULT_METHODS
    else:
        methods = [args.method]

    # Build overrides from CLI args
    overrides = {}
    if args.epochs_phase1:
        overrides.setdefault("phase1", {})["epochs"] = args.epochs_phase1
    if args.epochs_phase2:
        overrides.setdefault("phase2", {})["epochs"] = args.epochs_phase2
    if args.batch_size:
        overrides.setdefault("phase1", {})["batch_size"] = args.batch_size
        overrides.setdefault("phase2", {})["batch_size"] = args.batch_size * 2
    if args.lr:
        overrides.setdefault("phase1", {})["lr"] = args.lr
        overrides.setdefault("phase2", {})["lr"] = args.lr
    if args.out_dir:
        overrides["output_dir"] = args.out_dir
    if args.seed:
        overrides["seed"] = args.seed
    overrides["methods"] = methods

    all_results = {}

    if args.parallel and len(systems) > 1:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(systems), mp_context=ctx) as executor:
            futures = {}
            for i, sys_name in enumerate(systems):
                cfg = load_config(sys_name, args.config_dir, overrides)
                out_dir = Path(cfg.output_dir)
                future = executor.submit(
                    run_system_experiment, sys_name, methods, cfg,
                    out_dir, cfg.seed + i * 1000, i % n_gpus)
                futures[future] = sys_name

            for future in as_completed(futures):
                name, res = future.result()
                all_results[name] = res
    else:
        for sys_name in systems:
            cfg = load_config(sys_name, args.config_dir, overrides)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            out_dir = Path(cfg.output_dir)
            _, res = run_system_experiment(sys_name, methods, cfg, out_dir, cfg.seed)
            all_results[sys_name] = res

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY RESULTS")
    print(f"{'=' * 80}")
    for sys_name, res in all_results.items():
        cfg = load_config(sys_name, args.config_dir)
        inputs = cfg.system.natural_inputs
        short = [n.replace("traffic_", "") for n in inputs]
        header = f"{'Method':<20}" + "".join(f" {n:>12}" for n in short)
        print(f"\n{sys_name.upper()}:")
        print(header)
        print("-" * len(header))
        for method, mres in res.items():
            vals = [mres.get(it, {}).get("rmse_steady", -1) for it in inputs]
            print(f"{method:<20}" + "".join(f" {v:>12.4f}" for v in vals))

    # Save combined results
    combined_out = Path(overrides.get("output_dir", "./results"))
    combined_out.mkdir(parents=True, exist_ok=True)
    with open(combined_out / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
