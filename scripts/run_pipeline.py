#!/usr/bin/env python
"""
Extended Training and Evaluation Pipeline for HyperKKL.

Supports Phase 1 (autonomous KKL), Phase 2 (non-autonomous HyperKKL),
and modular Phase 1 loading via --phase1_dir.

Usage:
    # Full pipeline (Phase 1 + Phase 2)
    python -m hyperkkl.scripts.run_pipeline --system duffing --method all

    # Skip Phase 1, load pre-trained T and T*
    python -m hyperkkl.scripts.run_pipeline --system duffing --method dynamic_lora \
        --phase1_dir results/hyperkkl/duffing/v1

    # All systems in parallel
    python -m hyperkkl.scripts.run_pipeline --all_systems --method all --parallel
"""

import sys
import os
import argparse
import json
import copy
import multiprocessing
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from hyperkkl.src.training.configs import get_system_config
from hyperkkl.src.models.hypernetworks import (
    WindowEncoder, LSTMEncoder, InputInjectionNet,
    DualHyperNetwork, ResidualHyperNetwork,
    apply_weight_modulation, count_parameters,
)
from hyperkkl.src.data.data_gen import generate_hyperkkl_data_parallel
from hyperkkl.src.training.phase1 import train_autonomous, load_autonomous
from hyperkkl.src.training.phase2 import (
    train_hyperkkl, train_curriculum,
    train_dynamic_hyperkkl, train_dynamic_lora,
)
from hyperkkl.src.evaluation.evaluate import (
    simulate_observer, simulate_observer_dynamic,
    evaluate, evaluate_dynamic,
    get_plot_trajectories, get_plot_trajectories_dynamic,
    evaluate_dynamic_lora, get_plot_trajectories_dynamic_lora,
    plot_boxplot,
)
from hyperkkl.src.evaluation.plot import (
    plot_loss_history, plot_time_series, plot_attractor, plot_density,
)


# ============================================================================
# ORCHESTRATION
# ============================================================================

def _train_and_eval_method(method, T_net, T_inv_net, system, sys_config, train_data,
                           hyper_cfg, curr_cfg, device, seed, sys_dir, window_size,
                           n_eval_trials=100):
    """Train and evaluate a single method."""
    T_copy = copy.deepcopy(T_net)
    T_inv_copy = copy.deepcopy(T_inv_net)

    loss_history = None
    trajectories = None
    per_trial = None

    eval_input_types = sys_config.get(
        'natural_inputs', ['zero', 'constant', 'sinusoid', 'square'])

    try:
        if method == 'curriculum':
            T_curr, T_inv_curr, loss_history = train_curriculum(
                T_copy, T_inv_copy, system, sys_config, curr_cfg, device)
            torch.save(
                {'T_encoder': T_curr.state_dict(),
                 'T_inv_decoder': T_inv_curr.state_dict()},
                sys_dir / f'{method}.pt')
            eval_res, per_trial = evaluate(
                system, T_inv_curr, sys_config,
                eval_input_types, n_eval_trials, seed, device=device)
            trajectories = get_plot_trajectories(
                system, T_inv_curr, sys_config, None, None,
                device, window_size, seed)

        elif method in ['static_window', 'static_lstm']:
            enc_type = method.split('_')[1]
            enc, phi, loss_history = train_hyperkkl(
                T_copy, T_inv_copy, sys_config, train_data,
                hyper_cfg, device, enc_type)
            torch.save(
                {'encoder': enc.state_dict(), 'phi': phi.state_dict()},
                sys_dir / f'{method}.pt')
            eval_res, per_trial = evaluate(
                system, T_inv_copy, sys_config,
                eval_input_types, n_eval_trials, seed, enc, phi, device)
            trajectories = get_plot_trajectories(
                system, T_inv_copy, sys_config, enc, phi,
                device, window_size, seed)

        elif method in ['dynamic_window', 'dynamic_lstm', 'dynamic_gru']:
            enc_type = method.split('_')[1]
            hypernet, T_mod, Tinv_mod, loss_history = train_dynamic_hyperkkl(
                T_copy, T_inv_copy, sys_config, train_data,
                hyper_cfg, device, enc_type)
            torch.save(
                {'hypernet': hypernet.state_dict()},
                sys_dir / f'{method}.pt')
            eval_res, per_trial = evaluate_dynamic(
                system, hypernet, T_mod, Tinv_mod, sys_config,
                eval_input_types, n_eval_trials, seed, device)
            trajectories = get_plot_trajectories_dynamic(
                system, hypernet, T_mod, Tinv_mod, sys_config,
                device, window_size, seed)

        elif method in ['dynamic_lora', 'dynamic_lora_gru']:
            cell = 'gru' if method == 'dynamic_lora_gru' else 'lstm'
            hypernet, T_mod, Tinv_mod, loss_history = train_dynamic_lora(
                T_copy, T_inv_copy, sys_config, train_data,
                hyper_cfg, device, cell)
            torch.save(
                {'hypernet': hypernet.state_dict()},
                sys_dir / f'{method}.pt')
            eval_res, per_trial = evaluate_dynamic_lora(
                system, hypernet, T_mod, Tinv_mod, sys_config,
                eval_input_types, n_eval_trials, seed, device)
            trajectories = get_plot_trajectories_dynamic_lora(
                system, hypernet, T_mod, Tinv_mod, sys_config,
                device, window_size, seed)

        else:
            return method, {}, None, None, None

        print(f"  [{method}] Results: {eval_res}")
        return method, eval_res, loss_history, trajectories, per_trial

    except Exception as e:
        print(f"  [{method}] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return method, {}, None, None, None


def _next_version_dir(sys_dir: Path) -> Path:
    """Determine the next version directory (v1, v2, ...) inside sys_dir."""
    existing = sorted(
        [d for d in sys_dir.iterdir()
         if d.is_dir() and d.name.startswith('v') and d.name[1:].isdigit()],
        key=lambda d: int(d.name[1:]),
    ) if sys_dir.exists() else []
    next_num = int(existing[-1].name[1:]) + 1 if existing else 1
    return sys_dir / f'v{next_num}'


def run_system_experiment(sys_name: str, methods: List[str], train_cfg: dict,
                          hyper_cfg: dict, curr_cfg: dict, out_dir: Path,
                          seed: int, device_id: int = 0,
                          n_eval_trials: int = 100, phase1_dir: str = None):
    """Run all experiments for a single system."""
    device = torch.device(
        f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'#' * 80}")
    print(f"# SYSTEM: {sys_name.upper()} on {device}")
    print(f"{'#' * 80}")

    sys_config = get_system_config(sys_name)
    system = sys_config['class'](**sys_config['init_args'])

    sys_base = out_dir / sys_name
    sys_base.mkdir(exist_ok=True)
    sys_dir = _next_version_dir(sys_base)
    sys_dir.mkdir(exist_ok=True)
    print(f"  Results version: {sys_dir.name}")
    plot_dir = sys_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    results = {}
    all_loss_histories = {}
    all_trajectories = {}
    all_per_trial = {}

    # Phase 1: Load pre-trained or train from scratch
    if phase1_dir is not None:
        T_net, T_inv_net, auto_loss_history = load_autonomous(
            phase1_dir, sys_config, device)
    else:
        T_net, T_inv_net, auto_loss_history = train_autonomous(
            system, sys_config, train_cfg, device)
    torch.save({'model': T_net.state_dict()}, sys_dir / 'T_encoder.pt')
    torch.save({'model': T_inv_net.state_dict()}, sys_dir / 'T_inv_decoder.pt')

    eval_input_types = sys_config.get(
        'natural_inputs', ['zero', 'constant', 'sinusoid', 'square'])

    print(f"\nEvaluating Autonomous ({n_eval_trials} trials)...")
    auto_res, auto_per_trial = evaluate(
        system, T_inv_net, sys_config,
        eval_input_types, n_eval_trials, seed, device=device)
    results['autonomous'] = auto_res
    all_loss_histories['autonomous'] = auto_loss_history
    all_per_trial['autonomous'] = auto_per_trial
    print(f"  Results: {auto_res}")

    auto_trajectories = get_plot_trajectories(
        system, T_inv_net, sys_config, None, None, device,
        hyper_cfg['window_size'], seed)
    all_trajectories['autonomous'] = auto_trajectories

    train_data = None
    needs_train_data = any(
        m in methods for m in [
            'static_window', 'static_lstm',
            'dynamic_window', 'dynamic_lstm', 'dynamic_gru',
            'dynamic_lora', 'dynamic_lora_gru',
        ])
    if needs_train_data:
        train_data = generate_hyperkkl_data_parallel(
            sys_name, sys_config, sys_config['natural_inputs'],
            hyper_cfg['n_train_traj'], hyper_cfg['window_size'], seed)

    methods_to_run = [m for m in methods if m != 'autonomous']
    window_size = hyper_cfg['window_size']

    if methods_to_run:
        print(f"\nRunning {len(methods_to_run)} methods: {methods_to_run}")
        for method in methods_to_run:
            method, eval_res, loss_hist, trajectories, per_trial = \
                _train_and_eval_method(
                    method, T_net, T_inv_net, system, sys_config,
                    train_data, hyper_cfg, curr_cfg, device, seed,
                    sys_dir, window_size, n_eval_trials)

            if eval_res:
                results[method] = eval_res
            if loss_hist is not None:
                all_loss_histories[method] = loss_hist
            if trajectories is not None:
                all_trajectories[method] = trajectories
            if per_trial is not None:
                all_per_trial[method] = per_trial

    # Generate plots
    print(f"\nGenerating plots for {sys_name}...")
    for method_name in results:
        if method_name in all_loss_histories:
            plot_loss_history(
                all_loss_histories[method_name], sys_name, method_name,
                plot_dir / f'{method_name}_loss.png')

    if all_per_trial:
        for metric in ['rmse_total', 'rmse_steady', 'max_error']:
            plot_boxplot(
                all_per_trial, sys_name,
                plot_dir / f'boxplot_{metric}.png', metric=metric)

    is_traffic = sys_name == 'highway_traffic'
    for input_type in eval_input_types:
        signal_dir = plot_dir / input_type
        signal_dir.mkdir(exist_ok=True)
        for method_name in results:
            if method_name not in all_trajectories:
                continue
            method_trajs = all_trajectories[method_name]
            if input_type not in method_trajs:
                continue
            x_true, x_hat, t = method_trajs[input_type]
            plot_time_series(
                x_true, x_hat, t, sys_name,
                f'{method_name} ({input_type})',
                signal_dir / f'{method_name}_timeseries.png')
            plot_attractor(
                x_true, x_hat, sys_name,
                f'{method_name} ({input_type})',
                signal_dir / f'{method_name}_attractor.png')
            if is_traffic:
                plot_density(
                    x_true, x_hat, t, sys_name,
                    f'{method_name} ({input_type})',
                    signal_dir / f'{method_name}_density.png')

    print(f"  Plots saved to {plot_dir}")

    with open(sys_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    serializable_loss = {}
    for method_name, loss_data in all_loss_histories.items():
        if isinstance(loss_data, dict):
            serializable_loss[method_name] = {
                k: v if isinstance(v, list) else v
                for k, v in loss_data.items()
            }
        else:
            serializable_loss[method_name] = loss_data
    with open(sys_dir / 'loss_histories.json', 'w') as f:
        json.dump(serializable_loss, f, indent=2)

    return sys_name, results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extended HyperKKL Pipeline')

    parser.add_argument(
        '--system', type=str, default='duffing',
        choices=['duffing', 'vdp', 'lorenz', 'rossler',
                 'fitzhugh_nagumo', 'highway_traffic', 'all'])
    parser.add_argument(
        '--method', type=str, default='all',
        choices=['autonomous', 'curriculum', 'static_window', 'static_lstm',
                 'dynamic_window', 'dynamic_lstm', 'dynamic_gru',
                 'dynamic_lora', 'dynamic_lora_gru', 'all'])
    parser.add_argument('--all_systems', action='store_true')
    parser.add_argument('--parallel', action='store_true')

    parser.add_argument('--epochs_phase1', type=int, default=20)
    parser.add_argument('--epochs_phase2', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2049)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_ic', type=int, default=200)
    parser.add_argument('--num_hidden', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=150)

    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--n_train_traj', type=int, default=400)
    parser.add_argument('--n_eval_trials', type=int, default=100)

    parser.add_argument('--no_split_traj', action='store_true')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--no_lora_input_gate', action='store_true')
    parser.add_argument('--no_physics_norm', action='store_true')

    parser.add_argument('--phase1_dir', type=str, default=None,
                        help='Path to pre-trained T_encoder.pt / T_inv_decoder.pt '
                             '(skips Phase 1 training)')
    parser.add_argument('--out_dir', type=str, default='./results/hyperkkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = {
        'epochs': args.epochs_phase1, 'batch_size': args.batch_size,
        'lr': args.lr, 'num_ic': args.num_ic,
        'num_hidden': args.num_hidden, 'hidden_size': args.hidden_size,
        'use_pde': True, 'lambda_pde': 1.0, 'seed': args.seed,
        'split_traj': not args.no_split_traj,
        'normalize': not args.no_normalize,
    }

    hyper_cfg = {
        'epochs': args.epochs_phase2, 'batch_size': args.batch_size * 2,
        'lr': args.lr, 'window_size': args.window_size,
        'latent_dim': args.latent_dim, 'n_train_traj': args.n_train_traj,
        'lstm_hidden': 64, 'rank': 32, 'hypernet_hidden': 128,
        'lora_rank': 4,
        'lora_input_gate': not args.no_lora_input_gate,
        'physics_norm': not args.no_physics_norm,
    }

    curr_cfg = {
        'lr': args.lr, 'batch_size': args.batch_size * 2,
        'window_size': args.window_size, 'latent_dim': args.latent_dim,
        'n_traj_per_stage': 50, 'seed': args.seed,
        'stage1_epochs': 1, 'stage2_epochs': 20,
        'stage3_epochs': 30, 'stage4_epochs': 50,
    }

    ALL_SYSTEMS = [
        'duffing', 'vdp', 'lorenz', 'rossler',
        'fitzhugh_nagumo', 'highway_traffic',
    ]
    systems = ALL_SYSTEMS if (args.all_systems or args.system == 'all') \
        else [args.system]
    methods = [
        'autonomous', 'dynamic_lstm', 'static_lstm', 'dynamic_lora',
    ] if args.method == 'all' else [args.method]

    all_results = {}

    if args.parallel and len(systems) > 1:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        max_workers = len(systems)
        spawn_context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=spawn_context) as executor:
            futures = {}
            for i, sys_name in enumerate(systems):
                device_id = i % n_gpus
                future = executor.submit(
                    run_system_experiment, sys_name, methods, train_cfg,
                    hyper_cfg, curr_cfg, out_dir,
                    args.seed + i * 1000, device_id,
                    args.n_eval_trials, args.phase1_dir)
                futures[future] = sys_name
            for future in as_completed(futures):
                sys_name, results = future.result()
                all_results[sys_name] = results
    else:
        for sys_name in systems:
            _, results = run_system_experiment(
                sys_name, methods, train_cfg, hyper_cfg, curr_cfg,
                out_dir, args.seed, 0,
                args.n_eval_trials, args.phase1_dir)
            all_results[sys_name] = results

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY RESULTS")
    print(f"{'=' * 80}")
    for sys_name, res in all_results.items():
        sys_cfg = get_system_config(sys_name)
        sys_input_types = sys_cfg.get(
            'natural_inputs', ['zero', 'constant', 'sinusoid', 'square'])
        short_names = [n.replace('traffic_', '') for n in sys_input_types]
        header = f"{'Method':<20}" + "".join(f" {n:>12}" for n in short_names)
        print(f"\n{sys_name.upper()}:")
        print(header)
        print("-" * len(header))
        for method, mres in res.items():
            vals = [mres.get(it, {}).get('rmse_steady', -1) for it in sys_input_types]
            row = f"{method:<20}" + "".join(f" {v:>12.4f}" for v in vals)
            print(row)

    with open(out_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == '__main__':
    main()
