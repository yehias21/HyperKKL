#!/usr/bin/env python
"""
Checkpoint Evaluation Script for HyperKKL.

Given a results version directory (e.g. results/hyperkkl/duffing/v1/),
this script:
  1. Loads all available method checkpoints (T, T*, hypernetworks).
  2. For each input signal, generates overlay time-series plots
     (ground truth = solid black, methods = coloured dashed lines).
  3. Produces per-signal metric tables (steady RMSE and SMAPE).

Usage:
    python -m hyperkkl.scripts.evaluate --results_dir results/hyperkkl/duffing/v1
    python -m hyperkkl.scripts.evaluate --results_dir results/hyperkkl  # all systems
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperkkl.src.training.configs import get_system_config
from hyperkkl.src.data.signals import get_input_generator
from hyperkkl.src.models.nn import NN
from hyperkkl.src.models.hypernetworks import (
    WindowEncoder, LSTMEncoder, InputInjectionNet,
    DualHyperNetwork, ResidualHyperNetwork,
    apply_weight_modulation, count_parameters,
    PerLayerLoRAHyperNetwork, get_layer_sizes,
    count_weight_parameters, apply_weight_modulation_skip_bias,
)
from hyperkkl.src.evaluation.evaluate import (
    simulate_observer,
    simulate_observer_dynamic,
    simulate_observer_dynamic_lora,
)


# ============================================================================
# Dummy Normalizer (mirrors buffer names so state_dict loads correctly)
# ============================================================================

class _DummySystem:
    def __init__(self, x_size, z_size):
        self.x_size = x_size
        self.z_size = z_size


class DummyNormalizer(nn.Module):
    """Placeholder Normalizer whose buffers are overwritten by load_state_dict."""
    def __init__(self, x_size, z_size):
        super().__init__()
        self.x_size = x_size
        self.z_size = z_size
        self.sys = _DummySystem(x_size, z_size)
        self.register_buffer('mean_x', torch.zeros(x_size))
        self.register_buffer('std_x', torch.ones(x_size))
        self.register_buffer('mean_z', torch.zeros(z_size))
        self.register_buffer('std_z', torch.ones(z_size))
        self.register_buffer('mean_x_ph', torch.zeros(x_size))
        self.register_buffer('std_x_ph', torch.ones(x_size))
        self.register_buffer('mean_z_ph', torch.zeros(z_size))
        self.register_buffer('std_z_ph', torch.ones(z_size))

    def check_sys(self, tensor, mode):
        if tensor.size()[1] == self.sys.x_size:
            return (self.mean_x_ph, self.std_x) if mode == 'physics' else (self.mean_x, self.std_x)
        elif tensor.size()[1] == self.sys.z_size:
            return (self.mean_z_ph, self.std_z_ph) if mode == 'physics' else (self.mean_z, self.std_z)
        raise RuntimeError('Size of tensor unmatched with any system.')

    def Normalize(self, tensor, mode):
        mean, std = self.check_sys(tensor, mode)
        return (tensor - mean) / std

    def Denormalize(self, tensor, mode):
        mean, std = self.check_sys(tensor, mode)
        return tensor * std + mean


# ============================================================================
# Model reconstruction helpers
# ============================================================================

def _build_nn(num_hidden, hidden_size, in_size, out_size, x_size, z_size):
    norm = DummyNormalizer(x_size, z_size)
    return NN(num_hidden, hidden_size, in_size, out_size, F.relu, normalizer=norm)


def _load_state_dict_flexible(model: nn.Module, state_dict: dict):
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)


def load_T_and_Tinv(version_dir: Path, sys_config: dict, device: torch.device):
    """Load Phase-1 encoder T and decoder T* from a version directory."""
    num_hidden = sys_config['num_hidden']
    hidden_size = sys_config['hidden_size']
    x_size = sys_config['x_size']
    z_size = sys_config['z_size']

    T_net = _build_nn(num_hidden, hidden_size, x_size, z_size, x_size, z_size).to(device)
    T_inv_net = _build_nn(num_hidden, hidden_size, z_size, x_size, x_size, z_size).to(device)

    ckpt_T = torch.load(version_dir / 'T_encoder.pt', map_location=device, weights_only=False)
    ckpt_Tinv = torch.load(version_dir / 'T_inv_decoder.pt', map_location=device, weights_only=False)
    _load_state_dict_flexible(T_net, ckpt_T['model'])
    _load_state_dict_flexible(T_inv_net, ckpt_Tinv['model'])
    T_net.eval()
    T_inv_net.eval()
    return T_net, T_inv_net


# ============================================================================
# Method detection and loading
# ============================================================================

def _detect_methods(version_dir: Path) -> List[str]:
    method_files = {
        'autonomous': 'T_encoder.pt', 'curriculum': 'curriculum.pt',
        'static_window': 'static_window.pt', 'static_lstm': 'static_lstm.pt',
        'dynamic_window': 'dynamic_window.pt', 'dynamic_lstm': 'dynamic_lstm.pt',
        'dynamic_gru': 'dynamic_gru.pt', 'dynamic_lora': 'dynamic_lora.pt',
        'dynamic_lora_gru': 'dynamic_lora_gru.pt',
    }
    return [m for m, f in method_files.items() if (version_dir / f).exists()]


def _remap_legacy_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = re.sub(r'^gru_encoder\.gru\.', 'rnn.', key)
        new_key = re.sub(r'^lstm_encoder\.lstm\.', 'rnn.', new_key)
        remapped[new_key] = value
    return remapped


def _infer_cell_type_from_state_dict(state_dict, hidden_dim):
    for key, value in state_dict.items():
        if key.endswith('.weight_ih_l0'):
            gate_size = value.shape[0]
            if gate_size == 3 * hidden_dim:
                return 'gru'
            elif gate_size == 4 * hidden_dim:
                return 'lstm'
    return None


def _load_method_models(method, version_dir, sys_config, T_net, T_inv_net, device, hyper_cfg):
    import copy
    T_copy = copy.deepcopy(T_net)
    T_inv_copy = copy.deepcopy(T_inv_net)
    window_size = hyper_cfg['window_size']
    latent_dim = hyper_cfg['latent_dim']

    if method == 'autonomous':
        return {'type': 'static', 'T_inv': T_inv_copy, 'encoder': None, 'phi': None}

    elif method == 'curriculum':
        ckpt = torch.load(version_dir / 'curriculum.pt', map_location=device, weights_only=False)
        _load_state_dict_flexible(T_copy, ckpt['T_encoder'])
        _load_state_dict_flexible(T_inv_copy, ckpt['T_inv_decoder'])
        T_copy.eval(); T_inv_copy.eval()
        return {'type': 'static', 'T_inv': T_inv_copy, 'encoder': None, 'phi': None}

    elif method in ('static_window', 'static_lstm'):
        enc_type = method.split('_')[1]
        enc = (WindowEncoder(window_size, 1, latent_dim) if enc_type == 'window'
               else LSTMEncoder(1, hyper_cfg.get('lstm_hidden', 64), latent_dim)).to(device)
        phi = InputInjectionNet(sys_config['z_size'], latent_dim, 1).to(device)
        ckpt = torch.load(version_dir / f'{method}.pt', map_location=device, weights_only=False)
        enc.load_state_dict(ckpt['encoder']); phi.load_state_dict(ckpt['phi'])
        enc.eval(); phi.eval()
        return {'type': 'static', 'T_inv': T_inv_copy, 'encoder': enc, 'phi': phi}

    elif method in ('dynamic_window', 'dynamic_lstm', 'dynamic_gru'):
        enc_type = method.split('_')[1]
        theta_enc = count_parameters(T_copy)
        theta_dec = count_parameters(T_inv_copy)
        ckpt = torch.load(version_dir / f'{method}.pt', map_location=device, weights_only=False)
        sd = _remap_legacy_state_dict(ckpt['hypernet'])
        if enc_type == 'window':
            input_enc = WindowEncoder(window_size, 1, latent_dim).to(device)
            hypernet = DualHyperNetwork(
                input_encoder=input_enc, latent_dim=latent_dim,
                encoder_theta_size=theta_enc, decoder_theta_size=theta_dec,
                rank=hyper_cfg.get('rank', 32),
                shared_hidden_dim=hyper_cfg.get('hypernet_hidden', 128),
            ).to(device)
        else:
            rnn_hidden = hyper_cfg.get('lstm_hidden', 64)
            cell_type = _infer_cell_type_from_state_dict(sd, rnn_hidden) or (
                'gru' if enc_type == 'gru' else 'lstm')
            hypernet = ResidualHyperNetwork(
                n_u=1, hidden_dim=rnn_hidden,
                encoder_theta_size=theta_enc, decoder_theta_size=theta_dec,
                mlp_hidden_dim=hyper_cfg.get('hypernet_hidden', 128),
                scale_init=0.01, cell_type=cell_type,
            ).to(device)
        hypernet.load_state_dict(sd); hypernet.eval()
        for p in T_copy.parameters(): p.requires_grad = False
        for p in T_inv_copy.parameters(): p.requires_grad = False
        return {'type': 'dynamic', 'hypernet': hypernet,
                'T_base': T_copy, 'T_inv_base': T_inv_copy}

    elif method in ('dynamic_lora', 'dynamic_lora_gru'):
        rnn_hidden = hyper_cfg.get('lstm_hidden', 64)
        enc_sizes = get_layer_sizes(T_copy)
        dec_sizes = get_layer_sizes(T_inv_copy)
        ckpt = torch.load(version_dir / f'{method}.pt', map_location=device, weights_only=False)
        sd = _remap_legacy_state_dict(ckpt['hypernet'])
        cell_type = _infer_cell_type_from_state_dict(sd, rnn_hidden) or (
            'gru' if method == 'dynamic_lora_gru' else 'lstm')
        hypernet = PerLayerLoRAHyperNetwork(
            n_u=1, lstm_hidden_dim=rnn_hidden,
            enc_layer_sizes=enc_sizes, dec_layer_sizes=dec_sizes,
            rank=hyper_cfg.get('lora_rank', 4),
            mlp_hidden_dim=hyper_cfg.get('hypernet_hidden', 128),
            scale_init=0.01, cell_type=cell_type,
        ).to(device)
        hypernet.load_state_dict(sd); hypernet.eval()
        for p in T_copy.parameters(): p.requires_grad = False
        for p in T_inv_copy.parameters(): p.requires_grad = False
        return {'type': 'dynamic_lora', 'hypernet': hypernet,
                'T_base': T_copy, 'T_inv_base': T_inv_copy}

    raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Simulation dispatcher
# ============================================================================

def _simulate_method(method_name, models, system, sys_config, ic, input_func,
                     device, window_size=100):
    if models['type'] == 'static':
        return simulate_observer(
            system, models['T_inv'], sys_config, ic, input_func,
            models['encoder'], models['phi'], device, window_size)
    elif models['type'] == 'dynamic':
        return simulate_observer_dynamic(
            system, models['hypernet'], models['T_base'],
            models['T_inv_base'], sys_config, ic, input_func, device, window_size)
    elif models['type'] == 'dynamic_lora':
        return simulate_observer_dynamic_lora(
            system, models['hypernet'], models['T_base'],
            models['T_inv_base'], sys_config, ic, input_func, device, window_size)
    raise ValueError(f"Unknown model type: {models['type']}")


# ============================================================================
# Metrics
# ============================================================================

def compute_smape(x_true, x_hat):
    norm_true = np.linalg.norm(x_true, axis=1)
    norm_hat = np.linalg.norm(x_hat, axis=1)
    norm_diff = np.linalg.norm(x_true - x_hat, axis=1)
    denom = norm_true + norm_hat
    safe_denom = np.where(denom > 0, denom, 1.0)
    smape = 2.0 * norm_diff / safe_denom * 100.0
    return float(np.mean(np.where(denom > 0, smape, 0.0)))


def compute_metrics_multi_trial(results_per_trial, settle_time=5.0):
    rmse_list, smape_list = [], []
    for x_true, x_hat, t in results_per_trial:
        si = np.searchsorted(t, settle_time)
        error = np.linalg.norm(x_true[si:] - x_hat[si:], axis=1)
        rmse_list.append(float(np.sqrt(np.mean(error ** 2))))
        smape_list.append(compute_smape(x_true[si:], x_hat[si:]))
    return {'rmse_steady': float(np.mean(rmse_list)),
            'smape_steady': float(np.mean(smape_list))}


# ============================================================================
# Plotting
# ============================================================================

METHOD_COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

METHOD_DISPLAY_NAMES = {
    'autonomous': 'Autonomous', 'curriculum': 'Curriculum',
    'static_window': 'Static (Window)', 'static_lstm': 'Static (LSTM)',
    'dynamic_window': 'Dynamic (Window)', 'dynamic_lstm': 'Dynamic (LSTM)',
    'dynamic_gru': 'Dynamic (GRU)', 'dynamic_lora': 'Dynamic (LoRA)',
    'dynamic_lora_gru': 'Dynamic (LoRA-GRU)',
}


def plot_overlay_timeseries(trajectories, sys_name, input_type, mode_label, save_path):
    first_method = next(iter(trajectories))
    x_true, _, t = trajectories[first_method]
    n_states = x_true.shape[1]
    state_labels = [f'$x_{{{i+1}}}$' for i in range(n_states)]

    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, x_true[:, i], 'k-', linewidth=2.0, label='Ground Truth')
        for cidx, (method, (_, x_hat, _)) in enumerate(trajectories.items()):
            colour = METHOD_COLOURS[cidx % len(METHOD_COLOURS)]
            display = METHOD_DISPLAY_NAMES.get(method, method)
            ax.plot(t, x_hat[:, i], '--', color=colour, linewidth=1.4, alpha=0.85, label=display)
        ax.set_ylabel(state_labels[i], fontsize=13)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)

    axes[-1].set_xlabel('Time', fontsize=13)
    fig.suptitle(f'{sys_name.upper()} \u2014 {input_type} ({mode_label})', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Core evaluation
# ============================================================================

KNOWN_SYSTEMS = ['duffing', 'vdp', 'lorenz', 'rossler',
                 'fitzhugh_nagumo', 'highway_traffic']


def evaluate_version_dir(version_dir, sys_name, device, hyper_cfg, n_trials, seed):
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {sys_name.upper()} \u2014 {version_dir}")
    print(f"{'=' * 70}")

    sys_config = get_system_config(sys_name)
    system = sys_config['class'](**sys_config['init_args'])
    T_net, T_inv_net = load_T_and_Tinv(version_dir, sys_config, device)

    methods = _detect_methods(version_dir)
    print(f"  Detected methods: {methods}")

    loaded = {}
    for m in methods:
        try:
            loaded[m] = _load_method_models(m, version_dir, sys_config, T_net, T_inv_net, device, hyper_cfg)
            print(f"    Loaded: {m}")
        except Exception as e:
            print(f"    FAILED to load {m}: {e}")

    if not loaded:
        print("  No methods loaded \u2014 skipping.")
        return

    eval_dir = version_dir / 'evaluation'
    eval_dir.mkdir(exist_ok=True)

    input_types = sys_config.get('natural_inputs', ['zero', 'constant', 'sinusoid', 'square'])
    window_size = hyper_cfg['window_size']
    all_metrics = {}

    for mode, mode_label, sub_dir_name in [
        ('id', 'In-Distribution', 'in_distribution'),
        ('ood', 'Out-of-Distribution', 'out_of_distribution'),
    ]:
        mode_dir = eval_dir / sub_dir_name
        mode_dir.mkdir(exist_ok=True)
        all_metrics[mode] = {}

        rng_plot = np.random.RandomState(seed + 999)
        plot_ic = rng_plot.uniform(low=sys_config['limits'][:, 0], high=sys_config['limits'][:, 1])

        from smt.sampling_methods import LHS
        eval_ics = LHS(xlimits=sys_config['limits'])(n_trials)

        for input_type in input_types:
            print(f"    [{mode_label}] Signal: {input_type}")

            input_gen_plot = get_input_generator(input_type, mode)
            rng_plot_sig = np.random.RandomState(seed + 777)
            input_gen_plot.sample_params(rng_plot_sig)

            overlay_trajs = OrderedDict()
            for m_name, m_models in loaded.items():
                try:
                    x_true, x_hat, t_eval = _simulate_method(
                        m_name, m_models, system, sys_config,
                        plot_ic, input_gen_plot, device, window_size)
                    overlay_trajs[m_name] = (x_true, x_hat, t_eval)
                except Exception as e:
                    print(f"      Plot sim failed for {m_name}: {e}")

            if overlay_trajs:
                plot_overlay_timeseries(
                    overlay_trajs, sys_name, input_type, mode_label,
                    mode_dir / f'{input_type}_timeseries.png')

            for m_name, m_models in loaded.items():
                if m_name not in all_metrics[mode]:
                    all_metrics[mode][m_name] = {}
                rng_m = np.random.RandomState(seed)
                trials = []
                for ic in eval_ics:
                    input_gen_m = get_input_generator(input_type, mode)
                    input_gen_m.sample_params(rng_m)
                    try:
                        trials.append(_simulate_method(
                            m_name, m_models, system, sys_config,
                            ic, input_gen_m, device, window_size))
                    except Exception:
                        pass
                if trials:
                    all_metrics[mode][m_name][input_type] = compute_metrics_multi_trial(trials)

        _write_metrics_table(all_metrics[mode], mode_label, sys_name, mode_dir, input_types)

    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Results saved to {eval_dir}")


def _write_metrics_table(metrics, mode_label, sys_name, out_dir, input_types):
    methods = sorted(metrics.keys())
    lines = [f"# {sys_name.upper()} \u2014 {mode_label}", "",
             "## Steady-State RMSE", "",
             "| Method | " + " | ".join(input_types) + " |",
             "|" + "|".join(["---"] * (len(input_types) + 1)) + "|"]
    for m in methods:
        display = METHOD_DISPLAY_NAMES.get(m, m)
        vals = [f"{metrics.get(m, {}).get(it, {}).get('rmse_steady', float('nan')):.4f}" for it in input_types]
        lines.append(f"| {display} | " + " | ".join(vals) + " |")
    lines += ["", "## Steady-State SMAPE (%)", "",
              "| Method | " + " | ".join(input_types) + " |",
              "|" + "|".join(["---"] * (len(input_types) + 1)) + "|"]
    for m in methods:
        display = METHOD_DISPLAY_NAMES.get(m, m)
        vals = [f"{metrics.get(m, {}).get(it, {}).get('smape_steady', float('nan')):.2f}" for it in input_types]
        lines.append(f"| {display} | " + " | ".join(vals) + " |")

    table_text = "\n".join(lines)
    print(table_text)
    with open(out_dir / 'metrics_table.md', 'w') as f:
        f.write(table_text)


# ============================================================================
# Directory traversal
# ============================================================================

def _find_version_dirs(results_dir):
    pairs = []
    if (results_dir / 'T_encoder.pt').exists():
        for part in results_dir.parts:
            if part in KNOWN_SYSTEMS:
                pairs.append((results_dir, part))
                return pairs
        return pairs

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in KNOWN_SYSTEMS:
            for vdir in sorted(child.iterdir()):
                if (vdir.is_dir() and vdir.name.startswith('v')
                        and vdir.name[1:].isdigit()
                        and (vdir / 'T_encoder.pt').exists()):
                    pairs.append((vdir, child.name))
        elif child.name.startswith('v') and child.name[1:].isdigit():
            if (child / 'T_encoder.pt').exists():
                for part in results_dir.parts:
                    if part in KNOWN_SYSTEMS:
                        pairs.append((child, part))
                        break
    return pairs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate HyperKKL checkpoints with overlay plots and metric tables.')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--system', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--lstm_hidden', type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    hyper_cfg = {
        'window_size': args.window_size, 'latent_dim': args.latent_dim,
        'lstm_hidden': args.lstm_hidden, 'rank': 32,
        'hypernet_hidden': 128, 'lora_rank': 4,
    }

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist.")
        sys.exit(1)

    if args.system and (results_dir / 'T_encoder.pt').exists():
        pairs = [(results_dir, args.system)]
    else:
        pairs = _find_version_dirs(results_dir)

    if not pairs:
        print(f"No version directories with T_encoder.pt found under {results_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} version dir(s) to evaluate:")
    for vdir, sname in pairs:
        print(f"  {sname}: {vdir}")

    for vdir, sname in pairs:
        evaluate_version_dir(vdir, sname, device, hyper_cfg, args.n_trials, args.seed)

    print("\nDone.")


if __name__ == '__main__':
    main()
