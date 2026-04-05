#!/usr/bin/env python
"""
Checkpoint evaluation script.

Loads trained checkpoints, evaluates on ID and OOD signals,
produces overlay plots, metric tables, and boxplots.

Usage:
    python -m scripts.evaluate --results_dir results/duffing/v1
    python -m scripts.evaluate --results_dir results  # all systems
"""

import sys
import re
import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, SystemConfig
from src.systems import create_system
from src.signals import create_signal
from src.models import (
    Normalizer, KKLNetwork,
    RecurrentEncoder, InputInjectionNet,
    ResidualHyperNetwork, PerLayerLoRAHyperNetwork,
    count_parameters, get_layer_sizes,
)
from src.evaluation import (
    simulate_observer, compute_metrics,
)
from src.plotting import plot_overlay_timeseries, METHOD_DISPLAY


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _build_nn(sys_config, role="encoder"):
    """Build a KKL network with dummy normalizer for checkpoint loading."""
    norm = Normalizer.dummy(sys_config.x_size, sys_config.z_size)
    if role == "encoder":
        return KKLNetwork(sys_config.num_hidden, sys_config.hidden_size,
                          sys_config.x_size, sys_config.z_size, normalizer=norm)
    else:
        return KKLNetwork(sys_config.num_hidden, sys_config.hidden_size,
                          sys_config.z_size, sys_config.x_size, normalizer=norm)


def _load_flexible(model, state_dict):
    """Load state_dict, skipping shape-mismatched keys."""
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)


def _remap_legacy(state_dict):
    remapped = {}
    for key, val in state_dict.items():
        new_key = re.sub(r"^gru_encoder\.gru\.", "rnn.", key)
        new_key = re.sub(r"^lstm_encoder\.lstm\.", "rnn.", new_key)
        remapped[new_key] = val
    return remapped


def _infer_cell_type(state_dict, hidden_dim):
    for key, val in state_dict.items():
        if key.endswith(".weight_ih_l0"):
            return "gru" if val.shape[0] == 3 * hidden_dim else "lstm"
    return None


def load_phase1(version_dir: Path, sys_config, device):
    T = _build_nn(sys_config, "encoder").to(device)
    T_inv = _build_nn(sys_config, "decoder").to(device)
    _load_flexible(T, torch.load(version_dir / "T_encoder.pt", map_location=device, weights_only=False)["model"])
    _load_flexible(T_inv, torch.load(version_dir / "T_inv_decoder.pt", map_location=device, weights_only=False)["model"])
    T.eval()
    T_inv.eval()
    return T, T_inv


def _detect_methods(version_dir: Path) -> list:
    method_files = {
        "autonomous": "T_encoder.pt",
        "curriculum": "curriculum.pt",
        "augmented": "augmented.pt",
        "full": "full.pt",
        "lora": "lora.pt",
    }
    return [m for m, f in method_files.items() if (version_dir / f).exists()]


def load_method(method, version_dir, sys_config, T_net, T_inv_net, device, hyper_cfg):
    T_copy = copy.deepcopy(T_net)
    T_inv_copy = copy.deepcopy(T_inv_net)
    ws = hyper_cfg["window_size"]
    ld = hyper_cfg["latent_dim"]

    if method == "autonomous":
        return {"type": "autonomous", "T_inv": T_inv_copy}

    elif method == "curriculum":
        ckpt = torch.load(version_dir / "curriculum.pt", map_location=device, weights_only=False)
        _load_flexible(T_copy, ckpt["T_encoder"])
        _load_flexible(T_inv_copy, ckpt["T_inv_decoder"])
        T_copy.eval()
        T_inv_copy.eval()
        return {"type": "autonomous", "T_inv": T_inv_copy}

    elif method == "augmented":
        rnn_h = hyper_cfg.get("rnn_hidden", 64)
        enc = RecurrentEncoder(1, rnn_h, ld, cell_type=hyper_cfg.get("encoder_type", "lstm")).to(device)
        phi = InputInjectionNet(sys_config.z_size, ld, 1).to(device)
        ckpt = torch.load(version_dir / f"{method}.pt", map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["encoder"])
        phi.load_state_dict(ckpt["phi"])
        enc.eval()
        phi.eval()
        return {"type": "augmented", "T_inv": T_inv_copy, "encoder": enc, "phi": phi}

    elif method == "full":
        rnn_h = hyper_cfg.get("rnn_hidden", 64)
        theta_enc = count_parameters(T_copy)
        theta_dec = count_parameters(T_inv_copy)
        ckpt = torch.load(version_dir / f"{method}.pt", map_location=device, weights_only=False)
        sd = _remap_legacy(ckpt["hypernet"])
        cell = _infer_cell_type(sd, rnn_h) or hyper_cfg.get("encoder_type", "lstm")
        hypernet = ResidualHyperNetwork(1, rnn_h, theta_enc, theta_dec,
                                        mlp_hidden_dim=hyper_cfg.get("hypernet_hidden", 128),
                                        cell_type=cell).to(device)
        hypernet.load_state_dict(sd)
        hypernet.eval()
        for p in T_copy.parameters():
            p.requires_grad = False
        for p in T_inv_copy.parameters():
            p.requires_grad = False
        return {"type": "full", "hypernet": hypernet, "T_base": T_copy, "T_inv_base": T_inv_copy}

    elif method == "lora":
        rnn_h = hyper_cfg.get("rnn_hidden", 64)
        ckpt = torch.load(version_dir / f"{method}.pt", map_location=device, weights_only=False)
        sd = _remap_legacy(ckpt["hypernet"])
        cell = _infer_cell_type(sd, rnn_h) or hyper_cfg.get("encoder_type", "lstm")
        enc_sizes = get_layer_sizes(T_copy)
        dec_sizes = get_layer_sizes(T_inv_copy)
        hypernet = PerLayerLoRAHyperNetwork(
            1, rnn_h, enc_sizes, dec_sizes,
            rank=hyper_cfg.get("lora_rank", 4),
            mlp_hidden_dim=hyper_cfg.get("hypernet_hidden", 128),
            cell_type=cell).to(device)
        hypernet.load_state_dict(sd)
        hypernet.eval()
        for p in T_copy.parameters():
            p.requires_grad = False
        for p in T_inv_copy.parameters():
            p.requires_grad = False
        return {"type": "lora", "hypernet": hypernet, "T_base": T_copy, "T_inv_base": T_inv_copy}

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Simulation dispatch
# ---------------------------------------------------------------------------

def _simulate_method(method_name, models, system, sys_config, ic, input_func, device, ws):
    mtype = models["type"]
    kwargs = {"method_type": mtype, "window_size": ws}
    if mtype in ("autonomous", "augmented"):
        kwargs.update(T_inv=models["T_inv"], encoder=models.get("encoder"), phi_net=models.get("phi"))
    elif mtype in ("full", "lora"):
        kwargs.update(hypernet=models["hypernet"], T_base=models["T_base"],
                      T_inv_base=models["T_inv_base"], skip_bias=mtype == "lora")
    return simulate_observer(system, sys_config, ic, input_func, device, **kwargs)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_version_dir(version_dir, sys_name, device, hyper_cfg, n_trials, seed):
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {sys_name.upper()} -- {version_dir}")
    print(f"{'=' * 70}")

    cfg = load_config(sys_name)
    sys_config = cfg.system
    system = create_system(sys_config)

    T_net, T_inv_net = load_phase1(version_dir, sys_config, device)
    methods = _detect_methods(version_dir)
    print(f"  Methods: {methods}")

    loaded = {}
    for m in methods:
        try:
            loaded[m] = load_method(m, version_dir, sys_config, T_net, T_inv_net, device, hyper_cfg)
            print(f"    Loaded: {m}")
        except Exception as e:
            print(f"    FAILED {m}: {e}")

    if not loaded:
        return

    eval_dir = version_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    ws = hyper_cfg["window_size"]

    input_types = ["zero", "constant", "sinusoid", "square"]
    all_metrics = {}

    for mode, label, subdir in [("id", "In-Distribution", "in_distribution"),
                                 ("ood", "Out-of-Distribution", "out_of_distribution")]:
        mode_dir = eval_dir / subdir
        mode_dir.mkdir(exist_ok=True)
        all_metrics[mode] = {}

        rng_plot = np.random.RandomState(seed + 999)
        plot_ic = rng_plot.uniform(low=sys_config.limits_np[:, 0], high=sys_config.limits_np[:, 1])

        from smt.sampling_methods import LHS
        eval_ics = LHS(xlimits=sys_config.limits_np)(n_trials)
        rng_eval = np.random.RandomState(seed)

        for sig_type in input_types:
            print(f"    [{label}] {sig_type}")

            # Overlay plot
            sig_plot = create_signal(sig_type, mode)
            sig_plot.sample_params(np.random.RandomState(seed + 777))
            overlay = OrderedDict()
            for mname, mmodels in loaded.items():
                try:
                    overlay[mname] = _simulate_method(mname, mmodels, system, sys_config,
                                                      plot_ic, sig_plot, device, ws)
                except Exception as e:
                    print(f"      Plot sim failed {mname}: {e}")

            if overlay:
                plot_overlay_timeseries(overlay, sys_name, sig_type, label,
                                        mode_dir / f"{sig_type}_timeseries.png")

            # Metrics
            for mname, mmodels in loaded.items():
                if mname not in all_metrics[mode]:
                    all_metrics[mode][mname] = {}
                rng_m = np.random.RandomState(seed)
                trials = []
                for ic in eval_ics:
                    sig_m = create_signal(sig_type, mode)
                    sig_m.sample_params(rng_m)
                    try:
                        trials.append(_simulate_method(mname, mmodels, system, sys_config,
                                                       ic, sig_m, device, ws))
                    except Exception:
                        pass
                if trials:
                    all_metrics[mode][mname][sig_type] = compute_metrics(trials)

        _write_table(all_metrics[mode], label, sys_name, mode_dir)

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved to {eval_dir}")


def _write_table(metrics, mode_label, sys_name, out_dir):
    inputs = ["zero", "constant", "sinusoid", "square"]
    methods = sorted(metrics.keys())
    lines = [f"# {sys_name.upper()} -- {mode_label}", "", "## Steady-State RMSE", ""]
    header = "| Method | " + " | ".join(inputs) + " |"
    sep = "|" + "|".join(["---"] * (len(inputs) + 1)) + "|"
    lines.extend([header, sep])
    for m in methods:
        vals = [f"{metrics.get(m, {}).get(it, {}).get('rmse_steady', float('nan')):.4f}" for it in inputs]
        lines.append(f"| {METHOD_DISPLAY.get(m, m)} | " + " | ".join(vals) + " |")
    lines.extend(["", "## Steady-State SMAPE (%)", "", header, sep])
    for m in methods:
        vals = [f"{metrics.get(m, {}).get(it, {}).get('smape_steady', float('nan')):.2f}" for it in inputs]
        lines.append(f"| {METHOD_DISPLAY.get(m, m)} | " + " | ".join(vals) + " |")
    text = "\n".join(lines)
    print(text)
    with open(out_dir / "metrics_table.md", "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Directory traversal
# ---------------------------------------------------------------------------

KNOWN_SYSTEMS = ["duffing", "vdp", "lorenz", "rossler", "fhn", "highway_traffic"]


def _find_version_dirs(results_dir):
    pairs = []
    if (results_dir / "T_encoder.pt").exists():
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
                if vdir.is_dir() and vdir.name.startswith("v") and (vdir / "T_encoder.pt").exists():
                    pairs.append((vdir, child.name))
        elif child.name.startswith("v") and (child / "T_encoder.pt").exists():
            for part in results_dir.parts:
                if part in KNOWN_SYSTEMS:
                    pairs.append((child, part))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate HyperKKL checkpoints")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--rnn_hidden", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hyper_cfg = {
        "window_size": args.window_size, "latent_dim": args.latent_dim,
        "rnn_hidden": args.rnn_hidden, "encoder_type": "lstm", "hypernet_hidden": 128,
        "lora_rank": 4,
    }

    results_dir = Path(args.results_dir)
    if args.system and (results_dir / "T_encoder.pt").exists():
        pairs = [(results_dir, args.system)]
    else:
        pairs = _find_version_dirs(results_dir)

    if not pairs:
        print(f"No checkpoints found under {results_dir}")
        sys.exit(1)

    for vdir, sname in pairs:
        evaluate_version_dir(vdir, sname, device, hyper_cfg, args.n_trials, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
