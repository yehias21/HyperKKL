"""
Unified observer simulation, metrics, and evaluation.

Key improvement: ONE simulate function handles all method types
(autonomous, augmented, full, lora) via a unified decoder interface.

Standalone test:
    python -m src.evaluation
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from smt.sampling_methods import LHS

from src.signals import create_signal
from src.models import apply_weight_modulation, apply_weight_modulation_skip_bias


# ---------------------------------------------------------------------------
# True system trajectory (shared by all methods)
# ---------------------------------------------------------------------------

def simulate_true_system(system, sys_config, ic: np.ndarray, input_func: Callable):
    """Generate ground-truth trajectory using RK4.

    Returns: x_true (N+1, x_size), y (N+1, y_size), u_vals (N+1,), t_eval (N+1,)
    """
    dt = sys_config.dt
    N = sys_config.n_steps
    t_eval = np.linspace(sys_config.time_start, sys_config.time_end, N + 1)

    u_vals = np.array([input_func(t) if callable(input_func) else 0.0 for t in t_eval])

    noise_backup = getattr(system, "add_noise", False)
    system.add_noise = False

    x = torch.tensor(ic, dtype=torch.float64).unsqueeze(0)
    u_t = torch.tensor(u_vals, dtype=torch.float64)

    traj = [x.clone()]
    for i in range(N):
        u_i = u_t[i:i + 1]
        u_half = ((u_t[i] + u_t[min(i + 1, N)]) / 2.0).unsqueeze(0)
        u_next = u_t[min(i + 1, N):min(i + 1, N) + 1]

        k1 = system.function(t_eval[i], u_i, x)
        k2 = system.function(t_eval[i] + dt / 2, u_half, x + dt / 2 * k1)
        k3 = system.function(t_eval[i] + dt / 2, u_half, x + dt / 2 * k2)
        k4 = system.function(t_eval[min(i + 1, N)], u_next, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.clone())

    x_true = torch.stack(traj).squeeze(1).numpy()

    x_all_t = torch.tensor(x_true, dtype=torch.float64)
    y = system.output(x_all_t).numpy()
    if y.ndim == 1:
        y = y[:, np.newaxis]

    system.add_noise = noise_backup
    return x_true, y, u_vals, t_eval


# ---------------------------------------------------------------------------
# Unified observer simulation
# ---------------------------------------------------------------------------

def simulate_observer(system, sys_config, ic: np.ndarray, input_func: Callable,
                      device: torch.device, method_type: str = "autonomous",
                      T_inv=None, encoder=None, phi_net=None,
                      hypernet=None, T_base=None, T_inv_base=None,
                      skip_bias: bool = False, window_size: int = 100):
    """Unified observer simulation for all methods.

    Args:
        method_type: "autonomous", "augmented", "full", "lora"
        T_inv: decoder for autonomous/augmented methods
        encoder, phi_net: for augmented methods
        hypernet, T_base, T_inv_base: for full/lora methods
        skip_bias: True for LoRA methods

    Returns: x_true, x_hat, t_eval
    """
    x_true, y, u_vals, t_eval = simulate_true_system(system, sys_config, ic, input_func)
    dt = sys_config.dt
    N = sys_config.n_steps

    M_t = torch.tensor(sys_config.M_np, dtype=torch.float32, device=device)
    K_t = torch.tensor(sys_config.K_np, dtype=torch.float32, device=device).flatten()
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).squeeze(-1)
    u_tensor = torch.tensor(u_vals, dtype=torch.float32, device=device)

    z = torch.zeros(sys_config.z_size, dtype=torch.float32, device=device)

    modulate_fn = apply_weight_modulation_skip_bias if skip_bias else apply_weight_modulation

    # --- Dynamic methods: step-by-step decoding ---
    if method_type in ("full", "lora"):
        hypernet.eval()
        is_recurrent = hasattr(hypernet, "step")
        x_hat_list = []
        rnn_state = None

        if is_recurrent:
            for i in range(N + 1):
                u_step = u_tensor[i].view(1, 1, 1)
                with torch.no_grad():
                    delta_enc, delta_dec, rnn_state = hypernet.step(u_step, rnn_state)
                    mod_params = modulate_fn(T_inv_base, delta_dec)
                    x_hat = torch.func.functional_call(
                        T_inv_base, mod_params, z.unsqueeze(0)
                    ).cpu().numpy().squeeze()
                x_hat_list.append(x_hat)

                if i < N:
                    y_i = y_tensor[i]
                    k1 = torch.mv(M_t, z) + K_t * y_i
                    k2 = torch.mv(M_t, z + 0.5 * dt * k1) + K_t * y_i
                    k3 = torch.mv(M_t, z + 0.5 * dt * k2) + K_t * y_i
                    k4 = torch.mv(M_t, z + dt * k3) + K_t * y_i
                    z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            # Window-based
            pad_len = window_size - 1
            u_padded = torch.cat([u_tensor[0:1].repeat(pad_len), u_tensor])

            for i in range(N + 1):
                u_win = u_padded[i:i + window_size].unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    delta_enc, delta_dec = hypernet(u_win)
                    mod_params = modulate_fn(T_inv_base, delta_dec)
                    x_hat = torch.func.functional_call(
                        T_inv_base, mod_params, z.unsqueeze(0)
                    ).cpu().numpy().squeeze()
                x_hat_list.append(x_hat)

                if i < N:
                    y_i = y_tensor[i]
                    k1 = torch.mv(M_t, z) + K_t * y_i
                    k2 = torch.mv(M_t, z + 0.5 * dt * k1) + K_t * y_i
                    k3 = torch.mv(M_t, z + 0.5 * dt * k2) + K_t * y_i
                    k4 = torch.mv(M_t, z + dt * k3) + K_t * y_i
                    z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_true, np.array(x_hat_list), t_eval

    # --- Static / Autonomous methods ---
    z_traj = torch.zeros((N + 1, sys_config.z_size), dtype=torch.float32, device=device)

    if method_type == "augmented" and encoder is not None and phi_net is not None:
        # Precompute latents
        pad_len = window_size - 1
        u_padded = torch.cat([u_tensor[0:1].repeat(pad_len), u_tensor])
        u_windows = u_padded.unfold(0, window_size, 1)[:N].unsqueeze(-1)

        encoder.eval()
        phi_net.eval()
        with torch.no_grad():
            latents = encoder(u_windows)

        for i in range(N):
            y_i = y_tensor[i]
            latent_i = latents[i].unsqueeze(0)
            has_input = u_tensor[i].abs().item() > 0

            base = torch.mv(M_t, z) + K_t * y_i
            if has_input:
                z_in = z.unsqueeze(0)
                phi_val = phi_net(z_in, latent_i).squeeze(-1).flatten()
                base = base + phi_val

            k1 = base
            # Simplified: reuse base for RK4 sub-steps (input constant over dt)
            k2 = torch.mv(M_t, z + 0.5 * dt * k1) + K_t * y_i
            k3 = torch.mv(M_t, z + 0.5 * dt * k2) + K_t * y_i
            k4 = torch.mv(M_t, z + dt * k3) + K_t * y_i
            if has_input:
                z2 = z + 0.5 * dt * k1
                k2 = k2 + phi_net(z2.unsqueeze(0), latent_i).squeeze(-1).flatten()
                z3 = z + 0.5 * dt * k2
                k3 = torch.mv(M_t, z3) + K_t * y_i + phi_net(z3.unsqueeze(0), latent_i).squeeze(-1).flatten()
                z4 = z + dt * k3
                k4 = torch.mv(M_t, z4) + K_t * y_i + phi_net(z4.unsqueeze(0), latent_i).squeeze(-1).flatten()

            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            z_traj[i + 1] = z
    else:
        # Autonomous
        for i in range(N):
            y_i = y_tensor[i]
            k1 = torch.mv(M_t, z) + K_t * y_i
            k2 = torch.mv(M_t, z + 0.5 * dt * k1) + K_t * y_i
            k3 = torch.mv(M_t, z + 0.5 * dt * k2) + K_t * y_i
            k4 = torch.mv(M_t, z + dt * k3) + K_t * y_i
            z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            z_traj[i + 1] = z

    T_inv.eval()
    with torch.no_grad():
        x_hat = T_inv(z_traj).cpu().numpy()

    return x_true, x_hat, t_eval


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(x_true: np.ndarray, x_hat: np.ndarray, settle_idx: int = 0) -> float:
    error = np.linalg.norm(x_true[settle_idx:] - x_hat[settle_idx:], axis=1)
    return float(np.sqrt(np.mean(error ** 2)))


def compute_smape(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    norm_true = np.linalg.norm(x_true, axis=1)
    norm_hat = np.linalg.norm(x_hat, axis=1)
    norm_diff = np.linalg.norm(x_true - x_hat, axis=1)
    denom = norm_true + norm_hat
    safe_denom = np.where(denom > 0, denom, 1.0)
    smape = np.where(denom > 0, 2.0 * norm_diff / safe_denom * 100.0, 0.0)
    return float(np.mean(smape))


def compute_metrics(results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    settle_time: float = 5.0) -> dict:
    """Compute averaged metrics over multiple trial results."""
    rmse_list, smape_list = [], []
    for x_true, x_hat, t in results:
        settle_idx = np.searchsorted(t, settle_time)
        rmse_list.append(compute_rmse(x_true, x_hat, settle_idx))
        smape_list.append(compute_smape(x_true[settle_idx:], x_hat[settle_idx:]))
    return {
        "rmse_steady": float(np.mean(rmse_list)),
        "smape_steady": float(np.mean(smape_list)),
    }


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------

def _make_simulate_kwargs(method_name, models):
    """Convert loaded model dict to simulate_observer kwargs."""
    mtype = models["type"]
    kwargs = {"method_type": mtype}

    if mtype in ("autonomous", "augmented"):
        kwargs["T_inv"] = models["T_inv"]
        kwargs["encoder"] = models.get("encoder")
        kwargs["phi_net"] = models.get("phi")
    elif mtype in ("full", "lora"):
        kwargs["hypernet"] = models["hypernet"]
        kwargs["T_base"] = models["T_base"]
        kwargs["T_inv_base"] = models["T_inv_base"]
        kwargs["skip_bias"] = mtype == "lora"

    return kwargs


def evaluate_method(system, sys_config, models: dict, method_name: str,
                    signal_types: list, n_trials: int, seed: int,
                    device: torch.device, window_size: int = 100,
                    signal_mode: str = "id"):
    """Evaluate a single method across all signal types.

    Returns:
        results: {signal_type: {rmse_steady, smape_steady}} (averaged)
        per_trial: {signal_type: [{rmse_total, rmse_steady, max_error, smape}, ...]}
    """
    sampler = LHS(xlimits=sys_config.limits_np)
    test_ics = sampler(n_trials)
    rng = np.random.RandomState(seed)

    sim_kwargs = _make_simulate_kwargs(method_name, models)

    results = {}
    per_trial = {}
    for sig_type in signal_types:
        sig_gen = create_signal(sig_type, signal_mode)
        trial_metrics = []

        for ic in test_ics:
            sig_gen.sample_params(rng)
            try:
                x_true, x_hat, t = simulate_observer(
                    system, sys_config, ic, sig_gen, device,
                    window_size=window_size, **sim_kwargs)

                error = np.linalg.norm(x_true - x_hat, axis=1)
                settle_idx = np.searchsorted(t, 5.0)

                abs_sum = np.abs(x_true) + np.abs(x_hat) + 1e-8
                smape_val = float(np.mean(2.0 * np.abs(x_true - x_hat) / abs_sum) * 100.0)

                trial_metrics.append({
                    "rmse_total": float(np.sqrt(np.mean(error ** 2))),
                    "rmse_steady": float(np.sqrt(np.mean(error[settle_idx:] ** 2))),
                    "max_error": float(np.max(error)),
                    "smape": smape_val,
                })
            except Exception as e:
                print(f"    Eval failed for {method_name}/{sig_type}: {e}")

        if trial_metrics:
            results[sig_type] = {k: np.mean([m[k] for m in trial_metrics])
                                 for k in trial_metrics[0]}
            per_trial[sig_type] = trial_metrics

    return results, per_trial


def get_plot_trajectories(system, sys_config, models: dict, method_name: str,
                          device: torch.device, window_size: int, seed: int) -> dict:
    """Generate single-IC trajectories for all signal types (for plotting)."""
    rng = np.random.RandomState(seed + 999)
    ic = rng.uniform(low=sys_config.limits_np[:, 0], high=sys_config.limits_np[:, 1])

    sim_kwargs = _make_simulate_kwargs(method_name, models)
    trajectories = {}

    for sig_type in sys_config.natural_inputs:
        sig_gen = create_signal(sig_type, "id")
        sig_gen.sample_params(rng)
        try:
            x_true, x_hat, t = simulate_observer(
                system, sys_config, ic, sig_gen, device,
                window_size=window_size, **sim_kwargs)
            trajectories[sig_type] = (x_true, x_hat, t)
        except Exception as e:
            print(f"    Plot traj failed for {method_name}/{sig_type}: {e}")

    return trajectories


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.config import load_config
    from src.systems import create_system

    cfg = load_config("duffing")
    sys = create_system(cfg.system)
    sys.add_noise = False

    ic = np.array([0.5, 0.5])
    sig = create_signal("sinusoid", "id")
    sig.sample_params(np.random.RandomState(42))

    x_true, y, u_vals, t_eval = simulate_true_system(sys, cfg.system, ic, sig)
    print(f"True system: x={x_true.shape}, y={y.shape}, t={t_eval.shape}")
    print(f"RMSE (self, should be ~0): {compute_rmse(x_true, x_true):.6f}")
