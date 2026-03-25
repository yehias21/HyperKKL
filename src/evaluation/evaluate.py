"""
Observer simulation and evaluation for both static and dynamic HyperKKL.
"""

import numpy as np
import torch
from typing import Callable, Dict, List
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hyperkkl.src.data.signals import get_input_generator
from hyperkkl.src.models.hypernetworks import apply_weight_modulation, apply_weight_modulation_skip_bias


# ---------------------------------------------------------------------------
# Shared helper: generate true-system trajectory (same RK4 as training data)
# ---------------------------------------------------------------------------

def _simulate_true_system(system, sys_config: dict, ic: np.ndarray,
                          input_func: Callable):
    """Generate a single true-system trajectory using torch-batch RK4.

    Uses the **same** integration path as the training-data workers in
    ``pipeline/data.py`` (torch float64, ``system.function`` in batch mode)
    so that training and evaluation are perfectly consistent.

    Returns
    -------
    x_true : ndarray (N+1, x_size)
    y      : ndarray (N+1, y_size)  — always 2-D
    u_vals : ndarray (N+1,)
    t_eval : ndarray (N+1,)
    """
    dt = (sys_config['b'] - sys_config['a']) / sys_config['N']
    N = sys_config['N']
    t_eval = np.linspace(sys_config['a'], sys_config['b'], N + 1)

    # Build input signal on the time grid
    u_vals = np.array([input_func(t) if callable(input_func) else 0.0
                       for t in t_eval])

    # Disable measurement noise (matches data workers)
    noise_backup = getattr(system, 'add_noise', False)
    system.add_noise = False

    # --- Vectorised RK4 in torch (batch_size=1, float64) ------------------
    x = torch.tensor(ic, dtype=torch.float64).unsqueeze(0)   # (1, x_size)
    u_t = torch.tensor(u_vals, dtype=torch.float64)            # (N+1,)

    x_traj = [x.clone()]
    for i in range(N):
        u_i    = u_t[i:i+1]
        u_half = ((u_t[i] + u_t[min(i+1, N)]) / 2.0).unsqueeze(0)
        u_next = u_t[min(i+1, N):min(i+1, N)+1]

        h = dt
        k1 = system.function(t_eval[i],            u_i,    x)
        k2 = system.function(t_eval[i] + h / 2,    u_half, x + h / 2 * k1)
        k3 = system.function(t_eval[i] + h / 2,    u_half, x + h / 2 * k2)
        k4 = system.function(t_eval[min(i+1, N)],  u_next, x + h * k3)

        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_traj.append(x.clone())

    # (N+1, 1, x_size) → (N+1, x_size)
    x_true = torch.stack(x_traj).squeeze(1).numpy()

    # Measurements via system.output (batch mode, matching data workers)
    x_all_t = torch.tensor(x_true, dtype=torch.float64)
    y = system.output(x_all_t).numpy()            # (N+1, y_size) or (N+1,)
    if y.ndim == 1:
        y = y[:, np.newaxis]                       # always (N+1, y_size)

    # Restore noise flag
    system.add_noise = noise_backup

    return x_true, y, u_vals, t_eval


# ---------------------------------------------------------------------------
# Static / Autonomous / HyperKKL observer
# ---------------------------------------------------------------------------

def simulate_observer(system, T_inv, sys_config: dict, ic: np.ndarray, input_func: Callable,
                      input_encoder=None, phi_net=None, device=None, window_size: int = 20):
    """Simulate observer (GPU optimized)."""
    x_true, y, u_vals, t_eval = _simulate_true_system(
        system, sys_config, ic, input_func)
    dt = (sys_config['b'] - sys_config['a']) / sys_config['N']
    N = sys_config['N']

    # Simulate observer on GPU
    M_t = torch.tensor(sys_config['M'], dtype=torch.float32, device=device)
    K_t = torch.tensor(sys_config['K'], dtype=torch.float32, device=device).flatten()

    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).squeeze(-1)
    u_tensor = torch.tensor(u_vals, dtype=torch.float32, device=device)

    z = torch.zeros(sys_config['z_size'], dtype=torch.float32, device=device)
    z_traj = torch.zeros((N + 1, sys_config['z_size']), dtype=torch.float32, device=device)

    if input_encoder is None or phi_net is None:
        # Autonomous observer
        for i in range(N):
            y_i = y_tensor[i]
            def dynamics(z_val):
                return torch.mv(M_t, z_val) + K_t * y_i

            k1 = dynamics(z)
            k2 = dynamics(z + 0.5 * dt * k1)
            k3 = dynamics(z + 0.5 * dt * k2)
            k4 = dynamics(z + dt * k3)
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            z_traj[i+1] = z
    else:
        # HyperKKL observer - batch process windows
        pad_len = window_size - 1
        u_0 = u_tensor[0].unsqueeze(0).repeat(pad_len)
        u_padded = torch.cat([u_0, u_tensor], dim=0)
        u_windows = u_padded.unfold(0, window_size, 1)[:N].unsqueeze(-1)

        input_encoder.eval()
        phi_net.eval()

        with torch.no_grad():
            latents = input_encoder(u_windows)

        for i in range(N):
            y_i = y_tensor[i]
            u_i = u_tensor[i]
            latent_i = latents[i].unsqueeze(0)
            # Use u as indicator: add phi only when input is nonzero
            has_input = (u_i.abs() > 0).item()

            def dynamics(z_val, _has_input=has_input):
                base = torch.mv(M_t, z_val) + K_t * y_i
                if _has_input:
                    z_in = z_val.unsqueeze(0)
                    phi_val = phi_net(z_in, latent_i).squeeze(-1).flatten()
                    base = base + phi_val
                return base

            k1 = dynamics(z)
            k2 = dynamics(z + 0.5 * dt * k1)
            k3 = dynamics(z + 0.5 * dt * k2)
            k4 = dynamics(z + dt * k3)
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            z_traj[i+1] = z

    T_inv.eval()
    with torch.no_grad():
        x_hat = T_inv(z_traj).cpu().numpy()

    return x_true, x_hat, t_eval


# ---------------------------------------------------------------------------
# Dynamic HyperKKL observer (weight modulation of T / T*)
# ---------------------------------------------------------------------------

def simulate_observer_dynamic(system, hypernet, T_base, T_inv_base, sys_config: dict,
                               ic: np.ndarray, input_func: Callable, device, window_size: int = 20):
    """Simulate dynamic HyperKKL observer.

    Supports both window-based (DualHyperNetwork) and recurrent LSTM-based
    (ResidualHyperNetwork) hypernetworks. For LSTM-based networks, inputs
    are processed step-by-step maintaining the LSTM hidden state.
    """
    x_true, y, u_vals, t_eval = _simulate_true_system(
        system, sys_config, ic, input_func)
    dt = (sys_config['b'] - sys_config['a']) / sys_config['N']
    N = sys_config['N']

    M_t = torch.tensor(sys_config['M'], dtype=torch.float32, device=device)
    K_t = torch.tensor(sys_config['K'], dtype=torch.float32, device=device).flatten()
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).squeeze(-1)
    u_tensor = torch.tensor(u_vals, dtype=torch.float32, device=device)

    z = torch.zeros(sys_config['z_size'], dtype=torch.float32, device=device)
    x_hat_list = []

    hypernet.eval()

    is_recurrent = hasattr(hypernet, 'step')

    if is_recurrent:
        # LSTM-based ResidualHyperNetwork: process inputs step-by-step
        lstm_state = None

        for i in range(N + 1):
            u_step = u_tensor[i].view(1, 1, 1)

            with torch.no_grad():
                delta_enc, delta_dec, lstm_state = hypernet.step(u_step, lstm_state)
                mod_dec_params = apply_weight_modulation(T_inv_base, delta_dec)
                z_t = z.unsqueeze(0)
                x_hat = torch.func.functional_call(
                    T_inv_base, mod_dec_params, z_t
                ).cpu().numpy().squeeze()

            x_hat_list.append(x_hat)

            if i < N:
                y_i = y_tensor[i]
                def dynamics_lstm(z_val):
                    return torch.mv(M_t, z_val) + K_t * y_i
                k1 = dynamics_lstm(z)
                k2 = dynamics_lstm(z + 0.5 * dt * k1)
                k3 = dynamics_lstm(z + 0.5 * dt * k2)
                k4 = dynamics_lstm(z + dt * k3)
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    else:
        # Window-based DualHyperNetwork: process input windows
        pad_len = window_size - 1
        u_0 = u_tensor[0].unsqueeze(0).repeat(pad_len)
        u_padded = torch.cat([u_0, u_tensor], dim=0)

        for i in range(N + 1):
            u_win = u_padded[i:i+window_size].unsqueeze(0).unsqueeze(-1)

            with torch.no_grad():
                delta_enc, delta_dec = hypernet(u_win)
                mod_dec_params = apply_weight_modulation(T_inv_base, delta_dec)
                z_t = z.unsqueeze(0)
                x_hat = torch.func.functional_call(
                    T_inv_base, mod_dec_params, z_t
                ).cpu().numpy().squeeze()

            x_hat_list.append(x_hat)

            if i < N:
                y_i = y_tensor[i]
                def dynamics_win(z_val):
                    return torch.mv(M_t, z_val) + K_t * y_i
                k1 = dynamics_win(z)
                k2 = dynamics_win(z + 0.5 * dt * k1)
                k3 = dynamics_win(z + 0.5 * dt * k2)
                k4 = dynamics_win(z + dt * k3)
                z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_true, np.array(x_hat_list), t_eval


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(system, T_inv, sys_config: dict, input_types: List[str], n_trials: int, seed: int,
             input_encoder=None, phi_net=None, device=None, mode: str = 'id'):
    """Evaluate observer. Returns (mean_results, per_trial_metrics)."""
    from smt.sampling_methods import LHS
    sampler = LHS(xlimits=sys_config['limits'])
    test_ics = sampler(n_trials)
    rng = np.random.RandomState(seed)

    results = {}
    per_trial = {}
    for input_type in input_types:
        input_gen = get_input_generator(input_type, mode)
        metrics = []

        for ic in test_ics:
            input_gen.sample_params(rng)
            x_true, x_hat, t = simulate_observer(
                system, T_inv, sys_config, ic, input_gen, input_encoder, phi_net, device
            )

            error = np.linalg.norm(x_true - x_hat, axis=1)
            settle_idx = np.searchsorted(t, 5.0)

            metrics.append({
                'rmse_total': float(np.sqrt(np.mean(error**2))),
                'rmse_steady': float(np.sqrt(np.mean(error[settle_idx:]**2))),
                'max_error': float(np.max(error)),
            })

        results[input_type] = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        per_trial[input_type] = metrics

    return results, per_trial


def evaluate_dynamic(system, hypernet, T_base, T_inv_base, sys_config: dict,
                     input_types: List[str], n_trials: int, seed: int, device):
    """Evaluate dynamic HyperKKL observer. Returns (mean_results, per_trial_metrics)."""
    from smt.sampling_methods import LHS
    sampler = LHS(xlimits=sys_config['limits'])
    test_ics = sampler(n_trials)
    rng = np.random.RandomState(seed)

    results = {}
    per_trial = {}
    for input_type in input_types:
        input_gen = get_input_generator(input_type, 'id')
        metrics = []

        for ic in test_ics:
            input_gen.sample_params(rng)
            x_true, x_hat, t = simulate_observer_dynamic(
                system, hypernet, T_base, T_inv_base, sys_config, ic, input_gen, device
            )

            error = np.linalg.norm(x_true - x_hat, axis=1)
            settle_idx = np.searchsorted(t, 5.0)

            metrics.append({
                'rmse_total': float(np.sqrt(np.mean(error**2))),
                'rmse_steady': float(np.sqrt(np.mean(error[settle_idx:]**2))),
                'max_error': float(np.max(error)),
            })

        results[input_type] = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        per_trial[input_type] = metrics

    return results, per_trial


def get_plot_trajectories(system, T_inv, sys_config, input_encoder,
                          phi_net, device, window_size, seed):
    """Generate trajectories for all input types (same IC) for plotting."""
    rng = np.random.RandomState(seed + 999)
    ic = rng.uniform(low=sys_config['limits'][:, 0], high=sys_config['limits'][:, 1])

    input_types = sys_config.get('natural_inputs', ['zero', 'constant', 'sinusoid', 'square'])
    trajectories = {}
    for input_type in input_types:
        input_gen = get_input_generator(input_type, 'id')
        input_gen.sample_params(rng)
        x_true, x_hat, t = simulate_observer(
            system, T_inv, sys_config, ic, input_gen,
            input_encoder, phi_net, device, window_size
        )
        trajectories[input_type] = (x_true, x_hat, t)
    return trajectories


def get_plot_trajectories_dynamic(system, hypernet, T_base,
                                  T_inv_base, sys_config, device,
                                  window_size, seed):
    """Generate trajectories for all input types (dynamic methods, same IC)."""
    rng = np.random.RandomState(seed + 999)
    ic = rng.uniform(low=sys_config['limits'][:, 0], high=sys_config['limits'][:, 1])

    input_types = sys_config.get('natural_inputs', ['zero', 'constant', 'sinusoid', 'square'])
    trajectories = {}
    for input_type in input_types:
        input_gen = get_input_generator(input_type, 'id')
        input_gen.sample_params(rng)
        x_true, x_hat, t = simulate_observer_dynamic(
            system, hypernet, T_base, T_inv_base, sys_config,
            ic, input_gen, device, window_size
        )
        trajectories[input_type] = (x_true, x_hat, t)
    return trajectories


# ---------------------------------------------------------------------------
# Dynamic LoRA observer (weight modulation skipping biases)
# ---------------------------------------------------------------------------

def simulate_observer_dynamic_lora(system, hypernet, T_base, T_inv_base, sys_config: dict,
                                   ic: np.ndarray, input_func, device, window_size: int = 20):
    """Simulate dynamic LoRA observer using apply_weight_modulation_skip_bias.

    Uses recurrent (LSTM step) processing of inputs, same as the LSTM branch
    of simulate_observer_dynamic but with bias-skipping weight modulation.
    """
    x_true, y, u_vals, t_eval = _simulate_true_system(
        system, sys_config, ic, input_func)
    dt = (sys_config['b'] - sys_config['a']) / sys_config['N']
    N = sys_config['N']

    M_t = torch.tensor(sys_config['M'], dtype=torch.float32, device=device)
    K_t = torch.tensor(sys_config['K'], dtype=torch.float32, device=device).flatten()
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device).squeeze(-1)
    u_tensor = torch.tensor(u_vals, dtype=torch.float32, device=device)

    z = torch.zeros(sys_config['z_size'], dtype=torch.float32, device=device)
    x_hat_list = []

    hypernet.eval()
    lstm_state = None

    for i in range(N + 1):
        u_step = u_tensor[i].view(1, 1, 1)

        with torch.no_grad():
            delta_enc, delta_dec, lstm_state = hypernet.step(u_step, lstm_state)
            mod_dec_params = apply_weight_modulation_skip_bias(T_inv_base, delta_dec)
            z_t = z.unsqueeze(0)
            x_hat = torch.func.functional_call(
                T_inv_base, mod_dec_params, z_t
            ).cpu().numpy().squeeze()

        x_hat_list.append(x_hat)

        if i < N:
            y_i = y_tensor[i]
            def dynamics_lora(z_val):
                return torch.mv(M_t, z_val) + K_t * y_i
            k1 = dynamics_lora(z)
            k2 = dynamics_lora(z + 0.5 * dt * k1)
            k3 = dynamics_lora(z + 0.5 * dt * k2)
            k4 = dynamics_lora(z + dt * k3)
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return x_true, np.array(x_hat_list), t_eval


def evaluate_dynamic_lora(system, hypernet, T_base, T_inv_base, sys_config: dict,
                          input_types: List[str], n_trials: int, seed: int, device):
    """Evaluate dynamic LoRA observer. Returns (mean_results, per_trial_metrics)."""
    from smt.sampling_methods import LHS
    sampler = LHS(xlimits=sys_config['limits'])
    test_ics = sampler(n_trials)
    rng = np.random.RandomState(seed)

    results = {}
    per_trial = {}
    for input_type in input_types:
        input_gen = get_input_generator(input_type, 'id')
        metrics = []

        for ic in test_ics:
            input_gen.sample_params(rng)
            x_true, x_hat, t = simulate_observer_dynamic_lora(
                system, hypernet, T_base, T_inv_base, sys_config, ic, input_gen, device
            )

            error = np.linalg.norm(x_true - x_hat, axis=1)
            settle_idx = np.searchsorted(t, 5.0)

            metrics.append({
                'rmse_total': float(np.sqrt(np.mean(error**2))),
                'rmse_steady': float(np.sqrt(np.mean(error[settle_idx:]**2))),
                'max_error': float(np.max(error)),
            })

        results[input_type] = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        per_trial[input_type] = metrics

    return results, per_trial


def get_plot_trajectories_dynamic_lora(system, hypernet, T_base,
                                       T_inv_base, sys_config, device,
                                       window_size, seed):
    """Generate trajectories for all input types (dynamic LoRA, same IC)."""
    rng = np.random.RandomState(seed + 999)
    ic = rng.uniform(low=sys_config['limits'][:, 0], high=sys_config['limits'][:, 1])

    input_types = sys_config.get('natural_inputs', ['zero', 'constant', 'sinusoid', 'square'])
    trajectories = {}
    for input_type in input_types:
        input_gen = get_input_generator(input_type, 'id')
        input_gen.sample_params(rng)
        x_true, x_hat, t = simulate_observer_dynamic_lora(
            system, hypernet, T_base, T_inv_base, sys_config,
            ic, input_gen, device, window_size
        )
        trajectories[input_type] = (x_true, x_hat, t)
    return trajectories


# ---------------------------------------------------------------------------
# Boxplot visualisation
# ---------------------------------------------------------------------------

def plot_boxplot(all_per_trial: Dict[str, Dict[str, list]], sys_name: str,
                 save_path: Path, metric: str = 'rmse_steady'):
    """Generate boxplot of per-trial metrics across methods and input types.

    Args:
        all_per_trial: {method_name: {input_type: [{'rmse_total':..., 'rmse_steady':..., 'max_error':...}, ...]}}
        sys_name: system name for title
        save_path: path to save the figure
        metric: which metric to plot ('rmse_total', 'rmse_steady', 'max_error')
    """
    methods = list(all_per_trial.keys())
    if not methods:
        return

    # Collect all input types across methods
    input_types = []
    for m in methods:
        for it in all_per_trial[m]:
            if it not in input_types:
                input_types.append(it)

    n_inputs = len(input_types)
    if n_inputs == 0:
        return

    fig, axes = plt.subplots(1, n_inputs, figsize=(5 * n_inputs, 6), sharey=True)
    if n_inputs == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(methods), 3)))

    for ax_idx, input_type in enumerate(input_types):
        ax = axes[ax_idx]
        data_to_plot = []
        labels = []
        box_colors = []

        for m_idx, method in enumerate(methods):
            trials = all_per_trial[method].get(input_type, [])
            if trials:
                values = [t[metric] for t in trials]
                data_to_plot.append(values)
                labels.append(method)
                box_colors.append(colors[m_idx % len(colors)])

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                            showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(input_type, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel(metric, fontsize=12)
    fig.suptitle(f'{sys_name.upper()} - {metric} Distribution ({len(all_per_trial[methods[0]].get(input_types[0], []))} trials)',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
