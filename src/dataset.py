"""
Unified data generation for Phase 1 (autonomous) and Phase 2 (non-autonomous).

Key design: training and evaluation use the SAME generate_trajectories function,
only differing in signal mode ("train" vs "id" vs "ood").

Standalone test:
    python -m src.dataset --system duffing --phase 2
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.systems import create_system, SYSTEM_CLASSES
from src.signals import create_signal
from src.config import SystemConfig, load_config


# ---------------------------------------------------------------------------
# Vectorised RK4 trajectory generation (shared by Phase 1 and 2)
# ---------------------------------------------------------------------------

def _rk4_integrate(system, ics, t_eval, u_grid):
    """Vectorised RK4 integration for a batch of trajectories.

    Args:
        system: System instance (noise disabled)
        ics: (n_traj, x_size) initial conditions
        t_eval: (N+1,) time array
        u_grid: (N+1, n_traj) input signals

    Returns:
        x_traj: (N+1, n_traj, x_size) trajectories as numpy
    """
    N = len(t_eval) - 1
    dt = t_eval[1] - t_eval[0]

    x = torch.tensor(ics, dtype=torch.float64)
    u_t = torch.tensor(u_grid, dtype=torch.float64)
    traj = [x.clone()]

    for i in range(N):
        u_i = u_t[i]
        u_half = (u_t[i] + u_t[min(i + 1, N)]) / 2.0
        u_next = u_t[min(i + 1, N)]

        k1 = system.function(t_eval[i], u_i, x)
        k2 = system.function(t_eval[i] + dt / 2, u_half, x + dt / 2 * k1)
        k3 = system.function(t_eval[i] + dt / 2, u_half, x + dt / 2 * k2)
        k4 = system.function(t_eval[min(i + 1, N)], u_next, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.clone())

    return torch.stack(traj).numpy()  # (N+1, n_traj, x_size)


# ---------------------------------------------------------------------------
# Phase 1: Autonomous dataset (z via linear ODE)
# ---------------------------------------------------------------------------

def _vectorized_z_sim(output, M, K, ic_z, a, b, N):
    """Vectorised RK4 for linear z-dynamics: z' = Mz + Ky."""
    h = (b - a) / N
    M_T = M.T
    scalar_y = output.ndim == 2

    if scalar_y:
        Y = output[:, 1:, np.newaxis]
    else:
        Y = output[:, 1:]

    Ky_all = np.matmul(Y, K.T)
    Z = ic_z
    z_traj = [Z]

    for i in range(Ky_all.shape[1]):
        ky = Ky_all[:, i, :]
        k1 = Z @ M_T + ky
        k2 = (Z + 0.5 * h * k1) @ M_T + ky
        k3 = (Z + 0.5 * h * k2) @ M_T + ky
        k4 = (Z + h * k3) @ M_T + ky
        Z = Z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        z_traj.append(Z)

    return np.stack(z_traj, axis=1)  # (n_traj, N+1, z_size)


class Phase1Dataset(Dataset):
    """Phase 1 autonomous dataset: paired (x, z, y) samples.

    Generates x-trajectories, computes z via the linear observer ODE,
    and pairs them for training T (encoder) and T* (decoder).
    """

    def __init__(self, system, sys_config: SystemConfig, num_ic: int,
                 seed: int = 42, pretrained_T=None, pinn_mode: str = "split_traj"):
        self.system = system
        self.sys_config = sys_config
        M = sys_config.M_np
        K = sys_config.K_np

        system.add_noise = False
        ic = system.sample_ic(sys_config.limits_np, num_ic, seed)
        t_eval = np.linspace(sys_config.time_start, sys_config.time_end, sys_config.n_steps + 1)
        u_grid = np.zeros((len(t_eval), num_ic))

        # Integrate x-dynamics
        x_traj = _rk4_integrate(system, ic, t_eval, u_grid)  # (N+1, n_traj, x_size)
        x_traj = x_traj.transpose(1, 0, 2)  # (n_traj, N+1, x_size)

        # Compute y for each trajectory
        n_traj, n_steps_p1, x_size = x_traj.shape
        x_flat = torch.tensor(x_traj.reshape(-1, x_size), dtype=torch.float64)
        y_flat = system.output(x_flat).numpy()
        y_size = y_flat.shape[-1] if y_flat.ndim > 1 else 1
        if y_flat.ndim == 1:
            y_flat = y_flat[:, np.newaxis]
        y_traj = y_flat.reshape(n_traj, n_steps_p1, y_size)

        # Compute z-trajectories
        if pretrained_T is not None:
            device = next(pretrained_T.parameters()).device
            with torch.no_grad():
                z0 = pretrained_T(torch.tensor(ic, dtype=torch.float32).to(device)).cpu().numpy()
        else:
            # Use random z0 with forward-time cutoff
            z0 = np.random.RandomState(seed).rand(num_ic, sys_config.z_size)

        z_traj = _vectorized_z_sim(
            y_traj if y_traj.shape[-1] > 1 else y_traj.squeeze(-1),
            M, K, z0, sys_config.time_start, sys_config.time_end, sys_config.n_steps)

        if pretrained_T is None:
            # Cutoff transient (10%)
            cutoff = int(0.1 * n_steps_p1)
            x_traj = x_traj[:, cutoff:]
            z_traj = z_traj[:, cutoff:]
            y_traj = y_traj[:, cutoff:]

        # Flatten to (total_samples, dim)
        n_samples = x_traj.shape[0] * x_traj.shape[1]
        x_flat = torch.tensor(x_traj.reshape(n_samples, x_size), dtype=torch.float32)
        z_flat = torch.tensor(z_traj.reshape(n_samples, sys_config.z_size), dtype=torch.float32)
        y_flat_t = torch.tensor(y_traj.reshape(n_samples, y_size), dtype=torch.float32).squeeze(-1)

        if pinn_mode == "split_traj":
            half = n_samples // 2
            self.x_data = x_flat[::2]
            self.z_data = z_flat[::2]
            self.y_data = y_flat_t[::2]
            self.x_data_ph = x_flat[1::2]
            self.y_data_ph = y_flat_t[1::2]
        else:  # "no_physics" mode for decoder
            self.x_data = x_flat
            self.z_data = z_flat
            self.y_data = y_flat_t
            self.x_data_ph = x_flat
            self.y_data_ph = y_flat_t

        # Statistics for normalization
        self.mean_x = self.x_data.mean(0)
        self.std_x = self.x_data.std(0).clamp(min=1e-8)
        self.mean_z = self.z_data.mean(0)
        self.std_z = self.z_data.std(0).clamp(min=1e-8)
        self.mean_x_ph = self.x_data_ph.mean(0)
        self.std_x_ph = self.x_data_ph.std(0).clamp(min=1e-8)
        self.mean_z_ph = torch.zeros(sys_config.z_size)
        self.std_z_ph = torch.ones(sys_config.z_size)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.z_data[idx], self.y_data[idx],
                self.x_data_ph[idx], self.y_data_ph[idx])


# ---------------------------------------------------------------------------
# Phase 2: Non-autonomous data (windowed samples)
# ---------------------------------------------------------------------------

def _phase2_worker(args):
    """Worker for parallel Phase 2 data generation."""
    (n_traj, t_span, dt, window_size, limits,
     seed, signal_type, signal_mode, sys_cls_name, init_args) = args

    rng = np.random.RandomState(seed)
    a, b = t_span
    N = int((b - a) / dt)
    t_eval = np.linspace(a, b, N + 1)

    sys_cls = SYSTEM_CLASSES[sys_cls_name]
    system = sys_cls(**init_args)
    system.add_noise = False
    x_size = system.x_size

    ics = rng.uniform(low=limits[:, 0], high=limits[:, 1], size=(n_traj, x_size))

    signal_gen = create_signal(signal_type, signal_mode)
    u_grid = np.zeros((N + 1, n_traj))
    for i in range(n_traj):
        signal_gen.sample_params(rng)
        u_grid[:, i] = signal_gen(t_eval)

    x_traj = _rk4_integrate(system, ics, t_eval, u_grid)  # (N+1, n_traj, x_size)

    # Filter diverged trajectories
    traj_max = np.max(np.abs(x_traj), axis=(0, 2))
    valid = traj_max < 1e6
    if not np.any(valid):
        return {k: np.empty((0,) + s, dtype=np.float32)
                for k, s in [("x", (x_size,)), ("y", ()), ("u_window", (window_size,)),
                              ("u_window_prev", (window_size,)), ("u_current", ()),
                              ("dxdt", (x_size,))]}

    # Collect samples starting at window_size+1 for u_window_prev validity
    data = {k: [] for k in ["x", "y", "u_window", "u_window_prev", "u_current", "dxdt"]}

    for t_idx in range(window_size + 1, N):
        x_now = x_traj[t_idx]
        u_now = u_grid[t_idx]

        x_t = torch.tensor(x_now, dtype=torch.float64)
        y_now = system.output(x_t).squeeze(-1).numpy()
        u_t = torch.tensor(u_now, dtype=torch.float64)
        dxdt = system.function(t_eval[t_idx], u_t, x_t).numpy()

        u_win = u_grid[t_idx - window_size:t_idx].T
        u_win_prev = u_grid[t_idx - window_size - 1:t_idx - 1].T

        data["x"].append(x_now[valid])
        data["y"].append(y_now[valid])
        data["u_window"].append(u_win[valid])
        data["u_window_prev"].append(u_win_prev[valid])
        data["u_current"].append(u_now[valid])
        data["dxdt"].append(dxdt[valid])

    return {
        "x": np.vstack(data["x"]).astype(np.float32),
        "y": np.concatenate(data["y"]).astype(np.float32),
        "u_window": np.vstack(data["u_window"]).astype(np.float32),
        "u_window_prev": np.vstack(data["u_window_prev"]).astype(np.float32),
        "u_current": np.concatenate(data["u_current"]).astype(np.float32),
        "dxdt": np.vstack(data["dxdt"]).astype(np.float32),
    }


def generate_phase2_data(sys_config: SystemConfig, signal_types: list,
                         n_trajectories: int, window_size: int,
                         seed: int = 42, signal_mode: str = "train") -> dict:
    """Generate Phase 2 training/evaluation data with multiprocessing.

    Uses the same code path for training and evaluation - only signal_mode differs.

    Returns:
        dict with tensors: x, y, u_window, u_window_prev, u_current, dxdt
    """
    dt = sys_config.dt
    t_span = (sys_config.time_start, sys_config.time_end)

    n_cpu = min(multiprocessing.cpu_count(), 16)
    traj_per_type = max(1, n_trajectories // len(signal_types))

    chunks = []
    curr_seed = seed
    for sig_type in signal_types:
        n_chunks = min(n_cpu, traj_per_type)
        traj_per_chunk = max(1, traj_per_type // n_chunks)
        for i in range(n_chunks):
            count = traj_per_chunk + (1 if i < traj_per_type % n_chunks else 0)
            if count > 0:
                chunks.append((
                    count, t_span, dt, window_size,
                    sys_config.limits_np, curr_seed, sig_type, signal_mode,
                    sys_config.class_name, sys_config.init_args,
                ))
                curr_seed += count

    print(f"  Generating Phase 2 data: {n_trajectories} traj, "
          f"{len(chunks)} chunks, mode={signal_mode}")

    with multiprocessing.Pool(processes=min(n_cpu, len(chunks))) as pool:
        results = pool.map(_phase2_worker, chunks)

    # Concatenate results
    final = {}
    for key in results[0]:
        combined = np.concatenate([r[key] for r in results], axis=0)
        tensor = torch.tensor(combined)
        if key in ("y", "u_current"):
            tensor = tensor.unsqueeze(-1)
        elif key in ("u_window", "u_window_prev"):
            tensor = tensor.unsqueeze(-1)
        final[key] = tensor

    print(f"  Generated {final['x'].shape[0]} samples")
    return final


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset generation demo")
    parser.add_argument("--system", default="duffing")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()

    cfg = load_config(args.system)
    system = create_system(cfg.system)

    if args.phase == 1:
        ds = Phase1Dataset(system, cfg.system, num_ic=50, seed=42)
        print(f"Phase 1 dataset: {len(ds)} samples")
        print(f"  x shape: {ds.x_data.shape}")
        print(f"  z shape: {ds.z_data.shape}")
        sample = ds[0]
        print(f"  Sample: x={sample[0].shape}, z={sample[1].shape}, y={sample[2].shape}")
    else:
        data = generate_phase2_data(
            cfg.system, cfg.system.natural_inputs,
            n_trajectories=50, window_size=100, seed=42)
        for k, v in data.items():
            print(f"  {k}: {v.shape}")
