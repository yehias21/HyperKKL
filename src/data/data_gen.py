"""
Parallel data generation for HyperKKL training.

Uses System classes from Systems.py for dynamics (single source of truth).
RK4 integration with torch batch mode, matching the autonomous pipeline.
"""

import numpy as np
import torch
import multiprocessing

from hyperkkl.src.data.signals import get_input_generator


def _data_generation_worker(args):
    """Worker function for parallel data generation.

    Creates a System instance inside the worker and uses its function()
    and output() methods in torch batch mode for vectorised RK4.
    """
    (n_traj, t_span, dt, window_size, sample_space,
     seed, input_type, sys_cls, init_args) = args

    rng = np.random.RandomState(seed)
    a, b = t_span
    N = int((b - a) / dt)
    t_eval = np.linspace(a, b, N + 1)

    # Construct system in worker (clean, noise-free for data generation)
    system = sys_cls(**init_args)
    system.add_noise = False
    x_size = system.x_size

    # Sample initial conditions
    ics = rng.uniform(low=sample_space[:, 0], high=sample_space[:, 1],
                      size=(n_traj, x_size))

    # Generate input signals on the time grid
    input_gen = get_input_generator(input_type, 'train')
    u_grid = np.zeros((N + 1, n_traj))
    for i in range(n_traj):
        input_gen.sample_params(rng)
        u_grid[:, i] = input_gen(t_eval)

    # --- Vectorised RK4 using system.function (torch batch mode) ----------
    # Keep state as a torch tensor throughout integration for efficiency.
    x = torch.tensor(ics, dtype=torch.float64)           # (n_traj, x_size)
    u_grid_t = torch.tensor(u_grid, dtype=torch.float64) # (N+1, n_traj)

    x_traj = [x.clone()]

    for i in range(N):
        u_i    = u_grid_t[i]
        u_half = (u_grid_t[i] + u_grid_t[min(i + 1, N)]) / 2.0
        u_next = u_grid_t[min(i + 1, N)]

        h = dt
        k1 = system.function(t_eval[i],          u_i,    x)
        k2 = system.function(t_eval[i] + h / 2,  u_half, x + h / 2 * k1)
        k3 = system.function(t_eval[i] + h / 2,  u_half, x + h / 2 * k2)
        k4 = system.function(t_eval[min(i+1,N)], u_next, x + h * k3)

        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_traj.append(x.clone())

    # (N+1, n_traj, x_size) as numpy
    x_traj_np = torch.stack(x_traj).numpy()

    # --- Filter diverged trajectories before collecting samples -------------
    # Lorenz/Rossler can diverge to values exceeding float32 range (~3.4e38).
    # Mark trajectories as valid only if all state values stay within safe bounds.
    FLOAT32_SAFE = 1e6  # conservative bound; real trajectories stay well below this
    traj_max = np.max(np.abs(x_traj_np), axis=(0, 2))  # (n_traj,)
    valid_mask = traj_max < FLOAT32_SAFE  # per-trajectory boolean mask

    if not np.any(valid_mask):
        # All trajectories diverged — return empty arrays with correct shapes
        return {
            'x': np.empty((0, x_size), dtype=np.float32),
            'y': np.empty((0,), dtype=np.float32),
            'u_window': np.empty((0, window_size), dtype=np.float32),
            'u_window_prev': np.empty((0, window_size), dtype=np.float32),
            'u_current': np.empty((0,), dtype=np.float32),
            'dxdt': np.empty((0, x_size), dtype=np.float32),
        }

    # --- Collect training samples -----------------------------------------
    # Start at window_size+1 so that u_window_prev (shifted back by 1) is valid.
    valid_indices = range(window_size + 1, N)
    data_x, data_y, data_u_window, data_u_current, data_dxdt = [], [], [], [], []
    data_u_window_prev = []

    for t_idx in valid_indices:
        x_now = x_traj_np[t_idx]                        # (n_traj, x_size)
        u_now = u_grid[t_idx]                            # (n_traj,)

        # Output via system.output (batch torch mode, noise-free)
        x_t = torch.tensor(x_now, dtype=torch.float64)
        y_now = system.output(x_t).squeeze(-1).numpy()   # (n_traj,)

        # dx/dt via system.function (batch torch mode)
        u_t = torch.tensor(u_now, dtype=torch.float64)
        dxdt_now = system.function(
            t_eval[t_idx], u_t, x_t).numpy()             # (n_traj, x_size)

        u_win = u_grid[t_idx - window_size:t_idx].T      # (n_traj, window_size)
        u_win_prev = u_grid[t_idx - window_size - 1:t_idx - 1].T  # shifted back by 1 step

        # Only keep samples from non-diverged trajectories
        data_x.append(x_now[valid_mask])
        data_y.append(y_now[valid_mask])
        data_u_window.append(u_win[valid_mask])
        data_u_window_prev.append(u_win_prev[valid_mask])
        data_u_current.append(u_now[valid_mask])
        data_dxdt.append(dxdt_now[valid_mask])

    return {
        'x': np.vstack(data_x).astype(np.float32),
        'y': np.concatenate(data_y).astype(np.float32),
        'u_window': np.vstack(data_u_window).astype(np.float32),
        'u_window_prev': np.vstack(data_u_window_prev).astype(np.float32),
        'u_current': np.concatenate(data_u_current).astype(np.float32),
        'dxdt': np.vstack(data_dxdt).astype(np.float32),
    }


def _trajectory_data_worker(args):
    """Worker function for parallel trajectory-level data generation.

    Returns full trajectories (x, y, u) instead of individual samples.
    Used by the hypernet-GRU method which needs sequential data.
    """
    (n_traj, t_span, dt, sample_space,
     seed, input_type, sys_cls, init_args) = args

    rng = np.random.RandomState(seed)
    a, b = t_span
    N = int((b - a) / dt)
    t_eval = np.linspace(a, b, N + 1)

    system = sys_cls(**init_args)
    system.add_noise = False
    x_size = system.x_size

    ics = rng.uniform(low=sample_space[:, 0], high=sample_space[:, 1],
                      size=(n_traj, x_size))

    input_gen = get_input_generator(input_type, 'train')
    u_grid = np.zeros((N + 1, n_traj))
    for i in range(n_traj):
        input_gen.sample_params(rng)
        u_grid[:, i] = input_gen(t_eval)

    # Vectorised RK4
    x = torch.tensor(ics, dtype=torch.float64)
    u_grid_t = torch.tensor(u_grid, dtype=torch.float64)

    x_traj = [x.clone()]
    for i in range(N):
        u_i    = u_grid_t[i]
        u_half = (u_grid_t[i] + u_grid_t[min(i + 1, N)]) / 2.0
        u_next = u_grid_t[min(i + 1, N)]
        h = dt
        k1 = system.function(t_eval[i],          u_i,    x)
        k2 = system.function(t_eval[i] + h / 2,  u_half, x + h / 2 * k1)
        k3 = system.function(t_eval[i] + h / 2,  u_half, x + h / 2 * k2)
        k4 = system.function(t_eval[min(i+1,N)], u_next, x + h * k3)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_traj.append(x.clone())

    # x_traj_np: (N+1, n_traj, x_size)
    x_traj_np = torch.stack(x_traj).numpy()

    # Filter diverged trajectories before float32 cast
    FLOAT32_SAFE = 1e6
    traj_max = np.max(np.abs(x_traj_np), axis=(0, 2))  # (n_traj,)
    valid_mask = traj_max < FLOAT32_SAFE

    if not np.any(valid_mask):
        return {
            'x': np.empty((0, N + 1, x_size), dtype=np.float32),
            'y': np.empty((0, N + 1, 1), dtype=np.float32),
            'u': np.empty((0, N + 1, 1), dtype=np.float32),
            'dt': dt,
        }

    x_traj_np = x_traj_np[:, valid_mask, :]  # (N+1, n_valid, x_size)
    u_grid_valid = u_grid[:, valid_mask]      # (N+1, n_valid)

    # Compute measurements y for entire trajectory
    n_valid = x_traj_np.shape[1]
    x_all = torch.tensor(x_traj_np.reshape(-1, x_size), dtype=torch.float64)
    y_all = system.output(x_all).numpy()  # ((N+1)*n_valid, y_size)
    y_size = y_all.shape[-1]
    y_traj_np = y_all.reshape(N + 1, n_valid, y_size)

    return {
        # Transpose to (n_valid, N+1, features)
        'x': x_traj_np.transpose(1, 0, 2).astype(np.float32),
        'y': y_traj_np.transpose(1, 0, 2).astype(np.float32),
        'u': u_grid_valid.T[:, :, np.newaxis].astype(np.float32),  # (n_valid, N+1, 1)
        'dt': dt,
    }


def generate_trajectory_data_parallel(system_name: str, sys_config: dict,
                                       input_types: list, n_trajectories: int,
                                       seed: int = 42):
    """Generate full trajectory data for hypernet-GRU training.

    Returns:
        dict with keys 'x', 'y', 'u' as tensors of shape (n_traj, T, dim)
        and 'dt' (float).
    """
    dt = (sys_config['b'] - sys_config['a']) / sys_config['N']
    t_span = (sys_config['a'], sys_config['b'])
    sys_cls = sys_config['class']
    init_args = sys_config['init_args']

    n_cpu = multiprocessing.cpu_count()
    n_processes = min(n_cpu, n_trajectories, 16)
    traj_per_type = n_trajectories // len(input_types)

    chunks = []
    curr_seed = seed
    for inp_type in input_types:
        traj_per_proc = max(1, traj_per_type // n_processes)
        n_chunks = min(n_processes, traj_per_type)
        for i in range(n_chunks):
            count = traj_per_proc + (1 if i < traj_per_type % n_chunks else 0)
            if count > 0:
                chunks.append((
                    count, t_span, dt, sys_config['limits'],
                    curr_seed, inp_type, sys_cls, init_args
                ))
                curr_seed += count

    print(f"Generating trajectory data for {system_name}: {n_trajectories} trajectories, "
          f"{len(chunks)} chunks, {n_processes} processes")

    with multiprocessing.Pool(processes=n_processes) as pool:
        results_list = pool.map(_trajectory_data_worker, chunks)

    final_data = {
        'x': torch.tensor(np.concatenate([r['x'] for r in results_list], axis=0)),
        'y': torch.tensor(np.concatenate([r['y'] for r in results_list], axis=0)),
        'u': torch.tensor(np.concatenate([r['u'] for r in results_list], axis=0)),
        'dt': results_list[0]['dt'],
    }

    print(f"  Generated {final_data['x'].shape[0]} trajectories "
          f"of length {final_data['x'].shape[1]}")
    return final_data


def generate_hyperkkl_data_parallel(system_name: str, sys_config: dict,
                                     input_types: list, n_trajectories: int,
                                     window_size: int, seed: int = 42):
    """Generate training data with multiprocessing."""
    dt = (sys_config['b'] - sys_config['a']) / sys_config['N']
    t_span = (sys_config['a'], sys_config['b'])

    sys_cls = sys_config['class']
    init_args = sys_config['init_args']

    n_cpu = multiprocessing.cpu_count()
    n_processes = min(n_cpu, n_trajectories, 16)

    traj_per_type = n_trajectories // len(input_types)

    chunks = []
    curr_seed = seed
    for inp_type in input_types:
        traj_per_proc = max(1, traj_per_type // n_processes)
        n_chunks = min(n_processes, traj_per_type)

        for i in range(n_chunks):
            count = traj_per_proc + (1 if i < traj_per_type % n_chunks else 0)
            if count > 0:
                chunks.append((
                    count, t_span, dt, window_size, sys_config['limits'],
                    curr_seed, inp_type, sys_cls, init_args
                ))
                curr_seed += count

    print(f"Generating data for {system_name}: {n_trajectories} trajectories, "
          f"{len(chunks)} chunks, {n_processes} processes")

    with multiprocessing.Pool(processes=n_processes) as pool:
        results_list = pool.map(_data_generation_worker, chunks)

    final_data = {}
    keys = results_list[0].keys()

    for key in keys:
        combined = np.concatenate([res[key] for res in results_list], axis=0)
        tensor_data = torch.tensor(combined)

        if key == 'y':
            tensor_data = tensor_data.unsqueeze(-1)
        elif key in ['u_window', 'u_window_prev', 'u_current']:
            tensor_data = tensor_data.unsqueeze(-1)

        final_data[key] = tensor_data

    print(f"  Generated {final_data['x'].shape[0]} samples")
    return final_data
