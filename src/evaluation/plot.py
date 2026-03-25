"""
Plotting utilities: loss histories, time-series, phase portraits / attractors,
and space-time density plots for traffic systems.
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_loss_history(loss_data, sys_name: str, method_name: str, save_path: Path):
    """Plot training loss history for a (system, method) pair."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if isinstance(loss_data, dict) and 'encoder' in loss_data:
        # Autonomous: separate encoder and decoder curves
        ax.plot(range(1, len(loss_data['encoder']) + 1), loss_data['encoder'],
                label='Encoder (T)', marker='o', markersize=3)
        ax.plot(range(1, len(loss_data['decoder']) + 1), loss_data['decoder'],
                label='Decoder (T*)', marker='s', markersize=3)
        ax.legend()
    elif isinstance(loss_data, dict) and 'losses' in loss_data:
        # Curriculum: losses with stage boundaries
        losses = loss_data['losses']
        boundaries = loss_data['stage_boundaries']
        stage_names = loss_data['stage_names']
        ax.plot(range(1, len(losses) + 1), losses, marker='o', markersize=3)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (bdry, name) in enumerate(zip(boundaries, stage_names)):
            ax.axvline(x=bdry + 1, color=colors[i % len(colors)],
                       linestyle='--', alpha=0.7, label=name)
        ax.legend(fontsize=8)
    else:
        # Simple loss list
        losses = loss_data if isinstance(loss_data, list) else loss_data
        ax.plot(range(1, len(losses) + 1), losses, marker='o', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{sys_name.upper()} - {method_name} - Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_time_series(x_true: np.ndarray, x_hat: np.ndarray, t: np.ndarray,
                     sys_name: str, method_name: str, save_path: Path):
    """Plot true vs estimated states over time (line plot)."""
    n_states = x_true.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 3.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    state_labels = [f'$x_{{{i+1}}}$' for i in range(n_states)]

    for i, ax in enumerate(axes):
        ax.plot(t, x_true[:, i], label=f'{state_labels[i]} true', linewidth=1.5)
        ax.plot(t, x_hat[:, i], '--', label=f'{state_labels[i]} estimated',
                linewidth=1.5, alpha=0.8)
        ax.set_ylabel(state_labels[i], fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=12)
    fig.suptitle(f'{sys_name.upper()} - {method_name} - State Estimation', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_attractor(x_true: np.ndarray, x_hat: np.ndarray,
                   sys_name: str, method_name: str, save_path: Path):
    """Plot phase portrait / attractor: 2D for 2-state systems, 3D for 3-state systems."""
    n_states = x_true.shape[1]

    if n_states == 2:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.plot(x_true[:, 0], x_true[:, 1], label='True', linewidth=1.2)
        ax.plot(x_hat[:, 0], x_hat[:, 1], '--', label='Estimated',
                linewidth=1.2, alpha=0.8)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{sys_name.upper()} - {method_name} - Phase Portrait (2D)', fontsize=13)
        fig.tight_layout()

    elif n_states >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F811
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2],
                label='True', linewidth=1.0)
        ax.plot(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], '--',
                label='Estimated', linewidth=1.0, alpha=0.8)
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_zlabel('$x_3$', fontsize=11)
        ax.legend(fontsize=10)
        ax.set_title(f'{sys_name.upper()} - {method_name} - Attractor (3D)', fontsize=13)
        fig.tight_layout()

    else:
        return  # Single state, skip attractor plot

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_density(x_true: np.ndarray, x_hat: np.ndarray, t: np.ndarray,
                 sys_name: str, method_name: str, save_path: Path,
                 cell_labels=None):
    """Plot space-time density heatmaps for traffic systems.

    Creates two side-by-side heatmaps: ground truth density and estimated
    density across cells (y-axis) over time (x-axis), plus a difference plot.

    Args:
        x_true: (N+1, n_cells) ground truth density
        x_hat:  (N+1, n_cells) estimated density
        t:      (N+1,) time array
        sys_name: system name for title
        method_name: method name for title
        save_path: path to save figure
        cell_labels: optional list of cell names
    """
    n_cells = x_true.shape[1]
    if cell_labels is None:
        cell_labels = [f'Cell {i+1}' for i in range(n_cells)]

    # Shared color scale
    vmin = min(x_true.min(), x_hat.min())
    vmax = max(x_true.max(), x_hat.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Ground truth density
    im0 = axes[0].imshow(
        x_true.T, aspect='auto', origin='lower', norm=norm,
        extent=[t[0], t[-1], 0.5, n_cells + 0.5], cmap='YlOrRd')
    axes[0].set_yticks(range(1, n_cells + 1))
    axes[0].set_yticklabels(cell_labels)
    axes[0].set_title('Ground Truth Density (veh/m)')
    fig.colorbar(im0, ax=axes[0], label=r'$\rho$ (veh/m)')

    # Estimated density
    im1 = axes[1].imshow(
        x_hat.T, aspect='auto', origin='lower', norm=norm,
        extent=[t[0], t[-1], 0.5, n_cells + 0.5], cmap='YlOrRd')
    axes[1].set_yticks(range(1, n_cells + 1))
    axes[1].set_yticklabels(cell_labels)
    axes[1].set_title('Estimated Density (veh/m)')
    fig.colorbar(im1, ax=axes[1], label=r'$\rho$ (veh/m)')

    # Error (difference)
    error = np.abs(x_true - x_hat)
    im2 = axes[2].imshow(
        error.T, aspect='auto', origin='lower',
        extent=[t[0], t[-1], 0.5, n_cells + 0.5], cmap='hot')
    axes[2].set_yticks(range(1, n_cells + 1))
    axes[2].set_yticklabels(cell_labels)
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Time (s)')
    fig.colorbar(im2, ax=axes[2], label=r'$|\rho_{true} - \rho_{est}|$')

    fig.suptitle(f'{sys_name.upper()} - {method_name} - Density Plot', fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
