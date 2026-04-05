"""
Plotting utilities for HyperKKL experiments.

- Loss history (supports autonomous/curriculum/simple)
- Time-series overlay with zoom insets
- Phase portrait / 3D attractor
- Space-time density heatmap (traffic systems)
- Boxplots for metric comparison

Standalone test:
    python -m src.plotting
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Method display names and colours
# ---------------------------------------------------------------------------

METHOD_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

METHOD_DISPLAY = {
    "autonomous": "Autonomous",
    "curriculum": "Curriculum",
    "augmented": "Augmented Observer",
    "full": "Dynamic Full",
    "lora": "Dynamic LoRA",
}


# ---------------------------------------------------------------------------
# Loss history
# ---------------------------------------------------------------------------

def plot_loss_history(loss_data, sys_name: str, method_name: str, save_path: Path):
    """Plot training loss history for a (system, method) pair."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if isinstance(loss_data, dict) and "encoder" in loss_data:
        ax.plot(range(1, len(loss_data["encoder"]) + 1), loss_data["encoder"],
                label="Encoder (T)", marker="o", markersize=3)
        ax.plot(range(1, len(loss_data["decoder"]) + 1), loss_data["decoder"],
                label="Decoder (T*)", marker="s", markersize=3)
        ax.legend()
    elif isinstance(loss_data, dict) and "losses" in loss_data:
        losses = loss_data["losses"]
        boundaries = loss_data["stage_boundaries"]
        stage_names = loss_data["stage_names"]
        ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, (b, name) in enumerate(zip(boundaries, stage_names)):
            ax.axvline(x=b + 1, color=colors[i % 4], linestyle="--", alpha=0.7, label=name)
        ax.legend(fontsize=8)
    else:
        losses = loss_data if isinstance(loss_data, list) else list(loss_data)
        ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{sys_name.upper()} - {method_name} - Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Time-series
# ---------------------------------------------------------------------------

def plot_time_series(x_true, x_hat, t, sys_name: str, method_name: str, save_path: Path):
    """Plot true vs estimated states over time."""
    n_states = x_true.shape[1]
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 3.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        label = f"$x_{{{i + 1}}}$"
        ax.plot(t, x_true[:, i], label=f"{label} true", linewidth=1.5)
        ax.plot(t, x_hat[:, i], "--", label=f"{label} est", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time", fontsize=12)
    fig.suptitle(f"{sys_name.upper()} - {method_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overlay time-series (multiple methods)
# ---------------------------------------------------------------------------

def _find_zoom_region(t, x_true, trajectories, state_idx, window_frac=0.12):
    skip = int(0.2 * len(t))
    window_len = max(int(window_frac * len(t)), 10)

    all_signals = [x_true[skip:, state_idx]]
    for _, x_hat, _ in trajectories.values():
        all_signals.append(x_hat[skip:, state_idx])
    stacked = np.stack(all_signals, axis=0)
    spread = stacked.max(axis=0) - stacked.min(axis=0)

    kernel = np.ones(window_len) / window_len
    avg_spread = np.convolve(spread, kernel, mode="valid")

    best_start = np.argmin(avg_spread)
    idx_start = skip + best_start
    idx_end = min(idx_start + window_len, len(t) - 1)
    return t[idx_start], t[idx_end]


def plot_overlay_timeseries(trajectories: Dict[str, Tuple], sys_name: str,
                            input_type: str, mode_label: str, save_path: Path):
    """Overlay plot: ground truth (black) + methods (coloured dashed) with zoom."""
    first = next(iter(trajectories))
    x_true, _, t = trajectories[first]
    n_states = x_true.shape[1]

    fig, axes = plt.subplots(n_states, 1, figsize=(12, 3.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, x_true[:, i], "k-", linewidth=2.0, label="Ground Truth")

        for cidx, (method, (_, x_hat, _)) in enumerate(trajectories.items()):
            color = METHOD_COLOURS[cidx % len(METHOD_COLOURS)]
            display = METHOD_DISPLAY.get(method, method)
            ax.plot(t, x_hat[:, i], "--", color=color, linewidth=1.4, alpha=0.85, label=display)

        ax.set_ylabel(f"$x_{{{i + 1}}}$", fontsize=13)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

        try:
            t_s, t_e = _find_zoom_region(t, x_true, trajectories, i)
            mask = (t >= t_s) & (t <= t_e)
            y_vals = np.concatenate([x_true[mask, i]] + [xh[mask, i] for _, xh, _ in trajectories.values()])
            y_min, y_max = y_vals.min(), y_vals.max()
            margin = 0.15 * max(y_max - y_min, 0.1)

            loc = [0.02, 0.05, 0.35, 0.45] if i == 0 else [0.62, 0.50, 0.35, 0.45]
            axins = ax.inset_axes(loc)
            axins.plot(t[mask], x_true[mask, i], "k-", linewidth=2.0)
            for cidx, (_, (_, xh, _)) in enumerate(trajectories.items()):
                axins.plot(t[mask], xh[mask, i], "--", color=METHOD_COLOURS[cidx % len(METHOD_COLOURS)],
                           linewidth=1.4, alpha=0.85)
            axins.set_xlim(t_s, t_e)
            axins.set_ylim(y_min - margin, y_max + margin)
            axins.grid(True, alpha=0.3)
            axins.tick_params(labelsize=7)
            ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)
        except Exception:
            pass

    axes[-1].set_xlabel("Time", fontsize=13)
    fig.suptitle(f"{sys_name.upper()} - {input_type} ({mode_label})", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase portrait / attractor
# ---------------------------------------------------------------------------

def plot_attractor(x_true, x_hat, sys_name: str, method_name: str, save_path: Path):
    n_states = x_true.shape[1]

    if n_states == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x_true[:, 0], x_true[:, 1], label="True", linewidth=1.2)
        ax.plot(x_hat[:, 0], x_hat[:, 1], "--", label="Estimated", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{sys_name.upper()} - {method_name} - Phase Portrait")
    elif n_states >= 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], label="True", linewidth=1.0)
        ax.plot(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], "--", label="Estimated", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        ax.legend()
        ax.set_title(f"{sys_name.upper()} - {method_name} - Attractor (3D)")
    else:
        return

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Density heatmap (traffic)
# ---------------------------------------------------------------------------

def plot_density(x_true, x_hat, t, sys_name: str, method_name: str, save_path: Path):
    n_cells = x_true.shape[1]
    cell_labels = [f"Cell {i + 1}" for i in range(n_cells)]
    vmin = min(x_true.min(), x_hat.min())
    vmax = max(x_true.max(), x_hat.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax_idx, (data, title) in enumerate([
        (x_true, "Ground Truth"), (x_hat, "Estimated")
    ]):
        im = axes[ax_idx].imshow(
            data.T, aspect="auto", origin="lower", norm=norm,
            extent=[t[0], t[-1], 0.5, n_cells + 0.5], cmap="YlOrRd")
        axes[ax_idx].set_yticks(range(1, n_cells + 1))
        axes[ax_idx].set_yticklabels(cell_labels)
        axes[ax_idx].set_title(f"{title} Density")
        fig.colorbar(im, ax=axes[ax_idx])

    error = np.abs(x_true - x_hat)
    im2 = axes[2].imshow(error.T, aspect="auto", origin="lower",
                         extent=[t[0], t[-1], 0.5, n_cells + 0.5], cmap="hot")
    axes[2].set_yticks(range(1, n_cells + 1))
    axes[2].set_yticklabels(cell_labels)
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Time (s)")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(f"{sys_name.upper()} - {method_name} - Density", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Boxplots for metric comparison
# ---------------------------------------------------------------------------

def plot_boxplot(all_per_trial: Dict[str, Dict[str, list]], sys_name: str,
                 save_path: Path, metric: str = "rmse_steady"):
    """Boxplot of per-trial metrics across methods and input types."""
    methods = list(all_per_trial.keys())
    if not methods:
        return

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
        data_to_plot, labels, box_colors = [], [], []

        for m_idx, method in enumerate(methods):
            trials = all_per_trial[method].get(input_type, [])
            if trials:
                data_to_plot.append([t[metric] for t in trials])
                labels.append(method)
                box_colors.append(colors[m_idx % len(colors)])

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                            showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(input_type, fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel(metric, fontsize=12)
    fig.suptitle(f"{sys_name.upper()} - {metric} Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_stacked(all_per_trial: Dict[str, Dict[str, list]], sys_name: str,
                         save_path: Path):
    """Two-panel stacked boxplot: RMSE (top) and SMAPE (bottom)."""
    methods = list(all_per_trial.keys())
    if not methods:
        return

    method_colors = ["#FF00FF", "#0000FF", "#006400", "#FFFF00",
                     "#FF6347", "#00CED1", "#FF8C00", "#8A2BE2"]

    rmse_data, smape_data, labels, box_colors = [], [], [], []

    for m_idx, method in enumerate(methods):
        rmse_vals, smape_vals = [], []
        for trials in all_per_trial[method].values():
            for t in trials:
                rmse_vals.append(t.get("rmse_total", 0.0))
                smape_vals.append(t.get("smape", 0.0))
        if rmse_vals:
            rmse_data.append(rmse_vals)
            smape_data.append(smape_vals)
            labels.append(method)
            box_colors.append(method_colors[m_idx % len(method_colors)])

    if not rmse_data:
        return

    fig, (ax_rmse, ax_smape) = plt.subplots(2, 1, figsize=(max(3 * len(methods), 8), 10),
                                             sharex=True)

    for ax, data, ylabel, title in [
        (ax_rmse, rmse_data, "RMSE", "RMSE Distribution"),
        (ax_smape, smape_data, "SMAPE (%)", "SMAPE Distribution"),
    ]:
        bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=True,
                        flierprops=dict(marker="o", markersize=4, alpha=0.5,
                                        markerfacecolor="none", markeredgecolor="grey"),
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(color="grey"), capprops=dict(color="grey"))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
            patch.set_edgecolor("black")
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

    ax_smape.tick_params(axis="x", rotation=30, labelsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = np.linspace(0, 10, 500)
    x1 = np.column_stack([np.sin(t), np.cos(t)])
    x2 = x1 + np.random.normal(0, 0.1, x1.shape)

    plot_time_series(x1, x2, t, "test", "demo", Path("/tmp/test_ts.png"))
    plot_attractor(x1, x2, "test", "demo", Path("/tmp/test_attr.png"))
    plot_loss_history([1.0, 0.5, 0.3, 0.2, 0.15], "test", "demo", Path("/tmp/test_loss.png"))
    print("Plots saved to /tmp/test_*.png")
