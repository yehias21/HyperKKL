"""Fig. 3 and the Appendix E time series of the ICLR workshop paper, from node ICLR-figs.

    python experiments/iclr_figures.py [--out experiments/figures] [--episode 0]

Reads runs/<system>/seed0/traj.npz (first 4 clean episodes of each fam_* tier, saved by
evaluate.TRAJ_TIERS) and writes the 17 PNGs under the paper's paths:
    imgs/main_body.png                                     Duffing, square input (Fig. 3)
    imgs/<dir>/{zero,constant,sinusoid,square}_timeseries.png   Appendix E
Ground truth solid black; Autonomous blue, Curriculum orange, Static green, Dynamic red, dashed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path(__file__).resolve().parent / "tree" / "ICLR-figs" / "runs"
SYSTEMS = {  # system: (paper image dir, aspect ratio width / height)
    "iclr_duffing": ("Duffing/duffing_ready", 1.73),
    "iclr_vdp": ("Vand_Der_Pol/vdp", 1.73),
    "iclr_rossler": ("Rossler/rossler_ready", 1.15),
    "iclr_lorenz": ("Lorenz/lorenz", 1.15),
}
REGIMES = {"zero": "fam_zero", "constant": "fam_constant", "sinusoid": "fam_sinusoid", "square": "fam_square"}
METHODS = [("autonomous", "Autonomous", "tab:blue"), ("curriculum", "Curriculum", "tab:orange"),
           ("static", "Static HyperKKL", "tab:green"), ("dynamic", "Dynamic HyperKKL", "tab:red")]
WIDTH_IN = 8.0


def plot(system: str, tier: str, ep: int, aspect: float, path: Path):
    tr = np.load(RUNS / system / "seed0" / "traj.npz")
    res = json.loads((RUNS / system / "seed0" / "results.json").read_text())
    dt, sensors = res["scales"]["dt"], res["config"]["system"]["sensors"]
    X = tr[f"{tier}/clean/true"][ep]
    t = np.arange(len(X)) * dt
    nx = X.shape[1]
    fig, axes = plt.subplots(nx, 1, figsize=(WIDTH_IN, WIDTH_IN / aspect), sharex=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(t, X[:, j], color="black", lw=1.6, label="Ground truth")
        for key, lab, col in METHODS:
            ax.plot(t, tr[f"{tier}/clean/{key}"][ep][:, j], ls="--", lw=1.1, color=col, label=lab)
        ax.set_ylabel(f"$x_{j + 1}$" + (" (measured)" if j in sensors else " (unmeasured)"))
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    axes[-1].set_xlim(t[0], t[-1])
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=5, frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "figures"))
    ap.add_argument("--episode", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out) / "imgs"
    plot("iclr_duffing", "fam_square", a.episode, 1.73, out / "main_body.png")
    for system, (d, aspect) in SYSTEMS.items():
        for name, tier in REGIMES.items():
            plot(system, tier, a.episode, aspect, out / d / f"{name}_timeseries.png")
    print(f"17 figures in {out}")


if __name__ == "__main__":
    main()
