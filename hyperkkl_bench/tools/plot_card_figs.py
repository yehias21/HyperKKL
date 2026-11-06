"""Data figures for the dataset card (hyperkkl-bench 2.0.0rc1).

Reads only public splits of data_v2/ (train, test_op_id, test_op_ood, test_legacy_sequence_ood) through the package
loader; never the hidden labels. Writes PNGs to docs/figures/.
Usage: python tools/plot_card_figs.py [--root .]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# reference palette (dataviz skill, light mode): categorical slots 1-3, ink, surface; one sequential blue ramp
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#c9dcf5", "#2a78d6", "#0d2f5c"])

DRIVEN = ["battery", "bioreactor", "cartpole_b", "duffing", "fhn", "highway_v2", "lorenz", "lorenz96",
          "network", "quad_tank", "rossler", "two_link_arm_v2", "vdp"]
AUTONOMOUS = ["vdp", "fhn", "rossler", "lorenz", "network", "lorenz96", "lorenz96_n10"]

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "lines.linewidth": 1.2,
})

hkb = F = None


def spec(fam, plant):
    return f"{fam}/{plant}/{'realistic' if fam == 'driven' else 'partial'}"


def meta(root, fam, plant):
    with open(os.path.join(root, "data_v2", "metadata", fam, plant + ".json")) as f:
        return json.load(f)


def save(fig, root, name):
    out = os.path.join(root, "docs", "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name), dpi=160, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


def heading(fig, title, sub):
    fig.text(0.01, 1.0, title, ha="left", va="bottom", fontsize=12, color=INK, weight="bold")
    fig.text(0.01, 0.985, sub, ha="left", va="top", fontsize=8.5, color=INK2)


# ---------------------------------------------------------------- 1. the task
def task(root, plants=("cartpole_b", "battery", "quad_tank", "two_link_arm_v2"), ep=3):
    """What a method sees (input record + noisy measurements) and what it must reconstruct (hidden states)."""
    fig, axs = plt.subplots(3, len(plants), figsize=(12, 5.4), sharex="col",
                            gridspec_kw={"height_ratios": [0.8, 1.4, 1.4], "hspace": 0.12, "wspace": 0.22})
    for c, plant in enumerate(plants):
        m = meta(root, "driven", plant)
        es = hkb.load(spec("driven", plant), "test_op_id")
        t = np.arange(es.x.shape[1]) * float(m["dt"])
        k = len(t)
        axs[0, c].plot(t, es.u[ep, :k], color=INK2, lw=0.9)
        axs[0, c].set_title(plant, loc="left", weight="bold")
        s = m["sensors"][0]
        axs[1, c].scatter(t[::3], es.y[ep, ::3, 0], s=1.5, color=BLUE, alpha=0.55, lw=0)
        axs[1, c].plot(t, es.x[ep, :, s], color=BLUE, lw=0.6, alpha=0.9)
        for j, h in enumerate(m["unmeasured"][:2]):
            axs[2, c].plot(t, es.x[ep, :, h], color=ORANGE if j == 0 else AQUA, lw=1.3, label=f"x{h}")
        axs[2, c].axvspan(t[0], t[m["settle_index"]], color=GRID, alpha=0.45, lw=0)
        axs[2, c].legend(loc="upper right", fontsize=7, handlelength=1.2)
        axs[2, c].set_xlabel("time")
    axs[0, 0].set_ylabel("input u\n(given)", rotation=0, ha="right", va="center")
    axs[1, 0].set_ylabel("measured y\n(given, noisy)", rotation=0, ha="right", va="center")
    axs[2, 0].set_ylabel("hidden x\n(estimate it)", rotation=0, ha="right", va="center")
    axs[2, 0].text(0.02, 0.04, "transient (not scored)", transform=axs[2, 0].transAxes, color=MUTED, fontsize=7)
    heading(fig, "The task: observe (u, y), estimate the hidden states",
            "One test_op_id episode per plant. Dots: noisy sensor samples (coloured Student-t noise); thin line under "
            "them: the true measured state. Bottom: unmeasured states scored after the grey transient.")
    save(fig, root, "task.png")


# ------------------------------------------------------- 2. autonomous attractors
def colored_line(ax, x, y, lw=0.8):
    pts = np.stack([x, y], 1)[:, None, :]
    seg = np.concatenate([pts[:-1], pts[1:]], 1)
    lc = LineCollection(seg, cmap=SEQ, linewidths=lw)
    lc.set_array(np.linspace(0, 1, len(seg)))
    ax.add_collection(lc)
    ax.autoscale()


def attractors(root, ep=0):
    """Autonomous family: trajectories coloured by time (light = start, dark = end)."""
    fig = plt.figure(figsize=(12, 6.4))
    for k, plant in enumerate(AUTONOMOUS):
        m = meta(root, "autonomous", plant)
        es = hkb.load(spec("autonomous", plant), "train")
        x = es.x[ep]
        s, h = m["sensors"][0], m["unmeasured"][0]
        if plant in ("lorenz", "rossler"):
            ax = fig.add_subplot(2, 4, k + 1, projection="3d")
            ax.set_facecolor(SURF)
            n = len(x)
            for a in range(0, n - 1, 4):
                ax.plot(x[a:a + 5, 0], x[a:a + 5, 1], x[a:a + 5, 2], color=SEQ(a / n), lw=0.6)
            ax.set_xlabel("x0" + (" (measured)" if 0 in m["sensors"] else ""), fontsize=7, color=INK2)
            ax.set_ylabel("x1" + (" (measured)" if 1 in m["sensors"] else ""), fontsize=7, color=INK2)
            ax.set_zlabel("x2" + (" (measured)" if 2 in m["sensors"] else ""), fontsize=7, color=INK2)
            ax.tick_params(labelsize=6, colors=MUTED)
            for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                p.set_facecolor(SURF)
                p.set_edgecolor(GRID)
            ax.view_init(22, -55)
        else:
            ax = fig.add_subplot(2, 4, k + 1)
            colored_line(ax, x[:, s], x[:, h])
            ax.set_xlabel(f"x{s}  (measured)")
            ax.set_ylabel(f"x{h}  (hidden)")
        ax.set_title(f"{plant}   {m['n_y']} of {m['n_x']} states measured", loc="left", fontsize=8.5)
    cax = fig.add_axes([0.80, 0.14, 0.15, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["start", "end of episode"])
    cb.outline.set_visible(False)
    heading(fig, "Autonomous family (u = 0): the unforced dynamics to estimate",
            "One training episode per plant, coloured by time. 2-D panels plot a measured state against a hidden "
            "one: the observer sees only the horizontal axis.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, root, "attractors.png")


# ------------------------------------------------------------ 3. the conditions
def conditions(root, plant="cartpole_b", ep=5):
    """The robustness conditions applied to one episode."""
    sp = spec("driven", plant)
    m = meta(root, "driven", plant)
    es = hkb.load(sp, "test_op_id")
    t = np.arange(es.y.shape[1]) * float(m["dt"])
    clean = es.y[ep, :, 0]
    _, Yo, _, _ = F.apply_condition("outliers", es, sp)
    _, Yb, _, _ = F.apply_condition("burst_outage", es, sp)
    Uq, _, _, _ = F.apply_condition("input_quant", es, sp)
    Uc, _, _, _ = F.apply_condition("input_cmd_vs_applied", es, sp)
    Yo, Yb = np.asarray(Yo)[ep, :, 0], np.asarray(Yb)[ep, :, 0]
    fig, axs = plt.subplots(2, 2, figsize=(12, 5.2), gridspec_kw={"hspace": 0.45, "wspace": 0.18})
    a = axs[0, 0]
    a.plot(t, clean, color=BLUE, lw=0.8, label="clean y")
    bad = np.abs(Yo - clean) > 1e-6
    a.scatter(t[bad], Yo[bad], s=10, color=ORANGE, zorder=3, label="outlier samples (2 %, +-12 sd)")
    a.set_title("outliers", loc="left", weight="bold")
    a.legend(loc="upper right", fontsize=7)
    a = axs[0, 1]
    a.plot(t, clean, color=MUTED, lw=0.8, label="clean y")
    a.plot(t, Yb, color=BLUE, lw=1.1, label="what the observer gets")
    fr = np.abs(Yb - clean) > 1e-6
    if fr.any():
        i0, i1 = np.argmax(fr), len(fr) - np.argmax(fr[::-1])
        a.axvspan(t[i0], t[i1 - 1], color=ORANGE, alpha=0.15, lw=0)
        a.text(t[i0], a.get_ylim()[1], " sensor frozen", color=ORANGE, va="top", fontsize=7.5)
    a.set_title("burst_outage", loc="left", weight="bold")
    a.legend(loc="lower right", fontsize=7)
    u = es.u[ep]
    a = axs[1, 0]
    z = slice(len(t) // 3, len(t) // 3 + 120)
    a.step(t[z], np.asarray(Uq)[ep][z], where="post", color=ORANGE, lw=1.0, label="12-bit record")
    a.plot(t[z], u[z], color=BLUE, lw=0.9, label="applied u")
    a.set_title("input_quant (zoom)", loc="left", weight="bold")
    a.legend(loc="upper right", fontsize=7)
    a.set_xlabel("time")
    a = axs[1, 1]
    a.plot(t, np.asarray(Uc)[ep] - u, color=ORANGE, lw=0.9)
    a.axhline(0, color=MUTED, lw=0.6)
    a.set_title("input_cmd_vs_applied: command minus applied u", loc="left", weight="bold")
    a.text(0.01, 0.92, "the observer gets the command, the plant got the actuator output",
           transform=a.transAxes, fontsize=7, color=INK2)
    a.set_xlabel("time")
    heading(fig, f"Robustness conditions ({plant}, one test episode)",
            "Sensor conditions corrupt y; input-record conditions change only the observer's copy of u. The plant "
            "and the scored states are unchanged.")
    save(fig, root, "conditions.png")


# ------------------------------------------------------------- 4. input shift
def tier_shift(root, plants=("battery", "cartpole_b", "quad_tank", "two_link_arm_v2")):
    """Input distribution and spectrum per evaluation tier: how the operator changes."""
    tiers = [("test_op_id", "op_id", BLUE), ("test_op_ood", "op_ood (other operator)", ORANGE),
             ("test_legacy_sequence_ood", "legacy sequence_ood", AQUA)]
    fig, axs = plt.subplots(2, len(plants), figsize=(12, 5.0), gridspec_kw={"hspace": 0.45, "wspace": 0.25})
    for c, plant in enumerate(plants):
        m = meta(root, "driven", plant)
        dt = float(m["dt"])
        for split, lab, col in tiers:
            u = hkb.load(spec("driven", plant), split).u.astype(float)
            lo, hi = np.percentile(u, [0.5, 99.5])
            axs[0, c].hist(u.ravel(), bins=60, range=(lo, hi), density=True, histtype="step", color=col, lw=1.3,
                           label=lab)
            f = np.fft.rfftfreq(u.shape[1], dt)[1:]
            p = (np.abs(np.fft.rfft(u - u.mean(1, keepdims=True), axis=1)) ** 2).mean(0)[1:]
            axs[1, c].loglog(f, p / p.sum(), color=col, lw=1.0)
        axs[0, c].set_title(plant, loc="left", weight="bold")
        axs[0, c].set_yticks([])
        axs[0, c].set_xlabel("u value")
        axs[1, c].set_xlabel("frequency")
        if c == 0:
            axs[0, c].set_ylabel("share of time")
            axs[1, c].set_ylabel("power (normalised)")
    axs[0, 0].legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, -1.75), ncol=3)
    heading(fig, "What changes between test tiers: the operator's input",
            "Top: distribution of input values over all 64 episodes of each tier. Bottom: mean input power spectrum. "
            "Same plant, same actuator box; only the way it is driven changes.")
    save(fig, root, "tier_shift.png")


# ---------------------------------------------------------------- 5. overview
def overview(root):
    """Every plant as a row of state cells: filled = measured, open = hidden (to estimate)."""
    rows = [("autonomous", p) for p in AUTONOMOUS] + [("driven", p) for p in DRIVEN]
    nmax = max(meta(root, f, p)["n_x"] for f, p in rows)
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 1.6))
    for r, (fam, p) in enumerate(rows):
        m = meta(root, fam, p)
        yv = len(rows) - 1 - r + (0.6 if fam == "autonomous" else 0)
        for k in range(m["n_x"]):
            meas = k in m["sensors"]
            ax.scatter(k, yv, s=70, marker="s", color=BLUE if meas else SURF, edgecolor=BLUE if meas else ORANGE,
                       lw=1.4, zorder=3)
        ax.text(-1.0, yv, p, ha="right", va="center", fontsize=8, color=INK)
        ax.text(nmax + 0.6, yv, f"{m['n_y']}/{m['n_x']} measured   T = {m['T']:.0f}   "
                f"{'u = 0' if fam == 'autonomous' else str(m['n_u']) + ' input'}", va="center", fontsize=7.5,
                color=INK2)
    na = len(DRIVEN)
    ax.text(-8.5, na + 0.6 + len(AUTONOMOUS) / 2 - 0.5, "autonomous", rotation=90, va="center", ha="center",
            color=INK2, weight="bold")
    ax.text(-8.5, na / 2 - 0.5, "driven", rotation=90, va="center", ha="center", color=INK2, weight="bold")
    ax.axhline(na - 0.2, color=GRID, lw=1)
    ax.scatter([], [], s=70, marker="s", color=BLUE, label="measured state (sensor)")
    ax.scatter([], [], s=70, marker="s", color=SURF, edgecolor=ORANGE, lw=1.4, label="hidden state (estimate it)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=8)
    ax.set_xlim(-9.5, nmax + 9)
    ax.set_ylim(-0.8, len(rows) + 0.4)
    ax.axis("off")
    heading(fig, "The benchmark at a glance: 20 cells, what is measured and what is hidden",
            "Each row is a plant; each square one state. Lorenz-96 has 20 states with every second one measured.")
    save(fig, root, "overview.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.root, "src"))
    import hyperkkl_bench as hkb  # noqa: E402
    import hyperkkl_bench.families as F  # noqa: E402
    globals().update(hkb=hkb, F=F)
    overview(a.root)
    task(a.root)
    attractors(a.root)
    conditions(a.root)
    tier_shift(a.root)
