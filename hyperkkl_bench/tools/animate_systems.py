"""Animated system gallery for the dataset card: each plant's model is drawn and moved by one public test episode
of the benchmark (driven/<plant>/realistic, test_op_id; the true states of that episode), next to its signals.
Blue = measured, orange = to be estimated, grey = input.

Writes docs/figures/systems/<plant>.gif.   python tools/animate_systems.py [--root .] [--only cartpole_b,...]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patches
from matplotlib.collections import LineCollection

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
FRAMES, FPS, DPI = 240, 20, 100

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False,
})


def episode(root, plant, fam="driven", ep=2):
    import hyperkkl_bench as hkb
    level = "realistic" if fam == "driven" else "partial"
    es = hkb.load(f"{fam}/{plant}/{level}", "test_op_id" if fam == "driven" else "test_id")
    with open(os.path.join(root, "data_v2", "metadata", fam, plant + ".json")) as f:
        m = json.load(f)
    x = np.asarray(es.x[ep], float)
    u = np.asarray(es.u[ep], float)[: len(x)] if fam == "driven" else np.zeros(len(x))
    y = np.asarray(es.y[ep], float)
    t = np.arange(len(x)) * float(m["dt"])
    return t, u, y, x, m


class Gallery:
    """Left: the schematic (draw(k) updates it). Right: input, one measured and one estimated signal with a cursor."""

    def __init__(self, title, t, u, y, x, m, meas_label, hid_label, hid_idx=None, schematic_aspect="equal",
                 real=None):
        """real: optional (real-life caption, model caption); adds a real-life panel (self.real) before the model."""
        self.t, self.u, self.y, self.x, self.m = t, u, y, x, m
        self.idx = np.linspace(0, len(t) - 1, FRAMES).astype(int)
        cols = 3 if real else 2
        self.fig = plt.figure(figsize=(13.0 if real else 9.6, 4.0))
        gs = self.fig.add_gridspec(3, cols, width_ratios=[1.0, 1.0, 1.0] if real else [1.35, 1.0], hspace=0.35,
                                   wspace=0.32, left=0.02, right=0.98, top=0.84, bottom=0.11)
        if real:
            self.real = self.fig.add_subplot(gs[:, 0])
            self.real.set_aspect("equal")
            self.real.axis("off")
            self.real.set_title("1. in real life: " + real[0], loc="left", fontsize=9, color=INK2)
        self.ax = self.fig.add_subplot(gs[:, cols - 2])
        self.ax.set_aspect(schematic_aspect)
        self.ax.axis("off")
        if real:
            self.ax.set_title("2. the simplified model: " + real[1], loc="left", fontsize=9, color=INK2)
        self.fig.text(0.01, 0.98, title, ha="left", va="top", fontsize=10.5, weight="bold", color=INK)
        h = hid_idx if hid_idx is not None else m["unmeasured"][0]
        rows = [(u, "input u", INK2), (y[:, 0], meas_label, BLUE), (x[:, h], hid_label, ORANGE)]
        self.cur, self.live = [], []
        for r, (sig, lab, col) in enumerate(rows):
            a = self.fig.add_subplot(gs[r, cols - 1])
            if r == 0 and real:
                a.set_title("3. its signals (benchmark episode)", loc="left", fontsize=9, color=INK2)
            a.plot(t, sig, color=col, lw=0.8, alpha=0.12)                      # faint full trace: where it goes
            if r == 1:                                                          # noisy samples appear as they arrive
                ln, = a.plot([], [], ".", color=col, ms=1.6, alpha=0.7)
            else:
                ln, = a.plot([], [], color=col, lw=1.4)
            self.live.append((ln, sig, 4 if r == 1 else 1))
            a.set_xlim(t[0], t[-1])
            pad = 0.08 * (np.nanmax(sig) - np.nanmin(sig) + 1e-9)
            a.set_ylim(np.nanmin(sig) - pad, np.nanmax(sig) + pad)
            a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7.5, color=col if r else INK2)
            a.tick_params(labelsize=6.5)
            if r < 2:
                a.tick_params(labelbottom=False)
            self.cur.append(a.plot([], [], "o", color=col, ms=4)[0])
        a.set_xlabel("time", fontsize=7)

    def run(self, draw, path):
        def step(i):
            k = self.idx[i]
            for (ln, sig, st), c in zip(self.live, self.cur):
                ln.set_data(self.t[:k + 1:st], sig[:k + 1:st])
                c.set_data([self.t[k]], [sig[k]])
            return draw(k)
        an = animation.FuncAnimation(self.fig, step, frames=FRAMES, interval=1000 / FPS, blit=False)
        an.save(path, writer=animation.PillowWriter(fps=FPS), dpi=DPI, savefig_kwargs={"facecolor": SURF})
        plt.close(self.fig)
        print("wrote", path, f"{os.path.getsize(path) / 1e6:.2f} MB")


def tag(ax, xy, text, col, **kw):
    kw.setdefault("fontsize", 7.5)
    return ax.text(*xy, text, color=col, ha="center", va="center", **kw)


# ------------------------------------------------------------------ plants
def cartpole_b(root, out):
    t, u, y, x, m = episode(root, "cartpole_b")
    g = Gallery("Cart-pole: how a walking robot keeps its balance", t, u, y, x, m,
                "cart position\n(measured)", "cart velocity\n(estimated)", hid_idx=1,
                real=("a humanoid balancing over its feet", "cart-pole"))
    p_lo, p_hi = x[:, 0].min(), x[:, 0].max()
    vmax = np.percentile(np.abs(x[:, 1]), 99) + 1e-9

    # 1. real life: humanoid. Feet follow the cart, body lean = pole angle (same episode); the centre of mass (CoM)
    # and its velocity: the CoM velocity is what a humanoid has no sensor for and must estimate.
    rl = g.real
    rl.set_xlim(p_lo - 1.1, p_hi + 1.1); rl.set_ylim(-0.5, 2.55)
    rl.add_patch(patches.Rectangle((p_lo - 1.1, -0.5), p_hi - p_lo + 2.2, 0.3, color="#efeee9", lw=0))
    for xx in np.arange(np.floor(p_lo - 1), p_hi + 1.1, 0.5):
        rl.plot([xx, xx - 0.12], [-0.2, -0.32], color=GRID, lw=1)
    foot_l, = rl.plot([], [], color=INK2, lw=5, solid_capstyle="round")
    foot_r, = rl.plot([], [], color=INK2, lw=5, solid_capstyle="round")
    shin_l, = rl.plot([], [], color=MUTED, lw=6, solid_capstyle="round")
    shin_r, = rl.plot([], [], color=MUTED, lw=6, solid_capstyle="round")
    thigh_l, = rl.plot([], [], color=MUTED, lw=7, solid_capstyle="round")
    thigh_r, = rl.plot([], [], color=MUTED, lw=7, solid_capstyle="round")
    torso = rl.add_patch(patches.FancyBboxPatch((0, 0), 0.44, 0.7, boxstyle="round,pad=0.04", color=BLUE))
    head = rl.add_patch(patches.FancyBboxPatch((0, 0), 0.3, 0.26, boxstyle="round,pad=0.04", color=BLUE))
    visor, = rl.plot([], [], color=SURF, lw=3, solid_capstyle="round")
    arm, = rl.plot([], [], color=MUTED, lw=5, solid_capstyle="round", zorder=4)
    com = rl.add_patch(patches.Circle((0, 0), 0.07, fc=SURF, ec=ORANGE, lw=2, zorder=5))
    com_arrow = rl.annotate("", xy=(0, 0), xytext=(0, 0), zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2, mutation_scale=12))
    tag(rl, ((p_lo + p_hi) / 2, -0.38), "CoM velocity (orange): no sensor measures it, it must be estimated",
        ORANGE, fontsize=7)
    R = lambda th: np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])

    def draw_real(k):
        p, th = x[k, 0], x[k, 2]
        ank_l, ank_r = np.array([p - 0.13, -0.15]), np.array([p + 0.13, -0.15])
        foot_l.set_data([ank_l[0] - 0.08, ank_l[0] + 0.14], [-0.17, -0.17])
        foot_r.set_data([ank_r[0] - 0.08, ank_r[0] + 0.14], [-0.17, -0.17])
        hip = np.array([p, -0.15]) + R(th) @ np.array([0, 0.95])
        knee_l = (ank_l + hip) / 2 + R(th) @ np.array([0.06, 0]); knee_r = (ank_r + hip) / 2 + R(th) @ np.array([0.06, 0])
        shin_l.set_data([ank_l[0], knee_l[0]], [ank_l[1], knee_l[1]]); shin_r.set_data([ank_r[0], knee_r[0]], [ank_r[1], knee_r[1]])
        thigh_l.set_data([knee_l[0], hip[0]], [knee_l[1], hip[1]]); thigh_r.set_data([knee_r[0], hip[0]], [knee_r[1], hip[1]])
        tr = matplotlib.transforms.Affine2D().rotate_around(hip[0], hip[1], -th) + rl.transData
        torso.set_x(hip[0] - 0.22); torso.set_y(hip[1]); torso.set_transform(tr)
        head.set_x(hip[0] - 0.15); head.set_y(hip[1] + 0.82); head.set_transform(tr)
        visor.set_data([hip[0] - 0.05, hip[0] + 0.12], [hip[1] + 0.95, hip[1] + 0.95]); visor.set_transform(tr)
        sh = hip + R(th) @ np.array([0.26, 0.6])
        hand = sh + R(th) @ np.array([0.08, -0.5])
        arm.set_data([sh[0], hand[0]], [sh[1], hand[1]])
        c = hip + R(th) @ np.array([0, 0.3])
        com.center = tuple(c)
        com_arrow.xy, com_arrow.xyann = (c[0] + 0.7 * x[k, 1] / vmax, c[1]), tuple(c)

    # 2. the model: cart-pole with wheels, hinge and an angle arc
    ax = g.ax
    lo, hi = p_lo - 0.9, p_hi + 0.9
    ax.set_xlim(lo, hi); ax.set_ylim(-0.6, 1.55)
    ax.plot([lo, hi], [-0.1, -0.1], color=MUTED, lw=2)
    for xx in np.arange(np.ceil(lo), hi, 0.5):
        ax.plot([xx, xx], [-0.1, -0.16], color=MUTED, lw=1)
    cart = ax.add_patch(patches.FancyBboxPatch((0, 0), 0.56, 0.2, boxstyle="round,pad=0.03", color=BLUE))
    w1 = ax.add_patch(patches.Circle((0, 0), 0.055, color=INK2)); w2 = ax.add_patch(patches.Circle((0, 0), 0.055, color=INK2))
    pole, = ax.plot([], [], color=BLUE, lw=3.5, solid_capstyle="round")
    bob = ax.add_patch(patches.Circle((0, 0), 0.08, color=BLUE))
    hinge = ax.add_patch(patches.Circle((0, 0), 0.03, color=SURF, zorder=4))
    vert, = ax.plot([], [], color=GRID, lw=1, ls="--")
    arc = ax.add_patch(patches.Arc((0, 0), 0.6, 0.6, theta1=90, theta2=90, color=BLUE, lw=1.2))
    th_txt = tag(ax, (0, 0), "", BLUE, fontsize=7)
    arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2, mutation_scale=12))
    tag(ax, ((lo + hi) / 2, -0.42), "measured: cart position, pole angle (blue)\nestimated: velocities (orange)", INK2,
        fontsize=7)

    def draw(k):
        p, v, th = x[k, 0], x[k, 1], x[k, 2]
        cart.set_x(p - 0.28); cart.set_y(-0.04)
        w1.center, w2.center = (p - 0.17, -0.045), (p + 0.17, -0.045)
        base = (p, 0.2)
        tip = (p + 1.05 * np.sin(th), 0.2 + 1.05 * np.cos(th))
        pole.set_data([base[0], tip[0]], [base[1], tip[1]]); bob.center = tip; hinge.center = base
        vert.set_data([p, p], [0.2, 1.3])
        arc.center = base
        a0, a1 = sorted([90.0, 90.0 - np.degrees(th)])
        arc.theta1, arc.theta2 = a0, a1
        th_txt.set_position((p + 0.42 * np.sign(th + 1e-9), 0.62)); th_txt.set_text(f"θ {np.degrees(th):+.1f}°")
        arrow.xy, arrow.xyann = (p + 0.75 * v / vmax, 0.08), (p, 0.08)
        draw_real(k)
        return []
    g.run(draw, out)


def two_link_arm_v2(root, out):
    t, u, y, x, m = episode(root, "two_link_arm_v2")
    g = Gallery("Two-link arm: a robot manipulator's shoulder and elbow", t, u, y, x, m,
                "shoulder angle\n(measured)", "shoulder rate\n(estimated)", hid_idx=2)
    ax = g.ax
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.add_patch(patches.Rectangle((-0.25, -0.05), 0.5, 0.1, color=MUTED))
    l1, = ax.plot([], [], color=BLUE, lw=5, solid_capstyle="round")
    l2, = ax.plot([], [], color=AQUA, lw=4, solid_capstyle="round")
    trail, = ax.plot([], [], color=ORANGE, lw=0.8, alpha=0.6)
    tag(ax, (0, -2.0), "joint angles measured; joint speeds estimated. Torque on the shoulder", INK2)
    hist = []

    def draw(k):
        q1, q2 = x[k, 0], x[k, 1]
        e = (np.cos(q1), np.sin(q1))
        h = (e[0] + np.cos(q1 + q2), e[1] + np.sin(q1 + q2))
        l1.set_data([0, e[0]], [0, e[1]]); l2.set_data([e[0], h[0]], [e[1], h[1]])
        hist.append(h); trail.set_data(*np.array(hist[-25:]).T)
        return []
    g.run(draw, out)


def quad_tank(root, out):
    t, u, y, x, m = episode(root, "quad_tank")
    g = Gallery("Quadruple tank: level control in a chemical or water-treatment plant", t, u, y, x, m,
                "lower level h1\n(measured)", "upper level h3\n(estimated)", hid_idx=2)
    ax = g.ax
    ax.set_xlim(0, 4); ax.set_ylim(0, 4.4)
    hmax = np.percentile(x, 99.5, axis=0) * 1.1
    boxes = {0: (0.4, 0.3), 1: (2.4, 0.3), 2: (0.4, 2.3), 3: (2.4, 2.3)}
    fills = {}
    for i, (bx, by) in boxes.items():
        meas = i in m["sensors"]
        ax.add_patch(patches.Rectangle((bx, by), 1.2, 1.6, fill=False, ec=BLUE if meas else ORANGE, lw=1.6,
                                       ls="-" if meas else "--"))
        fills[i] = ax.add_patch(patches.Rectangle((bx, by), 1.2, 0, color=BLUE if meas else ORANGE, alpha=0.55))
        tag(ax, (bx + 0.6, by + 1.75), f"h{i + 1} " + ("measured" if meas else "estimated"), BLUE if meas else ORANGE)
    pump = tag(ax, (2.0, 4.25), "", INK2)

    def draw(k):
        for i, (bx, by) in boxes.items():
            fills[i].set_height(1.6 * np.clip(x[k, i] / hmax[i], 0, 1))
        pump.set_text(f"pump command u = {u[k]:+.2f}")
        return []
    g.run(draw, out)


def battery(root, out):
    t, u, y, x, m = episode(root, "battery")
    g = Gallery("Li-ion cell: the charge gauge of an electric car", t, u, y, x, m,
                "temperature rise\n(measured)", "state of charge\n(estimated)", hid_idx=0,
                real=("an EV and its charge gauge", "equivalent circuit"))
    umax = np.percentile(np.abs(u), 99) + 1e-9
    soc_lo, soc_hi = x[:, 0].min(), x[:, 0].max()

    # 1. real life: an EV; wheel spin follows the current demand u (discharge = driving, charge = regenerating);
    # the dashboard gauge shows the state of charge, which no sensor measures: the BMS estimates it
    rl = g.real
    rl.set_xlim(0, 4); rl.set_ylim(-0.2, 3.2)
    road, = rl.plot([], [], color=MUTED, lw=1.5, ls=(0, (6, 6)))
    rl.plot([0, 4], [0.05, 0.05], color=MUTED, lw=2)
    body = [(0.5, 0.3), (3.5, 0.3), (3.55, 0.75), (2.75, 0.85), (2.25, 1.25), (1.2, 1.25), (0.8, 0.85), (0.45, 0.8)]
    rl.add_patch(patches.Polygon(body, closed=True, color=BLUE, alpha=0.85))
    rl.add_patch(patches.Polygon([(1.3, 0.9), (2.2, 0.9), (2.2, 1.17), (1.35, 1.17)], color=SURF, alpha=0.8))
    rl.add_patch(patches.Rectangle((1.05, 0.38), 1.9, 0.18, color=ORANGE, alpha=0.5))
    rl.text(2.0, 0.47, "battery pack", ha="center", va="center", fontsize=6.5, color=INK)
    spokes = []
    for cx in (1.05, 2.95):
        rl.add_patch(patches.Circle((cx, 0.3), 0.24, color=INK2))
        rl.add_patch(patches.Circle((cx, 0.3), 0.12, color=SURF))
        spokes.append((cx, rl.plot([], [], color=INK2, lw=2)[0]))
    # dashboard gauge
    gc = (2.0, 2.2)
    rl.add_patch(patches.Wedge(gc, 0.8, 0, 180, width=0.16, color=GRID))
    gauge = rl.add_patch(patches.Wedge(gc, 0.8, 180, 180, width=0.16, color=ORANGE))
    needle, = rl.plot([], [], color=INK, lw=2)
    soc_txt = rl.text(gc[0], gc[1] + 0.22, "", ha="center", fontsize=9, weight="bold", color=ORANGE)
    rl.text(gc[0], gc[1] - 0.18, "state of charge: estimated by the BMS", ha="center", fontsize=6.5, color=ORANGE)
    mode = rl.text(2.0, 1.55, "", ha="center", fontsize=7, color=INK2)
    ang = [0.0]

    def draw_real(k):
        s = np.clip((x[k, 0] - soc_lo) / (soc_hi - soc_lo + 1e-9), 0, 1)
        a = 180 * (1 - s)
        gauge.set_theta1(a)
        needle.set_data([gc[0], gc[0] + 0.62 * np.cos(np.radians(a))], [gc[1], gc[1] + 0.62 * np.sin(np.radians(a))])
        soc_txt.set_text(f"{100 * x[k, 0]:.1f} %")
        ang[0] -= 0.9 * u[k] / umax
        for cx, sp in spokes:
            sp.set_data([cx - 0.24 * np.cos(ang[0]), cx + 0.24 * np.cos(ang[0])],
                        [0.3 - 0.24 * np.sin(ang[0]), 0.3 + 0.24 * np.sin(ang[0])])
        off = (-ang[0] * 0.15) % 0.4
        road.set_data(np.arange(-off, 4, 0.4), np.full(len(np.arange(-off, 4, 0.4)), -0.1))
        mode.set_text("driving: discharging" if u[k] > 0.02 * umax else ("braking: regenerating" if u[k] < -0.02 * umax
                                                                          else "idle"))

    # 2. the model: OCV(SOC) source, R0, two RC pairs; current flows with u
    ax = g.ax
    ax.set_xlim(0, 4.4); ax.set_ylim(-0.4, 3.0)
    W = dict(color=INK2, lw=1.6)
    ax.plot([0.5, 0.5], [0.6, 2.4], **W); ax.plot([0.5, 3.9], [2.4, 2.4], **W); ax.plot([0.5, 3.9], [0.6, 0.6], **W)
    src = ax.add_patch(patches.Circle((0.5, 1.5), 0.28, fc=SURF, ec=ORANGE, lw=2, zorder=3))
    ax.text(0.5, 1.5, "OCV\n(SOC)", ha="center", va="center", fontsize=6.5, color=ORANGE, zorder=4)
    def resistor(x0, y0, w=0.5, lab=""):
        zz = np.linspace(0, 1, 9)
        ax.plot(x0 + w * zz, y0 + 0.08 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]), color=INK2, lw=1.6, zorder=3)
        ax.add_patch(patches.Rectangle((x0, y0 - 0.1), w, 0.2, color=SURF, zorder=2))
        ax.text(x0 + w / 2, y0 + 0.2, lab, ha="center", fontsize=6.5, color=INK2)
    resistor(0.9, 2.4, lab="R0")
    for i, (x0, lab) in enumerate(((1.7, "R1 ‖ C1"), (2.6, "R2 ‖ C2"))):
        ax.add_patch(patches.Rectangle((x0, 2.12), 0.6, 0.56, fill=False, ec=INK2, lw=1.2))
        resistor(x0 + 0.05, 2.68, w=0.5)
        ax.plot([x0 + 0.25, x0 + 0.25], [2.04, 2.2], color=INK2, lw=2.5); ax.plot([x0 + 0.35, x0 + 0.35], [2.04, 2.2], color=INK2, lw=2.5)
        ax.text(x0 + 0.3, 1.88, lab, ha="center", fontsize=6.5, color=INK2)
    ax.text(4.1, 1.5, "V", ha="center", va="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(4.1, 1.1, "voltage\n(measured)", ha="center", fontsize=6.5, color=BLUE)
    therm = ax.add_patch(patches.Rectangle((1.4, 0.05), 1.6, 0.25, color=BLUE, alpha=0.2))
    ax.text(2.2, -0.2, "cell temperature (measured)", ha="center", fontsize=6.5, color=BLUE)
    dots = ax.scatter([], [], s=14, color=ORANGE, zorder=5)
    loop = np.concatenate([np.stack([np.linspace(0.5, 3.9, 30), np.full(30, 2.4)], 1),
                           np.stack([np.full(12, 3.9), np.linspace(2.4, 0.6, 12)], 1),
                           np.stack([np.linspace(3.9, 0.5, 30), np.full(30, 0.6)], 1),
                           np.stack([np.full(12, 0.5), np.linspace(0.6, 2.4, 12)], 1)])
    cur_txt = ax.text(2.2, 0.85, "", ha="center", fontsize=7, color=INK2)
    phase = [0.0]
    dTmax = np.percentile(x[:, 3], 99) + 1e-9

    def draw(k):
        phase[0] = (phase[0] + 2.5 * u[k] / umax) % len(loop)
        idx = (np.arange(0, len(loop), 6) + int(phase[0])) % len(loop)
        dots.set_offsets(loop[idx] if abs(u[k]) > 0.02 * umax else np.empty((0, 2)))
        cur_txt.set_text(f"current u = {u[k]:+.2f}")
        therm.set_alpha(0.1 + 0.6 * np.clip(x[k, 3] / dTmax, 0, 1))
        draw_real(k)
        return []
    g.run(draw, out)


def bioreactor(root, out):
    t, u, y, x, m = episode(root, "bioreactor")
    g = Gallery("Bioreactor: fermentation for vaccines, insulin or beer", t, u, y, x, m,
                "substrate\n(measured)", "biomass\n(estimated)", hid_idx=0)
    ax = g.ax
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.add_patch(patches.FancyBboxPatch((0.9, 0.4), 2.2, 2.9, boxstyle="round,pad=0.1", fill=False, ec=INK2, lw=2))
    rng = np.random.RandomState(0)
    cells = ax.scatter(rng.uniform(1.0, 3.0, 160), rng.uniform(0.5, 3.1, 160), s=9, color=ORANGE, alpha=0.7)
    sub = ax.scatter(rng.uniform(1.0, 3.0, 160), rng.uniform(0.5, 3.1, 160), s=6, color=BLUE, marker="s", alpha=0.6)
    feed = tag(ax, (2.0, 3.75), "", INK2)
    tag(ax, (2.0, 0.1), "orange: cells (biomass, estimated)   blue: sugar (substrate, measured)", INK2)
    b0, s0 = np.percentile(x[:, 0], 99), np.percentile(x[:, 1], 99)

    def draw(k):
        nb = int(160 * np.clip(x[k, 0] / b0, 0.02, 1)); ns = int(160 * np.clip(x[k, 1] / s0, 0.02, 1))
        cells.set_alpha(None); cells.set_sizes([9] * nb + [0] * (160 - nb))
        sub.set_sizes([6] * ns + [0] * (160 - ns))
        feed.set_text(f"feed / dilution change u = {u[k]:+.3f}")
        return []
    g.run(draw, out)


def highway_v2(root, out):
    t, u, y, x, m = episode(root, "highway_v2")
    g = Gallery("Freeway with an on-ramp: traffic density for ramp metering", t, u, y, x, m,
                "density cell 1\n(measured)", "density cell 2\n(estimated)", hid_idx=1)
    ax = g.ax
    ax.set_xlim(0, 5.2); ax.set_ylim(-1.2, 2.2)
    ax.add_patch(patches.Rectangle((0.1, 0.3), 5.0, 0.8, color="#f1f0ec"))
    ax.plot([2.3, 2.8], [-0.6, 0.3], color=MUTED, lw=6, alpha=0.5)
    rng = np.random.RandomState(1)
    cars = []
    dmax = np.percentile(x, 99.5)
    for c in range(5):
        meas = c in m["sensors"]
        col = BLUE if meas else ORANGE
        ax.add_patch(patches.Rectangle((0.1 + c, 0.3), 1.0, 0.8, fill=False, ec=col, lw=1.2, ls="-" if meas else "--"))
        tag(ax, (0.6 + c, 1.4), f"cell {c + 1}\n" + ("detector" if meas else "estimated"), col)
        cars.append(ax.scatter(0.15 + c + 0.9 * rng.rand(40), 0.35 + 0.7 * rng.rand(40), s=10, marker="s", color=col))
    ramp = tag(ax, (2.2, -0.9), "", INK2)

    def draw(k):
        for c in range(5):
            n = int(40 * np.clip(x[k, c] / dmax, 0, 1))
            cars[c].set_sizes([10] * n + [0] * (40 - n))
        ramp.set_text(f"on-ramp inflow change u = {u[k]:+.2f}")
        return []
    g.run(draw, out)


def fhn(root, out):
    t, u, y, x, m = episode(root, "fhn")
    g = Gallery("FitzHugh-Nagumo neuron: a nerve or heart cell firing", t, u, y, x, m,
                "membrane voltage\n(measured)", "recovery\n(estimated)", hid_idx=1)
    ax = g.ax
    ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4)
    body = ax.add_patch(patches.Circle((0, 0.3), 0.7, color=BLUE, alpha=0.3))
    for a in np.linspace(0, 2 * np.pi, 7)[:-1]:
        ax.plot([0.7 * np.cos(a), 1.6 * np.cos(a)], [0.3 + 0.7 * np.sin(a), 0.3 + 1.6 * np.sin(a)], color=MUTED, lw=1.2)
    ax.plot([0, 0], [-0.4, -2.1], color=MUTED, lw=2.5)
    rec = ax.add_patch(patches.Rectangle((1.7, -2.1), 0.35, 0, color=ORANGE, alpha=0.7))
    tag(ax, (1.87, -2.3), "recovery\n(estimated)", ORANGE)
    v0, v1 = x[:, 0].min(), x[:, 0].max(); w0, w1 = x[:, 1].min(), x[:, 1].max()

    def draw(k):
        a = (x[k, 0] - v0) / (v1 - v0)
        body.set_alpha(0.15 + 0.8 * a)
        rec.set_height(2.0 * (x[k, 1] - w0) / (w1 - w0))
        return []
    g.run(draw, out)


def vdp(root, out):
    t, u, y, x, m = episode(root, "vdp")
    g = Gallery("Van der Pol oscillator: valve circuits, heartbeat pacemaker models", t, u, y, x, m,
                "position\n(measured)", "velocity\n(estimated)", hid_idx=1, schematic_aspect="auto")
    ax = g.ax
    ax.axis("on"); ax.set_xlabel("position (measured)"); ax.set_ylabel("velocity (estimated)")
    ax.plot(x[:, 0], x[:, 1], color=GRID, lw=0.8)
    trail, = ax.plot([], [], color=ORANGE, lw=1.6)
    dot, = ax.plot([], [], "o", color=ORANGE, ms=6)

    def draw(k):
        a = max(0, k - 60)
        trail.set_data(x[a:k + 1, 0], x[a:k + 1, 1]); dot.set_data([x[k, 0]], [x[k, 1]])
        return []
    g.run(draw, out)


def chaotic(name, title, sub, root, out):
    t, u, y, x, m = episode(root, name)
    s, h = m["sensors"][0], m["unmeasured"][0]
    g = Gallery(title, t, u, y, x, m, f"x{s + 1}\n(measured)", f"x{h + 1}\n(estimated)", hid_idx=h,
                schematic_aspect="auto")
    ax = g.ax
    ax.axis("on"); ax.set_xlabel(f"x{h + 1} (estimated)"); ax.set_ylabel("x3" if name == "lorenz" else f"x{s + 1}")
    j = 2 if name == "lorenz" else s
    ax.plot(x[:, h], x[:, j], color=GRID, lw=0.5)
    trail, = ax.plot([], [], color=ORANGE, lw=1.3)
    dot, = ax.plot([], [], "o", color=ORANGE, ms=5)
    ax.text(0.02, 0.02, sub, transform=ax.transAxes, fontsize=7, color=INK2)

    def draw(k):
        a = max(0, k - 140)
        trail.set_data(x[a:k + 1, h], x[a:k + 1, j]); dot.set_data([x[k, h]], [x[k, j]])
        return []
    g.run(draw, out)


def ring(name, title, root, out, n_nodes, value, measured_nodes, sub):
    t, u, y, x, m = episode(root, name)
    g = Gallery(title, t, u, y, x, m, "node 1\n(measured)", "an estimated\nnode", hid_idx=m["unmeasured"][0])
    ax = g.ax
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.7, 1.5)
    ang = np.pi / 2 - 2 * np.pi * np.arange(n_nodes) / n_nodes
    P = np.stack([np.cos(ang), np.sin(ang)], 1)
    ax.add_collection(LineCollection([[P[i], P[(i + 1) % n_nodes]] for i in range(n_nodes)], colors=GRID, lw=1.5))
    V = value(x)
    lo, hi = np.percentile(V, 1), np.percentile(V, 99)
    sizes = lambda v: 40 + 360 * np.clip((v - lo) / (hi - lo), 0, 1)
    cols = [BLUE if i in measured_nodes else ORANGE for i in range(n_nodes)]
    sc = ax.scatter(P[:, 0], P[:, 1], s=sizes(V[0]), color=cols, alpha=0.8, edgecolor=SURF, lw=1.2, zorder=3)
    ax.text(0, -1.6, sub, ha="center", fontsize=7, color=INK2)

    def draw(k):
        sc.set_sizes(sizes(V[k]))
        return []
    g.run(draw, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.root, "src"))
    out = os.path.join(a.root, "docs", "figures", "systems")
    os.makedirs(out, exist_ok=True)
    jobs = {
        "cartpole_b": lambda p: cartpole_b(a.root, p),
        "two_link_arm_v2": lambda p: two_link_arm_v2(a.root, p),
        "quad_tank": lambda p: quad_tank(a.root, p),
        "battery": lambda p: battery(a.root, p),
        "bioreactor": lambda p: bioreactor(a.root, p),
        "highway_v2": lambda p: highway_v2(a.root, p),
        "fhn": lambda p: fhn(a.root, p),
        "vdp": lambda p: vdp(a.root, p),
        "lorenz": lambda p: chaotic("lorenz", "Lorenz system: convection rolls in the atmosphere", "the butterfly attractor", a.root, p),
        "rossler": lambda p: chaotic("rossler", "Rossler system: an oscillating chemical reaction", "a folded spiral", a.root, p),
        "network": lambda p: ring("network", "Network of 5 coupled neurons: a small neural circuit", a.root, p, 5,
                                  lambda x: x[:, 0::2], {0, 1, 3}, "node size = voltage; blue nodes measured, orange estimated; u drives node 1"),
        "lorenz96": lambda p: ring("lorenz96", "Lorenz-96: 20 weather sites around a latitude circle", a.root, p, 20,
                                   lambda x: x, set(range(0, 20, 2)), "marker size = local weather variable; every second site observed"),
    }
    for k, f in jobs.items():
        if a.only and k not in a.only.split(","):
            continue
        f(os.path.join(out, k + ".gif"))


if __name__ == "__main__":
    main()
