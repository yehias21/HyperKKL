"""Polished system animations for the dataset card (cairo vector rendering -> MP4 + GIF via ffmpeg).

Each animation is one public benchmark episode (driven/<plant>/realistic, test_op_id): the real-life system, the
simplified model, and its signals, all moved by the same simulated states. Nothing is mocked: every position,
angle and trace comes from the episode.

    python tools/render_systems.py [--root .] [--only cartpole_b]
Writes docs/figures/systems/<plant>.mp4 and .gif.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import cairo
import numpy as np

W, H = 1820, 540
FPS, SECONDS = 30, 12
FONT = "DejaVu Sans"


def hexc(h, a=1.0):
    h = C.get(h, h).lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (a,)


C = dict(bg="#f6f5f1", card="#ffffff", ink="#1d1d1f", ink2="#5b5a56", muted="#a4a29b", line="#e4e2dc",
         blue="#2a78d6", blue2="#7fb0ec", orange="#eb6834", aqua="#1baf7a", sky1="#dcebfb", sky2="#f6f5f1",
         ground="#d9d4c7", scope="#14171c", scope_grid="#262b33", rider="#33475b", skin="#e9b48f", steel="#8f97a3")


# ------------------------------------------------------------------ drawing helpers
def rgba(cr, key, a=1.0):
    cr.set_source_rgba(*hexc(C.get(key, key), a))


def rrect(cr, x, y, w, h, r):
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0); cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi); cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def shadow(cr, x, y, w, h, r, layers=6, a=0.035):
    for i in range(layers, 0, -1):
        rrect(cr, x - i + 2, y - i + 5, w + 2 * i - 4, h + 2 * i - 4, r + i)
        cr.set_source_rgba(0, 0, 0, a)
        cr.fill()


def text(cr, s, x, y, size=14, key="ink", bold=False, align="left", a=1.0):
    cr.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(size)
    ext = cr.text_extents(s)
    dx = {"left": 0, "center": -ext.width / 2 - ext.x_bearing, "right": -ext.width - ext.x_bearing}[align]
    cr.move_to(x + dx, y)
    rgba(cr, key, a)
    cr.show_text(s)


def line(cr, pts, key, wdt, a=1.0, cap=cairo.LINE_CAP_ROUND):
    cr.set_line_width(wdt); cr.set_line_cap(cap); cr.set_line_join(cairo.LINE_JOIN_ROUND)
    cr.move_to(*pts[0])
    for p in pts[1:]:
        cr.line_to(*p)
    rgba(cr, key, a)
    cr.stroke()


def glow_line(cr, pts, key, wdt):
    for k, a in ((4.0, 0.06), (2.6, 0.10), (1.6, 0.18)):
        line(cr, pts, key, wdt * k, a)
    line(cr, pts, key, wdt)


def circle(cr, x, y, r, key, a=1.0, fill=True, wdt=2):
    cr.new_sub_path(); cr.arc(x, y, r, 0, 2 * math.pi)
    rgba(cr, key, a)
    if fill:
        cr.fill()
    else:
        cr.set_line_width(wdt); cr.stroke()


def arrow(cr, x0, y0, x1, y1, key, wdt=4, head=12, glow=True):
    if abs(x1 - x0) + abs(y1 - y0) < 3:
        return
    a = math.atan2(y1 - y0, x1 - x0)
    pts = [(x0, y0), (x1 - head * 0.6 * math.cos(a), y1 - head * 0.6 * math.sin(a))]
    (glow_line if glow else line)(cr, pts, key, wdt)
    cr.move_to(x1, y1)
    cr.line_to(x1 - head * math.cos(a - 0.45), y1 - head * math.sin(a - 0.45))
    cr.line_to(x1 - head * math.cos(a + 0.45), y1 - head * math.sin(a + 0.45))
    cr.close_path(); rgba(cr, key); cr.fill()


def card(cr, x, y, w, h, title, step, dark=False):
    shadow(cr, x, y, w, h, 18)
    rrect(cr, x, y, w, h, 18)
    rgba(cr, "scope" if dark else "card"); cr.fill()
    circle(cr, x + 30, y + 30, 13, "orange" if step == 1 else ("blue" if step == 2 else "aqua"))
    text(cr, str(step), x + 30, y + 35, 14, "card", bold=True, align="center")
    text(cr, title, x + 52, y + 36, 15, "card" if dark else "ink", bold=True)


# ------------------------------------------------------------------ data
def episode(root, plant, fam="driven", ep=2):
    sys.path.insert(0, os.path.join(root, "src"))
    import hyperkkl_bench as hkb
    level = "realistic" if fam == "driven" else "partial"
    es = hkb.load(f"{fam}/{plant}/{level}", "test_op_id" if fam == "driven" else "test_id")
    with open(os.path.join(root, "data_v2", "metadata", fam, plant + ".json")) as f:
        m = json.load(f)
    x = np.asarray(es.x[ep], float)
    u = np.asarray(es.u[ep], float)[: len(x)]
    y = np.asarray(es.y[ep], float)
    return np.arange(len(x)) * float(m["dt"]), u, y, x, m


def scope(cr, X, Y, Wd, Hd, t, rows, k, title):
    """Dark signal card: each row draws itself up to sample k; a faint full trace shows where it goes."""
    card(cr, X, Y, Wd, Hd, title, 3, dark=True)
    top, n = Y + 62, len(rows)
    rh = (Hd - 90) / n
    for r, (sig, lab, key, dots) in enumerate(rows):
        y0 = top + r * rh
        lo, hi = np.nanmin(sig), np.nanmax(sig)
        pad = 0.12 * (hi - lo + 1e-9)
        sy = lambda v: y0 + rh - 14 - (v - lo + pad) / (hi - lo + 2 * pad) * (rh - 28)
        sx = lambda tt: X + 150 + (tt - t[0]) / (t[-1] - t[0]) * (Wd - 175)
        for g in range(5):
            gx = X + 150 + g * (Wd - 175) / 4
            line(cr, [(gx, y0 + 6), (gx, y0 + rh - 10)], "scope_grid", 1)
        line(cr, [(X + 150, y0 + rh - 10), (X + Wd - 25, y0 + rh - 10)], "scope_grid", 1)
        st = max(1, len(t) // 600)
        full = [(sx(t[i]), sy(sig[i])) for i in range(0, len(t), st)]
        line(cr, full, key, 1.2, 0.18)
        if dots:
            for i in range(0, k + 1, max(st, 3)):
                circle(cr, sx(t[i]), sy(sig[i]), 1.6, key, 0.8)
        else:
            pts = [(sx(t[i]), sy(sig[i])) for i in range(0, k + 1, st)] + [(sx(t[k]), sy(sig[k]))]
            if len(pts) > 1:
                glow_line(cr, pts, key, 2.0)
        circle(cr, sx(t[k]), sy(sig[k]), 9, key, 0.25); circle(cr, sx(t[k]), sy(sig[k]), 4.5, key)
        for i, ln in enumerate(lab.split("\n")):
            text(cr, ln, X + 22, y0 + rh / 2 + i * 16 - 2, 12.5 if i == 0 else 11, key if r else "muted",
                 bold=(i == 0))
    text(cr, f"t = {t[k]:5.1f} s", X + Wd - 25, Y + 36, 12, "muted", align="right")


def leaderboard(cr, X, Y, Wd, Hd, rows, note):
    """Static 'who won' card: settled unmeasured NRMSE on test_op_ood / clean, seed mean (lower is better)."""
    card(cr, X, Y, Wd, Hd, "Who won on this system", 4)
    text(cr, "error on hidden states (lower is better)", X + 24, Y + 72, 11.5, "ink2")
    vmax = max(v for _, v, key in rows if key != "muted")
    top, rh = Y + 92, (Hd - 150) / len(rows)
    for i, (name, v, key) in enumerate(rows):
        yy = top + i * rh
        text(cr, f"{i + 1 if key != 'muted' else ''}", X + 24, yy + 15, 12, "muted", bold=True)
        text(cr, name, X + 44, yy + 15, 12, "ink" if key != "muted" else "ink2", bold=(key == "orange"))
        full = Wd - 70
        bw = full if key == "muted" else full * 0.85 * v / vmax
        rrect(cr, X + 44, yy + 22, max(bw, 4), 9, 4.5); rgba(cr, key, 0.9 if key != "muted" else 0.5); cr.fill()
        if key == "muted":                                         # off-scale: broken bar
            for dx in (0.55, 0.58):
                line(cr, [(X + 44 + full * dx, yy + 34), (X + 44 + full * dx + 6, yy + 19)], "card", 3)
        text(cr, f"{v:.4f}", X + Wd - 24, yy + 15, 11.5, "ink2", align="right")
    for i, ln in enumerate(note.split("\n")):
        text(cr, ln, X + 24, Y + Hd - 40 + 17 * i, 11.5, "ink2")


# ------------------------------------------------------------------ cart-pole / Segway
def cartpole_b(root):
    t, u, y, x, m = episode(root, "cartpole_b")
    p_lo, p_hi = x[:, 0].min(), x[:, 0].max()
    vmax = np.percentile(np.abs(x[:, 1]), 99) + 1e-9
    rows = [(u, "input u\noperator command", "muted", False),
            (y[:, 0], "cart position\nmeasured (noisy)", "blue2", True),
            (x[:, 1], "cart velocity\nto be estimated", "orange", False)]
    trail = []
    # cart-pole leaderboard (experiments/bench_v2/results_default.json, driven/cartpole_b/realistic)
    board = [("UDE + gated EKF", 0.0127, "blue"), ("UDE + gated UKF", 0.0128, "blue"),
             ("HyperKKL + C5", 0.0134, "blue"), ("NODE + EKF", 0.0144, "blue"), ("HyperKKL", 0.0157, "orange"),
             ("GRU", 0.0177, "blue"), ("NLOX-M", 0.0211, "blue"), ("EKF / UKF (model)", 0.4069, "muted")]

    def frame(cr, k):
        rgba(cr, "bg"); cr.paint()
        text(cr, "Cart-pole: the balance problem of a Segway", 40, 50, 24, "ink", bold=True)
        text(cr, "one benchmark episode drives all three panels: real system  ·  simplified model  ·  signals",
             40, 78, 14, "ink2")
        p, v, th = x[k, 0], x[k, 1], x[k, 2]

        # ---- 1. real life: Segway rider
        X, Y, Wd, Hd = 40, 105, 430, 410
        card(cr, X, Y, Wd, Hd, "In real life: a Segway rider", 1)
        cr.save(); rrect(cr, X, Y + 55, Wd, Hd - 55, 18); cr.clip()
        g = cairo.LinearGradient(0, Y + 55, 0, Y + Hd)
        g.add_color_stop_rgba(0, *hexc("sky1")); g.add_color_stop_rgba(1, *hexc("sky2")); cr.set_source(g); cr.paint()
        gy = Y + Hd - 70
        rgba(cr, "ground"); cr.rectangle(X, gy, Wd, 70); cr.fill()
        sxp = lambda pp: X + 60 + (pp - p_lo) / (p_hi - p_lo + 1e-9) * (Wd - 120)
        # parallax trees / posts scroll with the cart position
        for i in range(-3, 12):
            tx = X + ((i * 70 - (p - p_lo) * 160) % (Wd + 140)) - 70
            line(cr, [(tx, gy), (tx, gy - 34)], "muted", 3, 0.5)
            circle(cr, tx, gy - 44, 16, "aqua", 0.25)
        wx, wr = sxp(p), 26
        trail.append(wx)
        for j, tx in enumerate(trail[-14:-1]):                       # motion streaks
            line(cr, [(tx - 6, gy - 30 + 4 * (j % 3)), (tx + 6, gy - 30 + 4 * (j % 3))], "muted", 2, 0.05 + 0.03 * j)
        cr.save(); cr.scale(1, 0.25); circle(cr, wx, (gy + 2) * 4, 42, "ink", 0.15); cr.restore()
        circle(cr, wx, gy - wr, wr, "ink"); circle(cr, wx, gy - wr, wr * 0.55, "steel")
        sp = -(p - p_lo) * 3.0
        line(cr, [(wx + wr * 0.5 * math.cos(sp), gy - wr + wr * 0.5 * math.sin(sp)),
                  (wx - wr * 0.5 * math.cos(sp), gy - wr - wr * 0.5 * math.sin(sp))], "ink", 3)
        base = (wx, gy - wr)
        R = lambda L, off=0.0: (base[0] + L * math.sin(th) + off * math.cos(th), base[1] - L * math.cos(th) + off * math.sin(th))
        line(cr, [R(0, -26), R(0, 26)], "ink", 9)                          # deck
        line(cr, [R(4, 18), R(150, 26)], "steel", 6)                       # stem
        line(cr, [R(150, 12), R(150, 40)], "ink", 8)                       # handlebar
        line(cr, [R(6, -10), R(78, -4)], "rider", 13); line(cr, [R(6, 8), R(78, 2)], "rider", 13)   # legs
        line(cr, [R(80, -2), R(150, -4)], "blue", 26)                      # torso (jacket)
        line(cr, [R(140, 0), R(148, 30)], "blue", 9)                       # arm
        circle(cr, *R(178, -4), 15, "skin");
        cr.new_sub_path(); cr.arc(*R(181, -4), 16, math.pi + th, 2 * math.pi + th); rgba(cr, "ink"); cr.fill()   # helmet
        hip = R(110, 0)
        arrow(cr, hip[0], hip[1] - 4, hip[0] + 70 * v / vmax, hip[1] - 4, "orange", 4)
        cr.restore()
        text(cr, "body lean = pole angle   ·   wheel = cart", X + Wd / 2, Y + Hd - 40, 12.5, "ink2", align="center")
        text(cr, "orange: speed, not measured here, must be estimated", X + Wd / 2, Y + Hd - 18, 12.5, "orange",
             align="center")

        # ---- 2. the model: cart-pole
        X2 = X + Wd + 30
        card(cr, X2, Y, Wd, Hd, "The simplified model: cart-pole", 2)
        ry = Y + Hd - 120
        line(cr, [(X2 + 30, ry), (X2 + Wd - 30, ry)], "line", 6)
        for i in range(12):
            tx = X2 + 30 + i * (Wd - 60) / 11
            line(cr, [(tx, ry + 6), (tx, ry + 13)], "muted", 1.5)
        cx = X2 + 60 + (p - p_lo) / (p_hi - p_lo + 1e-9) * (Wd - 120)
        shadow(cr, cx - 45, ry - 38, 90, 30, 8, layers=4)
        g = cairo.LinearGradient(0, ry - 38, 0, ry - 8)
        g.add_color_stop_rgba(0, *hexc("blue2")); g.add_color_stop_rgba(1, *hexc("blue"))
        rrect(cr, cx - 45, ry - 38, 90, 30, 8); cr.set_source(g); cr.fill()
        circle(cr, cx - 26, ry - 4, 8, "ink"); circle(cr, cx + 26, ry - 4, 8, "ink")
        L = 175
        tip = (cx + L * math.sin(th), ry - 38 - L * math.cos(th))
        line(cr, [(cx, ry - 38), (cx, ry - 38 - L - 10)], "line", 1.5)
        cr.new_sub_path(); cr.arc(cx, ry - 38, 60, -math.pi / 2, -math.pi / 2 + th) if th > 0 else \
            cr.arc_negative(cx, ry - 38, 60, -math.pi / 2, -math.pi / 2 + th)
        rgba(cr, "blue"); cr.set_line_width(2); cr.stroke()
        line(cr, [(cx, ry - 38), tip], "blue", 7)
        circle(cr, *tip, 14, "blue"); circle(cr, cx, ry - 38, 5, "card")
        text(cr, f"θ = {math.degrees(th):+.1f}°", cx + (70 if th >= 0 else -70), ry - 120, 13, "blue", bold=True,
             align="center")
        arrow(cr, cx, ry + 30, cx + 90 * v / vmax, ry + 30, "orange", 4)
        text(cr, f"v = {v:+.2f} m/s", cx, ry + 58, 13, "orange", bold=True, align="center")
        text(cr, "measured: cart position x, pole angle θ (blue)", X2 + Wd / 2, Y + Hd - 18, 12.5, "ink2",
             align="center")

        # ---- 3. signals
        X3 = X2 + Wd + 30
        scope(cr, X3, Y, 470, Hd, t, rows, k, "Its signals in the benchmark")
        leaderboard(cr, X3 + 500, Y, W - X3 - 540, Hd, board,
                    "ranks 1-5 statistically tied (3 seeds)\nfilters on the imperfect model: ~30x worse")

    return frame, len(t)


# ------------------------------------------------------------------ driver
def render(name, frame, n, out_dir):
    tmp = tempfile.mkdtemp(prefix="hkb_anim_")
    idx = np.linspace(0, n - 1, FPS * SECONDS).astype(int)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    for i, k in enumerate(idx):
        cr = cairo.Context(surf)
        cr.set_antialias(cairo.ANTIALIAS_BEST)
        frame(cr, int(k))
        surf.write_to_png(os.path.join(tmp, f"f{i:05d}.png"))
    mp4, gif = os.path.join(out_dir, name + ".mp4"), os.path.join(out_dir, name + ".gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf",
                    "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    gif], check=True)
    shutil.copy(os.path.join(tmp, f"f{len(idx) // 2:05d}.png"), os.path.join(out_dir, name + "_still.png"))
    shutil.rmtree(tmp)
    for f in (mp4, gif):
        print("wrote", f, f"{os.path.getsize(f) / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = os.path.join(a.root, "docs", "figures", "systems")
    os.makedirs(out, exist_ok=True)
    for name, fn in {"cartpole_b": cartpole_b}.items():
        if a.only and name not in a.only.split(","):
            continue
        frame, n = fn(a.root)
        render(name, frame, n, out)


if __name__ == "__main__":
    main()
