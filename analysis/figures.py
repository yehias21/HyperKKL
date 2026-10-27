"""Figures F1-F4 (matplotlib, Agg). Vector PDF + PNG, Okabe-Ito colour-blind-safe palette,
one colour (and one marker shape) per METHOD across every figure (analysis/names.py).

Every figure that goes into the paper is drawn at the size it is printed at: width W = 6.5 in (the
TMLR \\textwidth), constrained layout, saved WITHOUT a tight bounding box, so the PDF is exactly W wide
and is included at width=\\textwidth (scale 1; fig2a is drawn at its printed width too). Font floors at
print size: ticks/legend 6.5 pt, in-axes annotations >= 6 pt, titles 7.5 pt. Explanatory footers are off
by default (notes=False); the LaTeX captions carry that text."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
import numpy as np                               # noqa: E402
from matplotlib.colors import Normalize, to_rgb  # noqa: E402
from matplotlib.lines import Line2D              # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator  # noqa: E402

from . import rules as R                         # noqa: E402
from .names import COLOR, FILTER_MARKER, METHOD_MARKER, disp  # noqa: E402
from .store import DATA_DRIVEN, DEV, FILTERS, LEARNED_C3, LOCATIONS, OOD_TIERS, TEST, rel_model_error  # noqa: E402,F401

W = 6.5                                          # TMLR \textwidth, inches
FS_TICK, FS_LEG, FS_ANN, FS_TITLE = 6.5, 6.5, 6.0, 7.5
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": FS_TITLE, "axes.labelsize": 7, "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "lines.markersize": 3.5, "savefig.bbox": None, "savefig.pad_inches": 0.0,
    "font.family": "DejaVu Sans", "legend.frameon": False, "mathtext.default": "regular",
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.hspace": 0.03, "figure.constrained_layout.wspace": 0.03})

FMARK = FILTER_MARKER
SYSMARK = dict(zip(DEV + TEST, "osD^vP" * 2))
BEST_LEARNED, CLIM = "#882255", "#777777"
SHORT = {"highway_traffic": "highway", "two_link_arm": "2-link arm", "quad_tank": "quad tank",
         "bioreactor": "bioreact."}
NICE = {"sequence_ood": "sequence OOD", "shape_ood": "shape OOD", "id": "in-distribution"}
SCEN_DISP = {"exact+outliers": "exact + outliers", "colored_t": "coloured-$t$ noise",
             "drift30 (post-train)": "drift 30% (post-train)", "mismatch10": "mismatch 10%",
             "mismatch20": "mismatch 20%", "mismatch30": "mismatch 30%", "mismatch50": "mismatch 50%"}


def sysname(s):
    return SHORT.get(s, s).replace("_", " ")


def save(fig, out: Path, name: str, tight: bool = False) -> list:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    kw = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    fig.savefig(paths[0], **kw)
    fig.savefig(paths[1], dpi=220, **kw)
    plt.close(fig)
    return paths


def pending(ax, text):
    ax.set_facecolor("#f4f4f4")
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=7.5, color="#444")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linestyle(":"); s.set_color("#999")


def _pct_axis(ax, lo=0.25, hi=4.0):
    """Log-ratio axis: y = NRMSE ratio vs the reference, labelled as % change. Few ticks + light grid."""
    ax.set_yscale("log")
    ticks = [t for t in (0.1, 0.3, 0.5, 1, 2, 4) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([("0" if t == 1 else f"{100 * (t - 1):+.0f}%") for t in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(lo, hi)
    ax.grid(axis="y", color="#e4e4e4", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(1, color="#555", lw=0.6, zorder=1)


def _rel_lum(c):
    rgb = np.array(to_rgb(c))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def _foot(fig, text):
    fig.text(0, 0, text, fontsize=6, va="top")


# ---------------------------------------------------------------------------
# F1 (C1)
# ---------------------------------------------------------------------------

def _c1_panel(ax, T, tier, systems, title, clim=None, first_col=True):
    Tt = T.get(tier, {})
    sy = [s for s in systems if s in Tt]
    if not sy:
        pending(ax, f"{title}\nPENDING")
        return
    ms = ("concat", "film", "lora")             # the dense-subspace 'hyper' variant is an appendix ablation
    off = np.linspace(-0.22, 0.22, len(ms))
    lows = [1 + Tt[s]["pairs"][(m, "latent")]["ci"][0] for s in sy for m in ms if (m, "latent") in Tt[s]["pairs"]]
    lo_lim = 0.25 if min(lows, default=1) > 0.27 else max(0.02, 0.8 * max(min(lows), 0.025))
    hi_lim = 5.0
    for j, m in enumerate(ms):
        for i, s in enumerate(sy):
            p = Tt[s]["pairs"].get((m, "latent"))
            if not p:
                continue
            y = 1 + p["rel"]
            lo, hi = max(1 + p["ci"][0], lo_lim * 1.01), min(1 + p["ci"][1], hi_lim * 0.99)
            sig = p["better"] or p["worse"]
            ax.plot([i + off[j]] * 2, [lo, hi], color=COLOR[m], lw=0.9, zorder=3)
            ax.plot(i + off[j], min(max(y, lo_lim * 1.02), hi_lim * 0.98), METHOD_MARKER[m], ms=3.6,
                    mfc=COLOR[m] if sig else "white", mec=COLOR[m], mew=0.9, zorder=4,
                    label=disp(m) if i == 0 else None)
    _pct_axis(ax, lo_lim, hi_lim)
    ax.set_xticks(range(len(sy)))

    def lab(s):
        v = Tt[s]["means"]["latent"]
        bad = clim is not None and R.failed(v, clim(s, tier))
        return f"{sysname(s)}\n{v:.3f}" + ("$^\\dagger$" if bad else "")
    ax.set_xticklabels([lab(s) for s in sy], fontsize=FS_TICK)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.6, len(sy) - 0.4)
    ax.set_title(title, loc="left", fontsize=FS_TITLE)
    if first_col:
        ax.annotate("latent\nNRMSE", xy=(0, 0), xycoords="axes fraction", xytext=(-3, -10.5),
                    textcoords="offset points", fontsize=FS_ANN, ha="right", va="center", color="#555",
                    annotation_clip=False, linespacing=1.1)


def _c1_controls(ax, res, capacity):
    ax.axis("off")
    rows = []
    lw = {b: capacity.get(("vdp", "lora", b, 4), {}).get("width") for b in (35000, 140000, 560000)}
    spec = [("C1-budget-s", "budget 35k", f"w={lw[35000]}"),
            ("C1-where", "140k = C1-where", f"w={lw[140000]}"),
            ("C1-budget-l", "budget 560k", f"w={lw[560000]}"),
            ("C1-budget", "both budgets (rule)", ""),
            ("C1-rank1", "HyperKKL r=1", ""), ("C1-rank16", "HyperKKL r=16", ""),
            ("C1-ctx", "ctx in budget", ""), ("C1-basewidth-64", "base width 64", ""),
            ("C1-basewidth-256", "base width 256", "")]
    for nid, what, cap in spec:
        r = res.get(nid)
        if not r:
            continue
        p = r.get("progress") or {}
        jobs = f"{p.get('done', '-')}/{p.get('total', '-')}" if p else "-"
        verdict = r["verdict"].split("] ")[-1].split(" (need")[0].replace("C1-where winner ", "winner ")
        verdict = verdict.replace("_ood", "").replace("TOST(+-10%) version", "TOST").replace(
            "lora at this rank within +-10% of rank 4 or better (TOST non-inferiority, 90% CI)",
            "within ±10% of rank 4 or better (TOST)").replace("+-", "±").replace(">=", "≥")
        if nid == "C1-where":
            verdict = "location rule: " + r["verdict"].split(".")[0].replace("location rule over ", "")
        rows.append([what, cap, jobs, r.get("shown", r["label"]), verdict[:95]])
    tb = ax.table(cellText=rows, colLabels=["control", "HyperKKL trunk", "jobs", "rule", "evidence"],
                  loc="upper left", cellLoc="left", colLoc="left", colWidths=[0.16, 0.12, 0.07, 0.08, 0.57])
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.5)
    tb.scale(1, 1.1)
    for (i, j), cell in tb.get_celld().items():
        cell.set_linewidth(0.3)
        if i == 0:
            cell.set_text_props(weight="bold")
        if j == 3 and i > 0:
            cell.set_text_props(color={"PASS": "#006d2c", "FAIL": "#a50f15"}.get(cell.get_text().get_text().rstrip("*"), "#444"))
    ax.set_title("(e) C1 controls (pre-registered rules; TOST and seed counts in the decision-rule table)", loc="left")


def fig1(res: dict, capacity: dict, out: Path, store=None, panel_e: bool = True, name: str = None,
         notes: bool = False) -> list:
    """panel_e=False -> the main-text variant (fig1_c1_where_main): panels (a)-(d) only, no cropping."""
    name = name or ("fig1_c1_where" if panel_e else "fig1_c1_where_main")

    def clim_of(nid):
        return (lambda s, t: store.climatology(nid, s, t)[0]) if store is not None else None
    fig = plt.figure(figsize=(W, 5.7 if panel_e else 3.2), layout="constrained")
    gs = fig.add_gridspec(3 if panel_e else 2, 2, height_ratios=[1, 1, 0.85] if panel_e else [1, 1])
    cw, ct = res["C1-where"], res["TEST-C1"]
    ylab = "NRMSE change vs latent\n(log-ratio scale)"
    for c, tier in enumerate(OOD_TIERS):
        v = cw["data"]["rules"]["weight_space"][tier]["vote"]
        ax = fig.add_subplot(gs[0, c])
        _c1_panel(ax, cw["data"]["T"], tier, DEV,
                  f"({'ab'[c]}) dev plants, {NICE[tier]}\nHyperKKL (r=4) beats latent & concat: {v.frac}",
                  clim_of("C1-where"), first_col=(c == 0))
        if c == 0:
            ax.set_ylabel(ylab)
            ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.03), handletextpad=0.2, columnspacing=0.9,
                      borderaxespad=0.2)
        ax = fig.add_subplot(gs[1, c])
        T = ct["data"].get("T", {})
        if ct["label"] == R.PENDING or not T.get(tier):
            p = ct["progress"]
            pending(ax, f"({'cd'[c]}) held-out plants, {NICE[tier]}\nPENDING: TEST-C1 {p['done']}/{p['total']} jobs")
        else:
            vt = ct["data"]["rules"]["weight_space"][tier]["vote"]
            _c1_panel(ax, T, tier, TEST, f"({'cd'[c]}) held-out plants (frozen recipe), {NICE[tier]}\n"
                      f"HyperKKL (r=4) beats latent & concat: {vt.frac}", clim_of("TEST-C1"), first_col=(c == 0))
            if c == 0:
                ax.set_ylabel(ylab)
    if panel_e:
        _c1_controls(fig.add_subplot(gs[2, :]), res, capacity)
    if notes:
        _foot(fig, "Filled marker: paired 95% CI (t over seeds) excludes 0; open: n.s. Bars: 95% CI. Numbers under "
              "plant names: absolute NRMSE of latent injection; $\\dagger$ = above climatology.")
    return save(fig, out, name, tight=notes)


# ---------------------------------------------------------------------------
# F2 (C2)
# ---------------------------------------------------------------------------

FIG2A_W = 0.75 * W                               # printed at 0.75\textwidth (appendix/tables.tex)


def fig2a(res: dict, out: Path, notes: bool = False) -> list:
    fig, ax = plt.subplots(1, 1, figsize=(FIG2A_W, 2.6), layout="constrained")
    cs = [(n, res[n]["data"]["c"]) for n in ("C2-floor-025", "C2-floor-05", "C2-floor-1") if res[n]["data"].get("c")]
    cmark = {0.25: "o", 0.5: "s", 1.0: "^"}
    ylo, yhi = -60, 160
    m = "lora"
    ax.axhspan(-10, 10, color="#eeeeee", zorder=0, lw=0)
    ax.axhline(0, color="#555", lw=0.6)
    for i, s in enumerate(DEV):
        for j, (n, c) in enumerate(cs):
            for k, (tier, filled) in enumerate((("id", False), ("sequence_ood", True))):
                d = res[n]["data"]["pairs"].get(s, {}).get((m, tier))
                if not d:
                    continue
                p = d["p"]
                x = i + (j - 1) * 0.24 + (k - 0.5) * 0.09
                y, lo, hi = 100 * p["rel"], 100 * p["ci"][0], 100 * p["ci"][1]
                ax.plot([x, x], [max(lo, ylo), min(hi, yhi)], color=COLOR[m], lw=0.6, alpha=0.8)
                if hi > yhi:          # clipped: arrow beyond the edge, the CI end printed beside it
                    ax.plot(x, yhi + 5, marker=(3, 0, 0), color=COLOR[m], ms=2.8, clip_on=False)
                    ax.text(x, yhi + 10, f"{hi:+.0f}", fontsize=FS_ANN, color=COLOR[m], ha="center",
                            va="bottom", rotation=90, clip_on=False)
                if lo < ylo:
                    ax.plot(x, ylo - 5, marker=(3, 0, 180), color=COLOR[m], ms=2.8, clip_on=False)
                    ax.text(x, ylo - 10, f"{lo:+.0f}", fontsize=FS_ANN, color=COLOR[m], ha="center",
                            va="top", rotation=90, clip_on=False)
                ax.plot(x, min(max(y, ylo), yhi), cmark[c], color=COLOR[m], mfc=COLOR[m] if filled else "white",
                        ms=3.2, mew=0.8)
    ax.set_xticks(range(len(DEV)))
    ax.set_xticklabels([sysname(s) for s in DEV])
    ax.tick_params(axis="x", length=0, pad=26)
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-0.6, len(DEV) - 0.4)
    ax.set_title(f"{disp(m)}: forgetting floor vs no floor", loc="left", pad=24)
    ax.set_ylabel("NRMSE change vs no floor (%)")
    grey = "#555555"                 # legend handles neutral: the c-markers are shapes only
    h = [Line2D([], [], ls="", marker=cmark[c], color=grey, mfc=grey, ms=3.6, label=f"c = {c:g}") for _, c in cs]
    h += [Line2D([], [], ls="", marker="o", color=grey, mfc=grey, ms=3.6, label="filled: seq. OOD"),
          Line2D([], [], ls="", marker="o", color=grey, mfc="white", ms=3.6, label="open: id"),
          Line2D([], [], color="#dddddd", lw=5, label="±10% margin")]
    fig.legend(handles=h, loc="outside upper center", ncol=len(h), handletextpad=0.3, columnspacing=1.0)
    if notes:
        v = res["C2-floor-05"]
        _foot(fig, f"Rule (c = 0.5), {v.get('reread_note', '')}: {v['label']} (as pre-registered for lora and hyper: "
              f"{v.get('original_label')}). {v['verdict']}.")
    return save(fig, out, "fig2a_c2_floor", tight=notes)


def _probe_curve(runs_sys, m):
    L, D = None, []
    for r in runs_sys.values():
        mp = (r.get("memory_probe") or {}).get(m) or {}
        if "d_other" in mp:
            L = np.asarray(mp["L_s"], float)
            D.append(np.asarray(mp["d_other"], float))
    if not D:
        return None
    D = np.exp(np.mean(np.log(np.maximum(np.array(D), 1e-12)), 0))
    return L, D


def fig2b(store, res: dict, out: Path, notes: bool = False) -> list:
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.2), layout="constrained")
    for ax, s in zip(axes.ravel(), DEV):
        txt = []
        alpha, ref = None, None
        ys = []                                       # every positive value drawn as data in this panel
        for nid, ls, tag in (("C1-where", "-", "no floor"), ("C2-floor-05", "--", r"floor $0.5\alpha$")):
            runs = store.runs(nid).get(s, {})
            for m in ("lora",):
                c = _probe_curve(runs, m)
                if c is None:
                    continue
                L, D = c
                ax.plot(L, D, ls, color=COLOR[m], lw=1.0, label=f"{disp(m)}, {tag}" if s == DEV[0] else None)
                ys.append(D[D > 0])
                if ref is None and m == "lora" and nid == "C1-where":
                    i0 = int(np.searchsorted(L, 0.25 * L.max()))
                    ref = (L[i0], D[i0], L.max())
                mps = [(r.get("memory_probe") or {}).get(m) or {} for r in runs.values()]
                rate = np.nanmedian([p.get("rate_other", np.nan) for p in mps])
                pole = np.nanmedian([p.get("alpha_sfm", np.nan) for p in mps])
                alpha = np.nanmedian([p.get("alpha_design", np.nan) for p in mps])
                if m == "lora":
                    txt.append(f"{tag}: fit {rate:.2f}, SFM pole {pole:.2f}")
                # fitted rate through the tail
                if np.isfinite(rate):
                    Lt = L[L >= 0.25 * L.max()]
                    y0 = D[L >= 0.25 * L.max()][0]
                    ax.plot(Lt, y0 * np.exp(-rate * (Lt - Lt[0])), ":", color=COLOR[m], lw=0.7,
                            label="fitted tail rate" if (s == DEV[0] and nid == "C1-where") else None)
        ax.set_yscale("log")                          # log FIRST, limits from the data afterwards
        if alpha and ref is not None:
            L0, D0, Lmax = ref
            Lr = np.linspace(L0, Lmax, 50)
            yr = D0 * np.exp(-alpha * (Lr - L0))
            keep = yr > D0 * 1e-3
            ax.plot(Lr[keep], yr[keep], color="k", lw=0.8, label=r"decay at $\alpha$" if s == DEV[0] else None)
            ax.plot(Lr, D0 * np.exp(-0.5 * alpha * (Lr - L0)), color="k", lw=0.6, ls=(0, (1, 2)),
                    label=r"decay at $0.5\alpha$" if s == DEV[0] else None)
            txt.append(rf"$\alpha$ = {alpha:.2f} 1/s")
        pos = np.concatenate(ys) if ys else np.array([])
        pos = pos[np.isfinite(pos) & (pos > 0)]
        if pos.size:                                  # never a non-positive limit on a log axis
            lo, hi = pos.min() / 1.5, pos.max() * 1.5
            ax.set_ylim(lo * (lo / hi) ** 0.5, hi)      # bottom ~third left free for the text block
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(s.replace("_", " ") + (" (probe flat)" if s == "lorenz" else ""), loc="left")
        ax.text(0.02, 0.03, "\n".join(txt), transform=ax.transAxes, fontsize=FS_ANN, va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|\Delta\hat x|(L)$ (history swap)")
    for ax in axes[1]:
        ax.set_xlabel("kept input history L (s)")
    fig.legend(loc="outside upper center", ncol=5, handletextpad=0.3, columnspacing=1.2)
    if notes:
        _foot(fig, "Memory probe (id tier, 16 episodes, geometric mean over seeds).")
    return save(fig, out, "fig2b_c2_probe", tight=notes)


def fig2c(res: dict, out: Path, notes: bool = False) -> list:
    fig, axes = plt.subplots(2, 3, figsize=(W, 3.9), sharey=True, layout="constrained")
    W1 = res["C2-window-a1"]["data"].get("W", {})
    W2 = res["C2-window-a2"]["data"].get("W", {}) if res["C2-window-a2"]["label"] != R.PENDING else {}
    for ax, s in zip(axes.ravel(), DEV):
        if s not in W1:
            pending(ax, f"{s}\nPENDING")
            continue
        for Wd, ls, lab in ((W1, "-", r"$\alpha$"), (W2, "--", r"$2\alpha$")):
            w = Wd.get(s)
            if not w:
                continue
            for tier, filled in (("sequence_ood", True), ("id", False)):
                d = w.get(tier)
                if not d:
                    continue
                Ls = d["L"]
                y = [d["mean"][L] / d["mean"][d["Lmax"]] for L in Ls]
                ax.plot(Ls, y, ls + "o", color=COLOR["lora"], mfc=COLOR["lora"] if filled else "white", ms=2.8,
                        lw=0.8, label=f"{NICE[tier]}, {lab}" if s == DEV[0] else None)
            ax.axvline(w["Lstar"], color="k", lw=0.7, ls=ls)
            if w["Lref"] <= 4096:
                ax.axvline(w["Lref"], color=BEST_LEARNED, lw=0.7, ls=ls)
        ax.axhline(1.05, color=CLIM, lw=0.6, ls=":")
        ax.axhline(1.0, color="#555", lw=0.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        w = W1[s]
        ax.set_title(f"{s.replace('_', ' ')}: $L^*$ = {w['Lstar']:.0f}, refined {w['Lref']:.0f}", loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("e(L) / e(512)")
    for ax in axes[1]:
        ax.set_xlabel("window length L (samples)")
    h, _ = axes.ravel()[0].get_legend_handles_labels()
    h += [Line2D([], [], color="k", lw=0.7, label="$L^*$ (pre-registered)"),
          Line2D([], [], color=BEST_LEARNED, lw=0.7, label="refined $L^*$"),
          Line2D([], [], color=CLIM, lw=0.6, ls=":", label="+5%")]
    fig.legend(handles=h, loc="outside upper center", ncol=len(h), handletextpad=0.3, columnspacing=1.0)
    if notes:
        _foot(fig, "HyperKKL with a raw input window (lora@window), mean over 3 seeds."
              + ("" if W2 else " C2-window-a2 (2 alpha) PENDING."))
    return save(fig, out, "fig2c_c2_window", tight=notes)


# ---------------------------------------------------------------------------
# F3 (C3): the crossover
# ---------------------------------------------------------------------------

SCENARIOS = [  # (label, node, condition, knowledge, mismatch)
    ("exact", "C3-faults", "clean", "full", 0.0),
    ("exact+outliers", "C3-faults", "outliers", "full", 0.0),
    ("colored_t", "C3-noise-ct", "clean", "full", 0.0),
    ("drift30 (post-train)", "C3-faults", "plant_drift30", "full", 0.3),
    ("mismatch10", "C3-mismatch10", "clean", "full", 0.1),
    ("mismatch20", "C3-mismatch20", "clean", "full", 0.2),
    ("mismatch30", "C3-mismatch30", "clean", "full", 0.3),
    ("mismatch50", "C3-mismatch50", "clean", "full", 0.5),
    ("partial", "C3-partial", "clean", "partial", 0.0),
    ("realistic", "C3-realistic", "clean", "partial", 0.3),
]


def scenario_order(systems=DEV):
    me = {lab: np.median([rel_model_error(s, k, m) for s in systems]) for lab, _, _, k, m in SCENARIOS}
    order = sorted(range(len(SCENARIOS)), key=lambda i: (me[SCENARIOS[i][0]], i))
    return [SCENARIOS[i] for i in order], me


def c3_grid(store, systems, tier, scen):
    from .evidence import c3_block
    G = {}
    for lab, nid, cond, k, m in scen:
        for s in systems:
            G[(lab, s)] = c3_block(store, nid, s, tier, cond)
    return G


def _c3_legend(ax, learned, filters, dd_methods, dd_open):
    """Three one-row keys in a dedicated strip: learned side, model-based filters, data-driven baselines.
    Only methods that actually appear in the figure get an entry."""
    ax.axis("off")
    kw = dict(handletextpad=0.3, columnspacing=1.0, borderaxespad=0.0, frameon=False, handlelength=1.6)
    g = "#888888"
    rows = []
    h1 = [Line2D([], [], color=BEST_LEARNED, marker="o", mfc=g, mec=g, ms=3.4,
                 label="best learned" + (" (colour = method):" if len(learned) > 1 else ":"))]
    h1 += [Line2D([], [], ls="", marker=METHOD_MARKER.get(m, "o"), color=COLOR[m], ms=3.6, label=disp(m, short=True))
           for m in learned]
    h1 += [Line2D([], [], color="#a50f15", ls="", marker="o", mfc="none", ms=6, mew=0.9, label="learned FAILED"),
           Line2D([], [], color=CLIM, ls="--", label="climatology")]
    rows.append(h1)
    h2 = [Line2D([], [], color="k", marker="o", mfc="white", ms=3.2, label="best working filter:")]
    h2 += [Line2D([], [], ls="", marker=FMARK.get(f, "o"), color="k", mfc="white", ms=3.4, label=disp(f, short=True))
           for f in filters]
    rows.append(h2)
    h3 = [Line2D([], [], color=CLIM, ls="", marker="x", ms=3.4, label="best filter FAILED")]
    if dd_methods or dd_open:
        h3 += [Line2D([], [], ls="", marker="D", color=COLOR.get(m, "#CC79A7"), ms=3.0,
                     label=("best data-driven: " if k == 0 else "") + disp(m, short=True)) for k, m in enumerate(dd_methods)]
        if dd_open:
            h3 += [Line2D([], [], ls="", marker="D", color=COLOR["nlox"], mfc="white", ms=3.0, mew=0.8,
                          label="open: only NLOX merged so far")]
    rows.append(h3)
    ys = [1 - (k + 0.5) / len(rows) for k in range(len(rows))]
    for h, y in zip(rows, ys):
        ax.add_artist(ax.legend(handles=h, loc="center", bbox_to_anchor=(0.5, y), ncol=len(h), **kw))


def fig3(store, res, out: Path, tier="sequence_ood", systems=DEV, name="fig3_c3_crossover", scen=None,
         show_model_error: bool = True, only_data: bool = False, notes: bool = False) -> list:
    """show_model_error=False -> main-text variant (2x3 grid, no cropping needed).
    only_data=True -> draw only plants with finished runs, in one row; the others are named as pending."""
    scen = scen or scenario_order(DEV)[0]
    G = c3_grid(store, systems, tier, scen)
    pend = []
    if only_data:
        have = [s for s in systems if any(G[(lab, s)] is not None for lab, *_ in scen)]
        pend = [s for s in systems if s not in have]
        systems = have or systems
    ncol = 3 if (len(systems) > 3 or not only_data) else max(1, len(systems))
    nrow = int(np.ceil(len(systems) / ncol))
    one_x = len(scen) == 1
    # which keys are needed
    learned, filters, dd, dd_open = [], [], [], False
    for b in G.values():
        if b is None:
            continue
        if b["BL"] not in learned:
            learned.append(b["BL"])
        c = b["comps"].get("working")
        if c and c["filter"] not in filters:
            filters.append(c["filter"])
        d = b["comps"].get("data_driven")
        if d:
            dd.append(d["filter"])
        elif "nlox" in b["means"]:
            dd_open = True
    learned = [m for m in LEARNED_C3 if m in learned]
    dd = [m for m in DATA_DRIVEN if m in dd]
    filters = [f for f in FMARK if f in filters]
    dd_keys = []
    for m in dd:
        k = {"koopman_kf_bilinear": "koopman_kf", "greybox_ukf": "greybox_ekf", "greybox_ekf_oracle": "greybox_ekf"}.get(m, m)
        if k not in dd_keys:
            dd_keys.append(k)
    leg_h = 0.15 * 3
    if one_x:
        return _fig3_dots(G, scen[0][0], systems, tier, out, name, leg_h, learned, filters, dd_keys, dd_open, pend)
    xlab_h = 0.2 if (one_x or show_model_error) else 0.95
    row_h = 1.12
    hr = [leg_h] + [row_h] * nrow + ([1.35] if show_model_error else [])
    fig = plt.figure(figsize=(W, sum(hr) + xlab_h + (0.15 if pend else 0.0)), layout="constrained")
    gs = fig.add_gridspec(len(hr), ncol, height_ratios=hr)
    _c3_legend(fig.add_subplot(gs[0, :]), learned, filters, dd_keys, dd_open)
    x = np.arange(len(scen))
    labels = [SCEN_DISP.get(s[0], s[0]) for s in scen]
    axes = []
    for i, s in enumerate(systems):
        ax = fig.add_subplot(gs[1 + i // ncol, i % ncol], sharex=axes[0] if axes else None)
        axes.append(ax)
        bl, bw, cl = [], [], []
        any_data = False
        for j, (lab, *_rest) in enumerate(scen):
            b = G[(lab, s)]
            if b is None:
                bl.append(np.nan); bw.append(np.nan); cl.append(np.nan)
                ax.text(j, 0.5, "p", ha="center", va="center", fontsize=FS_ANN, color="#999",
                        transform=ax.get_xaxis_transform())
                continue
            any_data = True
            L = b["BL"]
            bl.append(b["means"][L])
            ax.plot(j, b["means"][L], METHOD_MARKER.get(L, "o"), color=COLOR[L], ms=3.6, zorder=4)
            if R.failed(b["means"][L], b["clim"]):
                ax.plot(j, b["means"][L], "o", mfc="none", mec="#a50f15", ms=6.5, mew=0.9, zorder=5)
            c = b["comps"].get("working")
            bw.append(b["means"][c["filter"]] if c else np.nan)
            if c:
                ax.plot(j, b["means"][c["filter"]], FMARK.get(c["filter"], "o"), color="k", ms=3.2, mfc="white", zorder=4)
                # win marks sit in a strip ABOVE the axes, so they never cover data or the climatology line
                if R.better(c["p"]):
                    ax.text(j, 1.015, "L", ha="center", va="bottom", fontsize=FS_ANN, color=BEST_LEARNED,
                            fontweight="bold", transform=ax.get_xaxis_transform(), clip_on=False)
                elif R.worse(c["p"]):
                    ax.text(j, 1.015, "F", ha="center", va="bottom", fontsize=FS_ANN, color="k",
                            fontweight="bold", transform=ax.get_xaxis_transform(), clip_on=False)
            if b["BF"] in b["failed"]:
                ax.plot(j, b["means"][b["BF"]], "x", color=CLIM, ms=3.2, zorder=3)
            d = b["comps"].get("data_driven")
            dm = d["filter"] if d else ("nlox" if "nlox" in b["means"] else None)
            if dm:                      # filled: best of all data-driven baselines; open: only nlox merged so far
                ax.plot(j + 0.18, b["means"][dm], "D", color=COLOR.get(dm, "#CC79A7"), ms=2.9,
                        mfc=COLOR.get(dm, "#CC79A7") if d else "white", mew=0.8, zorder=4)
            cl.append(b["clim"])
            if b["failed"]:
                ax.text(j, 0.025, f"{len(b['failed'])}", ha="center", va="bottom", fontsize=FS_ANN, color="#a50f15",
                        transform=ax.get_xaxis_transform())
        if not any_data:
            pending(ax, f"{s}\nPENDING")
            continue
        if not one_x:
            ax.plot(x, bl, "-", color=BEST_LEARNED, lw=0.9, zorder=2)
            ax.plot(x, bw, "-", color="k", lw=0.9, zorder=2)
            ax.plot(x, cl, "--", color=CLIM, lw=0.8, zorder=1)
        else:
            ax.axhline(cl[0], color=CLIM, ls="--", lw=0.8, zorder=1)
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo / 3.0, hi)               # head-room at the bottom for the FAILED counts
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.set_xticks(x)
        bottom = (i // ncol == nrow - 1) or (i + ncol >= len(systems))
        if bottom and not show_model_error and not one_x:
            ax.set_xticklabels(labels, rotation=50, ha="right", rotation_mode="anchor")
        elif one_x:
            ax.set_xticklabels(labels)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlim(-0.6, len(scen) - 0.4)
        ax.set_title(s.replace("_", " "), loc="left", pad=11)
        if i % ncol == 0:
            ax.set_ylabel("NRMSE" if nrow > 1 else f"NRMSE ({NICE[tier]})")
    if pend:
        fig.text(0.995, 0.0, "pending (no finished runs yet): " + ", ".join(p.replace("_", " ") for p in pend),
                 ha="right", va="bottom", fontsize=FS_ANN, color="#444")
    if show_model_error and not one_x:
        ax = fig.add_subplot(gs[-1, :])
        for s in systems:
            me = [rel_model_error(s, k, m) for _, _, _, k, m in scen]
            ax.plot(x, np.maximum(me, 1e-3), "-" + SYSMARK[s], color="#444", lw=0.6, ms=3, mfc="white",
                    label=s.replace("_", " "))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.4, len(scen) - 0.6)
        ax.set_ylabel("relative model error")
        ax.set_title(r"relative model error, median $\|f_{real}-f_{model}\|\,/\,\|f_{real}\|$ over the IC box "
                     r"(exact model drawn at $10^{-3}$)", loc="left")
        ax.legend(ncol=2, loc="upper left", columnspacing=1.0)
    if notes:
        _foot(fig, "Top marks: L = learned beats the best working filter (paired 95% CI), F = filter wins; red numbers "
              "= filters above climatology. 'p' = pending.")
    return save(fig, out, name, tight=notes)


def _fig3_dots(G, lab, systems, tier, out, name, leg_h, learned, filters, dd_keys, dd_open, pend):
    """One scenario only (held-out plants): a single dot plot, x = plant, instead of one panel per plant."""
    fig = plt.figure(figsize=(W, leg_h + 2.1), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[leg_h, 2.1])
    _c3_legend(fig.add_subplot(gs[0]), learned, filters, dd_keys, dd_open)
    ax = fig.add_subplot(gs[1])
    shown = []
    for j, s in enumerate(systems):
        b = G[(lab, s)]
        shown.append(s)
        if b is None:
            ax.text(j, 0.5, "pending", ha="center", va="center", fontsize=FS_ANN, color="#999",
                    transform=ax.get_xaxis_transform())
            continue
        L = b["BL"]
        ax.plot(j - 0.12, b["means"][L], METHOD_MARKER.get(L, "o"), color=COLOR[L], ms=4, zorder=4)
        if R.failed(b["means"][L], b["clim"]):
            ax.plot(j - 0.12, b["means"][L], "o", mfc="none", mec="#a50f15", ms=7, mew=0.9, zorder=5)
        ax.plot([j - 0.38, j + 0.38], [b["clim"]] * 2, "--", color=CLIM, lw=0.8, zorder=1)
        c = b["comps"].get("working")
        if c:
            ax.plot(j + 0.06, b["means"][c["filter"]], FMARK.get(c["filter"], "o"), color="k", ms=3.6, mfc="white", zorder=4)
            mark = "L" if R.better(c["p"]) else ("F" if R.worse(c["p"]) else "")
            if mark:
                ax.text(j, 1.015, mark, ha="center", va="bottom", fontsize=FS_ANN, fontweight="bold",
                        color=BEST_LEARNED if mark == "L" else "k", transform=ax.get_xaxis_transform(), clip_on=False)
        if b["BF"] in b["failed"]:
            ax.plot(j + 0.06, b["means"][b["BF"]], "x", color=CLIM, ms=3.6, zorder=3)
        d = b["comps"].get("data_driven")
        dm = d["filter"] if d else ("nlox" if "nlox" in b["means"] else None)
        if dm:
            ax.plot(j + 0.24, b["means"][dm], "D", color=COLOR.get(dm, "#CC79A7"), ms=3.2,
                    mfc=COLOR.get(dm, "#CC79A7") if d else "white", mew=0.8, zorder=4)
        if b["failed"]:
            ax.text(j, 0.025, f"{len(b['failed'])}", ha="center", va="bottom", fontsize=FS_ANN, color="#a50f15",
                    transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo / 2.5, hi)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([s.replace("_", " ") for s in shown])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(shown) - 0.4)
    for j in range(1, len(shown)):
        ax.axvline(j - 0.5, color="#eeeeee", lw=0.6, zorder=0)
    ax.set_ylabel(f"NRMSE ({NICE[tier]})")
    ax.set_title(f"{SCEN_DISP.get(lab, lab)}, held-out plants", loc="left", pad=11)
    if pend:
        ax.text(1.0, 1.08, "no finished runs yet: " + ", ".join(p.replace("_", " ") for p in pend),
                ha="right", va="bottom", fontsize=FS_ANN, color="#444", transform=ax.transAxes)
    return save(fig, out, name)


# ---------------------------------------------------------------------------
# F4 (benchmark): scorecard heat-map
# ---------------------------------------------------------------------------

HIGHER_BETTER = {"auroc"}


def scorecard_ranks(store, nid, systems):
    from hyperkkl.tree import SC_COLS, scorecard_table
    tabs = {s: scorecard_table(ss) for s, ss in store.runs(nid).items() if s in systems and ss}
    if not tabs:
        return None
    methods = []
    for t in tabs.values():
        for m in t:
            if m not in methods and m not in ("autonomous", "hyper"):     # hyper: appendix ablation only
                methods.append(m)
    cols = [c for c in SC_COLS if c[0] != "us" and any(t[m].get(c[0], (None,))[0] is not None
                                                       for t in tabs.values() for m in t)]
    ranks = np.full((len(methods), len(cols), len(tabs)), np.nan)
    for k, (s, t) in enumerate(tabs.items()):
        for j, (key, _, _) in enumerate(cols):
            v = np.array([(t.get(m, {}).get(key, (None,))[0]) for m in methods], dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            vv = -v if key in HIGHER_BETTER else v
            from scipy.stats import rankdata
            ranks[ok, j, k] = rankdata(vv[ok], method="average")      # ties share their mean rank
    return methods, cols, ranks, tabs


def _col_label(s):
    return (s.replace("sequence_ood", "sequence OOD").replace("shape_ood", "shape OOD")
            .replace("poor_init", "poor init").replace("_", " "))


def fig4(store, res, out: Path, systems=DEV, notes: bool = False) -> list:
    """Panels stacked (each gets the full width); each panel has its own colour bar, because a rank among
    n methods and a rank among n' methods are on different scales. Cell text black/white by luminance."""
    data = []
    for nid, title in (("C3-faults", "(a) exact model, stress conditions (C3-faults)"),
                       ("C3-realistic", "(b) realistic deployment (C3-realistic)")):
        data.append((nid, title, scorecard_ranks(store, nid, systems)))
    nrows = [len(r[0]) if r else 6 for _, _, r in data]
    fig = plt.figure(figsize=(W, 2.3 + 0.16 * sum(nrows)), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[n + 6.0 for n in nrows])
    cmap = plt.get_cmap("cividis_r")
    for k, (nid, title, r) in enumerate(data):
        ax = fig.add_subplot(gs[k])
        if r is None:
            pending(ax, f"{nid}\nPENDING")
            continue
        methods, cols, ranks, tabs = r
        M = np.nanmean(ranks, 2)
        norm = Normalize(1, len(methods))
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=FS_TICK,
                            color="white" if _rel_lum(cmap(norm(M[i, j]))) < 0.179 else "black")   # WCAG contrast crossover
        # failed counts (sequence_ood clean above climatology)
        lab = []
        for m in methods:
            nf = 0
            for s in tabs:
                v = tabs[s].get(m, {}).get("sequence_ood", (None,))[0]
                c = store.climatology(nid, s, "sequence_ood")[0]
                nf += bool(v is not None and R.failed(v, c))
            lab.append(disp(m, short=True) + (f" [F{nf}/{len(tabs)}]" if nf else ""))
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(lab)
        for t, m in zip(ax.get_yticklabels(), methods):
            t.set_color(COLOR.get(m, "k") if (m in LOCATIONS or m in DATA_DRIVEN) else "k")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([_col_label(c[1]) for c in cols], rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(length=0)
        ax.set_title(title, loc="left")
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.9, aspect=22, pad=0.01,
                     label=f"mean rank among {len(methods)} methods\n(1 = best)")
    if notes:
        _foot(fig, "Rank of each method per scorecard column per dev plant (ties share the mean rank), averaged.")
    return save(fig, out, "fig4_scorecard", tight=notes)
