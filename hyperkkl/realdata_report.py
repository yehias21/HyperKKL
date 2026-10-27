"""Statistics, decision rules (experiments/real/DESIGN.md), results.json, figures for the
real-data battery SOC experiment. Input: experiments/real/cache/{prep,predictions}.pt."""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from . import realdata as R
from .realdata_run import CACHE, METHODS, OUT, SEEDS, SETTLE, soc_metrics

FIG = OUT / "figures"
TEST_SETS = {"US06": ["test25/US06"], "HWFET": ["test25/HWFTa", "test25/HWFTb"], "NN": ["test25/NN"]}
BLOCK = 6
N_BOOT = 10000


def per_window(vals, meta_sets):
    """vals: list over sets of per-episode values -> list over sets of (starts, per-window mean)."""
    out = []
    for v, m in zip(vals, meta_sets):
        st = m["start"]
        us = np.unique(st)
        out.append((us, np.array([v[st == s].mean() for s in us])))
    return out


def block_bootstrap(series: list, block=BLOCK, n_boot=N_BOOT, seed=0) -> tuple:
    """Moving-block bootstrap of the mean over windows, blocks within each recording."""
    rng = np.random.RandomState(seed)
    allv = np.concatenate(series)
    means = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for s in series:
            n = len(s)
            L = min(block, n)
            k = int(math.ceil(n / L))
            st = rng.randint(0, n - L + 1, size=k)
            parts.append(np.concatenate([s[i:i + L] for i in st])[:n])
        means[b] = np.concatenate(parts).mean()
    return float(allv.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(log=print):
    prep = torch.load(CACHE / "prep.pt", weights_only=False)
    P = torch.load(CACHE / "predictions.pt", weights_only=False)
    pred, truth, meta, full = P["pred"], P["truth"], P["meta"], P["full"]
    data_bl = []
    pbs = sorted(CACHE.glob("predictions_baselines_seed*.pt"))
    if pbs:                                 # Addendum A: data-driven baselines (seeded)
        from .realdata_baselines import DATA_BASELINES
        for p_ in pbs:
            B = torch.load(p_, weights_only=False)
            pred.update(B["pred"])
            for k, d in B["full"].items():
                full[k].update(d)
        data_bl = list(DATA_BASELINES)
        missing = [(n, b, sd) for n in truth for b in data_bl for sd in SEEDS if (n, b, sd) not in pred]
        if missing:
            raise RuntimeError(f"data-baseline predictions missing, e.g. {missing[:3]} ({len(missing)})")
    sets = list(truth)
    met = {}                                    # (set, method, seed|None) -> metrics dict
    for k, v in pred.items():
        met[k] = soc_metrics(v, truth[k[0]])
    learned = [m for m in METHODS] + data_bl          # every seeded method
    # WS-best per seed, on validation
    ws_choice = {}
    for s in SEEDS:
        a = {m: float(np.mean(met[("val", m, s)]["rmse_settled"])) for m in ("lora", "hyper")}
        ws_choice[s] = min(a, key=a.get)
    for name in sets:
        for s in SEEDS:
            met[(name, "ws_best", s)] = met[(name, ws_choice[s], s)]
            met[(name, "hyperkkl", s)] = met[(name, "lora", s)]     # the paper's method: lora, r = 4
            pred[(name, "hyperkkl", s)] = pred[(name, "lora", s)]
            pred[(name, "ws_best", s)] = pred[(name, ws_choice[s], s)]
    learned_all = learned + ["ws_best", "hyperkkl"]
    filters = ["ekf", "ukf", "cc"]

    def summary(name, m, key="rmse_settled"):
        if m in filters:
            v = met[(name, m)][key]
            return {"mean": float(np.mean(v)), "per_seed": None}
        ps = [float(np.mean(met[(name, m, s)][key])) for s in SEEDS]
        return {"mean": float(np.mean(ps)), "per_seed": ps, "sd": float(np.std(ps, ddof=1))}

    def family_summary(fam_sets, m, key):
        """Episode-weighted over the family's recordings."""
        if m in filters:
            v = np.concatenate([met[(n, m)][key] for n in fam_sets])
            return {"mean": float(np.mean(v)), "n_episodes": int(len(v))}
        ps = [float(np.mean(np.concatenate([met[(n, m, s)][key] for n in fam_sets]))) for s in SEEDS]
        return {"mean": float(np.mean(ps)), "per_seed": ps, "sd": float(np.std(ps, ddof=1)),
                "n_episodes": int(sum(len(met[(n, m, SEEDS[0])][key]) for n in fam_sets))}

    keys = ("rmse_settled", "rmse_all", "max_settled", "t_conv_min")
    fam25 = dict(TEST_SETS)
    fam0 = {n.split("/")[1]: [n] for n in sets if n.startswith("test0/")}
    tables = {}
    for tag, fams in (("val", {"Cycle_4": ["val"]}), ("test25", fam25), ("test0", fam0)):
        tables[tag] = {f: {m: {k: family_summary(ss, m, k) for k in keys} for m in learned_all + filters}
                       for f, ss in fams.items()}
        for f, ss in fams.items():
            for m in learned_all + filters:
                if m in filters:
                    tc = np.concatenate([met[(n, m)]["t_conv_min"] for n in ss])
                    cen = np.concatenate([met[(n, m)]["conv_censored"] for n in ss])
                else:
                    tc = np.concatenate([met[(n, m, s)]["t_conv_min"] for n in ss for s in SEEDS])
                    cen = np.concatenate([met[(n, m, s)]["conv_censored"] for n in ss for s in SEEDS])
                tables[tag][f][m]["t_conv_median_min"] = float(np.median(tc))
                tables[tag][f][m]["conv_censored_frac"] = float(np.mean(cen))

    def paired(a, b, fam_sets):
        """d = RMSE_a - RMSE_b per window (mean over episodes and seeds), block-bootstrap CI,
        per-seed mean differences."""
        series, per_seed = [], []
        for n in fam_sets:
            va = np.mean([met[(n, a, s)]["rmse_settled"] for s in SEEDS], 0) if a not in filters else met[(n, a)]["rmse_settled"]
            vb = np.mean([met[(n, b, s)]["rmse_settled"] for s in SEEDS], 0) if b not in filters else met[(n, b)]["rmse_settled"]
            series.append(per_window([va - vb], [meta[n]])[0][1])
        for s in SEEDS:
            d = []
            for n in fam_sets:
                va = met[(n, a, s)]["rmse_settled"] if a not in filters else met[(n, a)]["rmse_settled"]
                vb = met[(n, b, s)]["rmse_settled"] if b not in filters else met[(n, b)]["rmse_settled"]
                d.append(va - vb)
            per_seed.append(float(np.mean(np.concatenate(d))))
        mean, lo, hi = block_bootstrap(series)
        n_win = int(sum(len(x) for x in series))
        return {"mean_diff": mean, "ci95": [lo, hi], "per_seed_diff": per_seed, "n_windows": n_win,
                "a_better": bool(hi < 0 and all(d < 0 for d in per_seed)),
                "b_better": bool(lo > 0 and all(d > 0 for d in per_seed))}

    tests = {}
    for (a, b) in (("ws_best", "ekf"), ("ws_best", "latent"), ("ws_best", "ukf"), ("lora", "ekf"),
                   ("hyper", "ekf"), ("latent", "ekf"), ("lora", "latent"), ("hyper", "latent"),
                   ("concat", "ekf"), ("ukf", "ekf")):
        tests[f"{a}-{b}"] = {"test25": {f: paired(a, b, ss) for f, ss in fam25.items()},
                             "test0": {f: paired(a, b, ss) for f, ss in fam0.items()}}
    for b in data_bl:
        tests[f"ws_best-{b}"] = {"test25": {f: paired("ws_best", b, ss) for f, ss in fam25.items()},
                                 "test0": {f: paired("ws_best", b, ss) for f, ss in fam0.items()}}
    def verdict(t):
        wa = sum(v["a_better"] for v in t.values())
        wb = sum(v["b_better"] for v in t.values())
        return "A wins" if wa >= 2 else ("B wins" if wb >= 2 else "inconclusive"), wa, wb

    v1, v2 = verdict(tests["ws_best-ekf"]["test25"]), verdict(tests["ws_best-latent"]["test25"])
    lab = {"A wins": "A beats B", "B wins": "B beats A", "inconclusive": "inconclusive"}

    def vs_data(a):
        """Addendum A for method a: vs the best data-driven baseline per test cycle, and vs the
        validation-chosen one."""
        for b in data_bl:
            tests.setdefault(f"{a}-{b}", {"test25": {f: paired(a, b, ss) for f, ss in fam25.items()},
                                          "test0": {f: paired(a, b, ss) for f, ss in fam0.items()}})
        best_c = {f: min(data_bl, key=lambda b: tables["test25"][f][b]["rmse_settled"]["mean"]) for f in fam25}
        best_v = min(data_bl, key=lambda b: tables["val"]["Cycle_4"][b]["rmse_settled"]["mean"])
        t_c = {f: dict(tests[f"{a}-{best_c[f]}"]["test25"][f], baseline=best_c[f]) for f in fam25}
        t_v = {f: dict(tests[f"{a}-{best_v}"]["test25"][f], baseline=best_v) for f in fam25}
        vc, vv = verdict(t_c), verdict(t_v)
        return {"per_cycle_best": {"verdict": f"{a} vs best data baseline: " + lab[vc[0]],
                                   "cycles_method_better": vc[1], "cycles_baseline_better": vc[2], "per_cycle": t_c},
                "validation_chosen": {"baseline": best_v, "verdict": f"{a} vs {best_v}: " + lab[vv[0]],
                                      "cycles_method_better": vv[1], "cycles_baseline_better": vv[2], "per_cycle": t_v}}

    for (a, b) in (("hyperkkl", "ekf"), ("hyperkkl", "ukf"), ("hyperkkl", "latent"), ("hyperkkl", "hyper")):
        tests[f"{a}-{b}"] = {"test25": {f: paired(a, b, ss) for f, ss in fam25.items()},
                             "test0": {f: paired(a, b, ss) for f, ss in fam0.items()}}
    addA = vs_data("ws_best") if data_bl else None
    addA_h = vs_data("hyperkkl") if data_bl else None
    h1h, h2h = verdict(tests["hyperkkl-ekf"]["test25"]), verdict(tests["hyperkkl-latent"]["test25"])
    hyperkkl_reading = {
        "note": "author's single-method re-reading: HyperKKL = lora (rank 4); the pre-registered rule "
                "(DESIGN.md) names WS-best; same CI rule",
        "H1_hyperkkl_vs_ekf": {"verdict": "hyperkkl vs tuned EKF: " + lab[h1h[0]], "cycles_hyperkkl_better": h1h[1],
                               "cycles_ekf_better": h1h[2]},
        "H2_hyperkkl_vs_latent": {"verdict": "hyperkkl vs latent: " + lab[h2h[0]], "cycles_hyperkkl_better": h2h[1],
                                  "cycles_latent_better": h2h[2]},
        "addendum_A_hyperkkl_vs_data_baselines": addA_h}
    decision = {
        "H1_ws_best_vs_ekf": {"verdict": {"A wins": "WS-best (learned) beats the tuned EKF",
                                          "B wins": "the tuned EKF beats WS-best",
                                          "inconclusive": "inconclusive"}[v1[0]],
                              "cycles_learned_better": v1[1], "cycles_ekf_better": v1[2]},
        "H2_ws_best_vs_latent": {"verdict": {"A wins": "weight-space beats latent injection",
                                             "B wins": "latent injection beats weight-space",
                                             "inconclusive": "inconclusive"}[v2[0]],
                                 "cycles_ws_better": v2[1], "cycles_latent_better": v2[2]},
        "ws_choice_per_seed": ws_choice, "addendum_A_ws_best_vs_data_baselines": addA,
        "hyperkkl_r4": hyperkkl_reading}
    # whole recordings: RMSE over t >= 15 min
    fullres = {}
    for key, d in full.items():
        sel = d["t_min"] >= 15.0
        e = lambda v: float(np.sqrt(np.mean(((v - d["soc"]) * 100)[sel] ** 2)))
        mx = lambda v: float(np.abs((v - d["soc"]) * 100)[sel].max())
        row = {m: {"rmse": e(d[m]), "max": mx(d[m])} for m in filters}
        for m in learned:
            row[m] = {"rmse": float(np.mean([e(d[f"{m}/{s}"]) for s in SEEDS])),
                      "max": float(np.mean([mx(d[f"{m}/{s}"]) for s in SEEDS])),
                      "per_seed": [e(d[f"{m}/{s}"]) for s in SEEDS]}
        row["hyperkkl"] = row["lora"]
        row["ws_best"] = {"rmse": float(np.mean([e(d[f"{ws_choice[s]}/{s}"]) for s in SEEDS])),
                          "max": float(np.mean([mx(d[f"{ws_choice[s]}/{s}"]) for s in SEEDS]))}
        fullres[key] = row
    train_info = {}
    for s in SEEDS:
        mm = torch.load(CACHE / f"models_seed{s}.pt", weights_only=False, map_location="cpu")
        train_info[s] = {"phase1_diag": mm["base"]["diag"],
                         **{m: {k: mm["trained"][m].get(k) for k in ("n_params", "width", "val", "wall_s")}
                            for m in METHODS}}
    res = {"design": "experiments/real/DESIGN.md", "data": "experiments/real/DATA.md",
           "identification": {k: prep["identification"][k] for k in ("params", "ecm", "thermal", "ocv_fit")},
           "Q_Ah": prep["Q"], "thermocouple_offset_K": prep["offset_K"],
           "arrhenius_factor_0C": prep["arrhenius_factor_0C"], "open_loop_model_fit": prep["model_fit"],
           "scales": prep["scales"], "n_episodes": {k: int(len(v)) for k, v in truth.items()},
           "n_windows": {k: int(len(np.unique(meta[k]["start"]))) for k in truth},
           "filter_tuning": {k: json.loads((CACHE / f"tune_{k}.json").read_text()) for k in ("ekf", "ukf")},
           "tables": tables, "paired": tests, "decision": decision, "whole_recordings": fullres,
           "train": train_info}
    if data_bl:
        res["data_baselines"] = {}
        for sd in SEEDS:
            bb = torch.load(CACHE / f"baselines_seed{sd}.pt", weights_only=False, map_location="cpu")
            res["data_baselines"][sd] = {"tuning": {k: (v.get("choice", v) if isinstance(v, dict) else v)
                                                    for k, v in bb["tuning"].items()},
                                         "info": {k: {kk: vv for kk, vv in v.items() if kk != "hist"}
                                                  for k, v in bb["info"].items()}}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    figures(prep, P, pred, truth, meta, tables, ws_choice, full, data_bl)
    log("H1: " + json.dumps(decision["H1_ws_best_vs_ekf"]) + "  H2: " + json.dumps(decision["H2_ws_best_vs_latent"]))
    log("hyperkkl (lora r=4): " + json.dumps({k: v for k, v in hyperkkl_reading.items()
                                               if k.startswith("H")}))
    if addA:
        log("Addendum A: " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_cycle"} for k, v in addA.items()}))
    return res


# Same palette as analysis/figures.py (Okabe-Ito, one colour per method in every figure of the paper).
FIG_COLOR = {"latent": "#E69F00", "concat": "#56B4E9", "film": "#009E73", "lora": "#0072B2", "hyper": "#D55E00",
             "ws_best": "#0072B2", "autonomous": "#999999", "nlox": "#CC79A7", "koopman_kf": "#AA4499",
             "koopman_kf_bilinear": "#AA4499", "greybox_ekf": "#999933", "greybox_ukf": "#999933",
             "greybox_ekf_oracle": "#999933", "ekf": "#000000", "ukf": "#777777", "cc": "#999999",
             "soc": "#BBBBBB"}
FIG_NAME = {"latent": "latent", "concat": "concat", "film": "FiLM", "lora": "HyperKKL", "hyper": "dense (abl.)",
            "ws_best": "WS-best", "autonomous": "autonomous", "nlox": "NLOX", "koopman_kf": "Koopman KF",
            "koopman_kf_bilinear": "Koopman KF (bil.)", "greybox_ekf": "grey-box EKF", "greybox_ukf": "grey-box UKF",
            "greybox_ekf_oracle": "grey-box EKF-or", "ekf": "EKF", "ukf": "UKF", "cc": "CC (wrong prior)"}
FIG_HATCH = {"ws_best": "////", "ukf": "xxxx", "greybox_ukf": "....", "koopman_kf_bilinear": "...."}
PANEL = (3.2, 2.6)          # each real-data panel, inches; included at 0.49\textwidth (= 3.19 in), scale ~1


def _style(plt):
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "hatch.linewidth": 0.5, "mathtext.default": "regular"})


def _save(fig, plt, name):
    fig.savefig(FIG / f"{name}.pdf")                 # vector, exact figsize: included without scaling
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)


def figures(prep, P, pred, truth, meta, tables, ws_choice, full, data_bl=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    col, nm = FIG_COLOR, FIG_NAME
    s0 = SEEDS[0]
    # (1) a held-out US06 window: the one whose start SOC is closest to 0.6, prior -20 %
    n = "test25/US06"
    m = meta[n]
    cand = np.flatnonzero(m["sgn"] < 0)
    i = cand[np.argmin(np.abs(m["soc0"][cand] - 0.6))]
    t = np.arange(truth[n].shape[1]) / 60.0
    fig, ax = plt.subplots(2, 1, figsize=PANEL, sharex=True, gridspec_kw={"height_ratios": [3, 1.1]},
                           layout="constrained")
    ax[0].plot(t, 100 * truth[n][i], color=col["soc"], lw=2.6, label="reference", zorder=1)
    ax[0].plot(t, 100 * pred[(n, "cc")][i], color=col["cc"], lw=0.9, ls=":", label=nm["cc"], zorder=2)
    ax[0].plot(t, 100 * pred[(n, "ekf")][i], color=col["ekf"], lw=1.0, ls="-", label=nm["ekf"], zorder=3)
    ax[0].plot(t, 100 * pred[(n, "ukf")][i], color=col["ukf"], lw=1.0, ls="--", label=nm["ukf"], zorder=3)
    ax[0].plot(t, 100 * pred[(n, "latent", s0)][i], color=col["latent"], lw=0.9, label=nm["latent"], zorder=4)
    ax[0].plot(t, 100 * pred[(n, ws_choice[s0], s0)][i], color=col["ws_best"], lw=0.9, label=nm["ws_best"], zorder=5)
    for b in ("greybox_ekf", "koopman_kf"):
        if b in data_bl:
            ax[0].plot(t, 100 * pred[(n, b, s0)][i], color=col[b], lw=0.9, ls="-.", label=nm[b], zorder=4)
    ax[0].axvline(SETTLE / 60.0, color="k", lw=0.5, ls=(0, (1, 2)))
    ax[0].set_ylabel("SOC (%)")
    fig.legend(loc="outside upper center", ncol=3, handlelength=1.6, handletextpad=0.4, columnspacing=0.9)
    rec = prep["recs"]["25/US06"]
    st = int(m["start"][i])
    ax[1].plot(t, rec.u[st:st + len(t)], color="#444444", lw=0.5)
    ax[1].axvline(SETTLE / 60.0, color="k", lw=0.5, ls=(0, (1, 2)))
    ax[1].set_ylabel("current (A)")
    ax[1].set_xlabel("time in window (min)")
    fig.align_ylabels(ax)
    _save(fig, plt, "us06_window")
    # (1b) the whole US06 recording from full charge, prior 80 %
    d = full["25/US06"]
    fig, ax = plt.subplots(figsize=(6.5, 2.4), layout="constrained")
    ax.plot(d["t_min"], 100 * d["soc"], color=col["soc"], lw=2.6, label="reference")
    for k, ls in (("cc", ":"), ("ekf", "-"), ("ukf", "--")):
        ax.plot(d["t_min"], 100 * d[k], color=col[k], lw=1.0, ls=ls, label=nm[k])
    ax.plot(d["t_min"], 100 * d[f"latent/{s0}"], color=col["latent"], lw=0.9, label=nm["latent"])
    ax.plot(d["t_min"], 100 * d[f"{ws_choice[s0]}/{s0}"], color=col["ws_best"], lw=0.9, label=nm["ws_best"])
    ax.set_xlabel("time (min)"); ax.set_ylabel("SOC (%)")
    fig.legend(loc="outside upper center", ncol=6)
    _save(fig, plt, "us06_full")
    # (2) settled RMSE per method per test cycle (learned: mean over seeds, bars = min..max)
    meths = ["latent", "concat", "lora", "hyper", "ws_best", "ekf", "ukf"] + list(data_bl)   # autonomous (15-35 %) in the tables only
    for tag in ("test25", "test0"):
        fams = list(tables[tag])
        fig, ax = plt.subplots(figsize=PANEL if len(fams) <= 3 else (6.5, 2.6), layout="constrained")
        w = 0.84 / len(meths)
        for j, mm in enumerate(meths):
            xs = np.arange(len(fams)) + (j - (len(meths) - 1) / 2) * w
            mean = [tables[tag][f][mm]["rmse_settled"]["mean"] for f in fams]
            if tables[tag][fams[0]][mm]["rmse_settled"].get("per_seed"):
                ps = np.array([tables[tag][f][mm]["rmse_settled"]["per_seed"] for f in fams])
                err = np.stack([np.array(mean) - ps.min(1), ps.max(1) - np.array(mean)])
            else:
                err = None
            h = FIG_HATCH.get(mm)
            fc = "white" if mm == "ukf" else col[mm]
            ax.bar(xs, mean, w, yerr=err, color=fc, hatch=h, label=nm[mm],
                   edgecolor="white" if (h and mm != "ukf") else "k", lw=0.3 if not h else 0.0,
                   error_kw={"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6, "ecolor": "#333333"})
            if h:   # outline drawn separately so the hatch can be white on the filled bars
                ax.bar(xs, mean, w, fill=False, edgecolor="k", lw=0.3)
        ax.set_xticks(np.arange(len(fams))); ax.set_xticklabels(fams)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("settled SOC RMSE (%)")
        fig.legend(loc="outside upper center", ncol=4 if len(fams) <= 3 else 9, handlelength=1.2,
                  handletextpad=0.4, columnspacing=0.8)
        _save(fig, plt, f"rmse_{tag}")


def replot(log=print):
    """Redraw the figures from the cached predictions and the existing results.json, without
    recomputing statistics or rewriting results.json (presentation only)."""
    prep = torch.load(CACHE / "prep.pt", weights_only=False)
    P = torch.load(CACHE / "predictions.pt", weights_only=False)
    pred, truth, meta, full = P["pred"], P["truth"], P["meta"], P["full"]
    res = json.loads((OUT / "results.json").read_text())
    tables = res["tables"]
    ws_choice = {int(k): v for k, v in res["decision"]["ws_choice_per_seed"].items()}
    known = set(tables["test25"][next(iter(tables["test25"]))])
    data_bl = []
    pbs = [CACHE / f"predictions_baselines_seed{sd}.pt" for sd in SEEDS]
    if all(p_.exists() for p_ in pbs):
        from .realdata_baselines import DATA_BASELINES
        data_bl = [b for b in DATA_BASELINES if b in known]
        for p_ in pbs:
            B = torch.load(p_, weights_only=False)
            pred.update(B["pred"])
            for k, d in B["full"].items():
                full[k].update(d)
    for name in truth:
        for s in SEEDS:
            pred[(name, "ws_best", s)] = pred[(name, ws_choice[s], s)]
    figures(prep, P, pred, truth, meta, tables, ws_choice, full, data_bl)
    log(f"replotted {FIG}")


if __name__ == "__main__":
    replot()
