"""Mechanical verdicts of PREREG.md section 5 -> experiments/structural/RESULTS_auto.md + scores.json.

    python analyze.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
from hyperkkl.metrics import paired_relative  # noqa: E402

EXP = HERE.parent
RUNS = EXP / "runs"
W = ["quad_tank", "battery", "cartpole", "highway_traffic"]
R = ["bioreactor", "vdp", "lorenz"]
SEEDS = [0, 1, 2]
TIERS = ["sequence_ood", "id", "shape_ood"]
GB3 = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
GB5 = GB3 + ["gbw_ekf", "gbw_ukf"]
METHODS = ["hyperkkl_A4", "hyperkkl_A0", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle", "gbw_ekf",
           "gbw_ukf", "ekf", "ukf", "climatology"]


def load():
    d = {}
    for f in RUNS.glob("*/*/seed*/results.json"):
        s, tier, sd = f.parts[-4], f.parts[-3], int(f.parts[-2][4:])
        d[(s, tier, sd)] = json.load(open(f))
    return d


def complete(d, s, tier):
    return all((s, tier, k) in d for k in SEEDS)


def per_seed(d, s, tier, m, et="sequence_ood", key="nrmse"):
    return [np.asarray(d[(s, tier, k)]["results"][et][m][key], float) for k in SEEDS]


def seedmean(d, s, tier, m, et="sequence_ood", key="nrmse"):
    v = [x.mean() for x in per_seed(d, s, tier, m, et, key)]
    return float(np.mean(v)), v


def best(d, s, tier, et, cands):
    return min(cands, key=lambda m: seedmean(d, s, tier, m, et)[0])


def cmp(d, s, tier, et, a, b):
    return paired_relative(per_seed(d, s, tier, a, et), per_seed(d, s, tier, b, et))


def welch(a, b, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if va + vb > 0 else 1e9
    h = student_t.ppf(1 - alpha / 2, df) * np.sqrt(va + vb)
    m = a.mean() - b.mean()
    return float(m), [float(m - h), float(m + h)]


def margins(d, s, tier, et, cands=GB3):
    """per-seed relative change of A4 vs the best grey-box (chosen on seed means)."""
    b = best(d, s, tier, et, cands)
    A, B = per_seed(d, s, tier, "hyperkkl_A4", et), per_seed(d, s, tier, b, et)
    return [float((a.mean() - bb.mean()) / bb.mean()) for a, bb in zip(A, B)], b


def main():
    d = load()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M %Z"), "n_jobs": len(d)}
    Wd = [s for s in W if complete(d, s, "ref") and complete(d, s, "struct")]
    Rd = [s for s in R if complete(d, s, "ref") and complete(d, s, "struct")]
    hw_ok = "highway_traffic" in Wd
    out["W_done"], out["R_done"] = Wd, Rd
    L = []
    L.append(f"# Structural model error: mechanical results\n\nGenerated {out['generated']} by `code/analyze.py` "
             f"from {len(d)} job results (rules: PREREG.md). Nothing hand-edited.\n")
    if len(d) < 42:
        L.append(f"**INTERIM: {len(d)}/42 jobs finished. Tables show every (plant, tier) with 3 seeds done; the "
                 f"verdict lines below are computed on incomplete data and are NOT the final verdicts.**\n")
    # ---- table
    for et in TIERS:
        L.append(f"\n## NRMSE, eval tier {et} (seed mean [min, max over seeds]; all visible states)\n")
        L.append("| plant | tier | " + " | ".join(METHODS) + " |")
        L.append("|---|---|" + "---|" * len(METHODS))
        for s in W + R:
            for tier in ("ref", "struct"):
                if not complete(d, s, tier):
                    continue
                cells = []
                bg = best(d, s, tier, et, GB3)
                for m in METHODS:
                    mu, v = seedmean(d, s, tier, m, et)
                    c = f"{mu:.4f} [{min(v):.4f}, {max(v):.4f}]"
                    cells.append(f"**{c}**" if m == bg else c)
                L.append(f"| {s} | {tier} | " + " | ".join(cells) + " |")
        L.append("\nBold = best of the three standard grey-box filters (the P-S1 comparator).")
    # ---- A4 vs best grey-box, all cells
    L.append("\n## HyperKKL A4 vs best grey-box (paired relative change, t-CI over 3 seeds; < 0 = HyperKKL better)\n")
    L.append("| plant | group | tier | eval | best GB (3 std) | A4 vs it | verdict | best of 5 incl. window | A4 vs it | verdict | A0 vs best GB(3) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    vs = {}
    for s in W + R:
        for tier in ("ref", "struct"):
            if not complete(d, s, tier):
                continue
            for et in TIERS:
                b3, b5 = best(d, s, tier, et, GB3), best(d, s, tier, et, GB5)
                c3, c5 = cmp(d, s, tier, et, "hyperkkl_A4", b3), cmp(d, s, tier, et, "hyperkkl_A4", b5)
                c0 = cmp(d, s, tier, et, "hyperkkl_A0", b3)
                vs[(s, tier, et)] = {"best3": b3, "c3": c3, "best5": b5, "c5": c5, "a0": c0}
                v = lambda c: "beats" if c["better"] else ("worse" if c["worse"] else "n.s.")
                f = lambda c: f"{100 * c['rel']:+.0f}% [{100 * c['ci'][0]:+.0f}, {100 * c['ci'][1]:+.0f}]"
                L.append(f"| {s} | {'W' if s in W else 'R'} | {tier} | {et} | {b3} | {f(c3)} | {v(c3)} | {b5} | "
                         f"{f(c5)} | {v(c5)} | {f(c0)} {v(c0)} |")
    out["vs"] = {"|".join(k): v for k, v in vs.items()}
    # ---- P-S1
    et = "sequence_ood"
    nW = len(Wd)
    a = [s for s in Wd if vs[(s, "struct", et)]["c3"]["better"]]
    b = [s for s in Wd if vs[(s, "ref", et)]["c3"]["worse"]]
    need = 3
    ps1 = len(a) >= need and len(b) >= need and nW >= 3
    out["P-S1"] = {"W": Wd, "struct_beats": a, "ref_worse": b, "need": need, "pass": ps1, "highway_included": hw_ok}
    a5 = [s for s in Wd if vs[(s, "struct", et)]["c5"]["better"]]
    L.append("\n## Pre-registered verdicts (sequence_ood)\n")
    L.append(f"* **P-S1: {'PASS' if ps1 else 'FAIL'}** - W = {Wd} ({'4 plants' if hw_ok else 'highway missing: of-3 variant'}). "
             f"(a) struct, A4 beats best grey-box on {len(a)}/{nW} {a} (need >= {need}); "
             f"(b) ref, A4 significantly worse on {len(b)}/{nW} {b} (need >= {need}). "
             f"[Reported: struct, A4 beats best of all 5 grey-box incl. window control on {len(a5)}/{nW} {a5}.]")
    # ---- P-S1, addendum 1 variant (b): cartpole excluded, of-3 rule
    Wb = [s for s in Wd if s != "cartpole"]
    ab = [s for s in Wb if vs[(s, "struct", et)]["c3"]["better"]]
    bb = [s for s in Wb if vs[(s, "ref", et)]["c3"]["worse"]]
    ps1b = len(Wb) == 3 and len(ab) >= 3 and len(bb) >= 3
    out["P-S1b"] = {"W": Wb, "struct_beats": ab, "ref_worse": bb, "need": 3, "pass": ps1b}
    L.append(f"* **P-S1 (b), cartpole excluded (addendum 1): {'PASS' if ps1b else 'FAIL'}** - W = {Wb}; "
             f"(a) struct beats on {len(ab)}/{len(Wb)} {ab}; (b) ref worse on {len(bb)}/{len(Wb)} {bb} (need 3 of 3).")
    # ---- P-S2
    rows, shrink = [], []
    for s in Rd:
        ms, bs = margins(d, s, "struct", et)
        mr, br = margins(d, s, "ref", et)
        diff, ci = welch(ms, mr)
        sig = ci[0] > 0
        if sig:
            shrink.append(s)
        rows.append({"plant": s, "m_struct": ms, "m_ref": mr, "diff": diff, "ci": ci, "shrinks": sig})
    ps2 = len(Rd) == 3 and not shrink
    out["P-S2"] = {"rows": rows, "pass": ps2}
    L.append(f"* **P-S2: {'PASS' if ps2 else 'FAIL'}** - rich plants {Rd}; margin m = rel. change A4 vs best grey-box "
             f"(seed values), m_struct - m_ref (Welch 95% CI): " + "; ".join(
                 f"{r['plant']}: m_ref {np.mean(r['m_ref']) * 100:+.0f}%, m_struct {np.mean(r['m_struct']) * 100:+.0f}%, "
                 f"diff {r['diff'] * 100:+.0f} pp [{r['ci'][0] * 100:+.0f}, {r['ci'][1] * 100:+.0f}]"
                 f"{' SHRINKS' if r['shrinks'] else ''}" for r in rows))
    # ---- P-S3
    pl = [s for s in W + R if complete(d, s, "ref") and complete(d, s, "struct")]
    D, NR, rho_r = [], [], []
    for s in pl:
        gs = seedmean(d, s, "struct", best(d, s, "struct", et, GB3), et)[0]
        gr = seedmean(d, s, "ref", best(d, s, "ref", et, GB3), et)[0]
        D.append((gs - gr) / gr)
        NR.append(1 - float(np.mean([d[(s, "struct", k)]["info"]["repairability"]["rho"] for k in SEEDS])))
        rho_r.append(float(np.mean([d[(s, "ref", k)]["info"]["repairability"]["rho"] for k in SEEDS])))
    if len(pl) >= 3:
        rs, p = spearmanr(D, NR)
        rs, p = float(rs), float(p)
    else:
        rs, p = float("nan"), float("nan")
    ps3 = len(pl) >= 6 and rs >= 0.6
    pb = [i for i, s in enumerate(pl) if s != "cartpole"]
    if len(pb) >= 3:
        rsb, pbv = spearmanr([D[i] for i in pb], [NR[i] for i in pb])
        rsb, pbv = float(rsb), float(pbv)
    else:
        rsb, pbv = float("nan"), float("nan")
    out["P-S3b"] = {"spearman": rsb, "p": pbv, "n": len(pb), "pass": len(pb) >= 6 and rsb >= 0.6}
    out["P-S3"] = {"plants": pl, "D": D, "nonrep": NR, "rho_ref": rho_r, "spearman": rs, "p": p, "pass": ps3}
    L.append(f"* **P-S3: {'PASS' if ps3 else 'FAIL'}** - Spearman(D, 1 - rho_struct) = {rs:.3f} (p = {p:.3g}, "
             f"n = {len(pl)}; need >= 0.6).")
    L.append(f"* **P-S3 (b), cartpole excluded (addendum 1): {'PASS' if out['P-S3b']['pass'] else 'FAIL'}** - "
             f"Spearman = {rsb:.3f} (p = {pbv:.3g}, n = {len(pb)}).\n")
    L.append("| plant | best GB ref | best GB struct | D (GB degradation) | rho_ref | rho_struct | 1 - rho_struct | A4 ref | A4 struct | A4 change |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(pl):
        gr = seedmean(d, s, "ref", best(d, s, "ref", et, GB3), et)[0]
        gs = seedmean(d, s, "struct", best(d, s, "struct", et, GB3), et)[0]
        ar, as_ = seedmean(d, s, "ref", "hyperkkl_A4", et)[0], seedmean(d, s, "struct", "hyperkkl_A4", et)[0]
        L.append(f"| {s} | {gr:.4f} | {gs:.4f} | {100 * D[i]:+.0f}% | {rho_r[i]:.3f} | {1 - NR[i]:.3f} | {NR[i]:.3f} | "
                 f"{ar:.4f} | {as_:.4f} | {100 * (as_ - ar) / ar:+.0f}% |")
    # ---- secondary: counts on other tiers
    L.append("\n## Secondary (no verdict): P-S1 counts on the other eval tiers\n")
    for et2 in ("id", "shape_ood"):
        a2 = [s for s in Wd if vs[(s, "struct", et2)]["c3"]["better"]]
        b2 = [s for s in Wd if vs[(s, "ref", et2)]["c3"]["worse"]]
        L.append(f"* {et2}: struct A4 beats best GB on {len(a2)}/{nW} {a2}; ref A4 worse on {len(b2)}/{nW} {b2}")
    # residual fits
    L.append("\n## Residual fits (validation R^2 per state, seed 0) and repairability\n")
    L.append("| plant | tier | greybox R^2 | window-greybox R^2 | rho (H-step output) |")
    L.append("|---|---|---|---|---|")
    for s in W + R:
        for tier in ("ref", "struct"):
            if (s, tier, 0) in d:
                inf = d[(s, tier, 0)]["info"]
                L.append(f"| {s} | {tier} | {[round(x, 3) for x in inf['greybox']['val_r2']]} | "
                         f"{[round(x, 3) for x in inf['gbw']['val_r2']]} | {inf['repairability']['rho']:.3f} |")
    (EXP / "RESULTS_auto.md").write_text("\n".join(L) + "\n")
    (EXP / "scores.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(L[-40:]))


if __name__ == "__main__":
    main()
