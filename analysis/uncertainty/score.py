"""Stage 3: mechanical scoring of the uncertainty study against PREREG.md -> experiments/uncertainty/RESULTS.md
(+ cells.csv, summary.json). usage: score.py RAWDIR OUTDIR"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
raw, outd = Path(sys.argv[1]), Path(sys.argv[2])
outd.mkdir(parents=True, exist_ok=True)
DEV = ["battery", "bioreactor", "lorenz", "quad_tank", "vdp"]
TEST = ["cartpole", "fhn", "highway_traffic", "network", "rossler", "two_link_arm"]
NODES = ["C1-where", "C3-partial", "TEST-C1", "TEST-C3"]
TIERS = ["id", "shape_ood", "sequence_ood"]
CONDS = ["clean", "outliers", "noise10", "plant_drift10"]
LEARNED = ["U1", "U2", "U2mf", "U3", "U3oracle", "U4", "U4c"]
FILT = ["ekf", "ukf", "ekf_oracle", "ekf-c", "ukf-c", "ekf_oracle-c"]

rows, checks = [], {}
for f in sorted(raw.glob("*.json")):
    d = json.loads(f.read_text())
    rows += d["rows"]
    checks[f.stem] = d["extra"]["checks"]
df = pd.DataFrame(rows)
df["seed"] = df["seed"].astype(str)
df.to_csv(outd / "cells.csv", index=False)
KEYS = ["node", "system", "tier", "cond", "method", "state", "measured"]
num = ["coverage", "width", "width_med", "rmse", "IS", "cov_hi", "rho_w_e", "cap_frac", "anees", "mnees", "nrmse_centre", "ep_cov_min"]
num = [c for c in num if c in df]
sm = df.groupby(KEYS, as_index=False)[num].mean()           # seed mean (U4: the single ensemble row)
st = sm.groupby(["node", "system", "tier", "cond", "method"], as_index=False)[num].mean()   # state mean

L = []
P = L.append


def fmt(x, n=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def cell(node, system, tier, cond, method, col="coverage", table=st):
    r = table[(table.node == node) & (table.system == system) & (table.tier == tier) & (table.cond == cond) & (table.method == method)]
    return float(r[col].iloc[0]) if len(r) else np.nan


# ---------------- information floor (experiments/information/budgets.json, full-past Bayesian PCRB)
floor = {}
bp = ROOT / "experiments" / "information" / "budgets.json"
if bp.exists():
    B = json.loads(bp.read_text())
    for s, v in B.items():
        try:
            floor[s] = v["dists"]["eval_id"]["bayes"]["full"]["crb_state"]
        except (KeyError, TypeError):
            pass
floor_src = f"experiments/information/budgets.json (dists.eval_id.bayes.full.crb_state, read {time.strftime('%H:%M')})" if floor else "none"

# ================================================================= criteria
verdict = {}
# P1 validity on id/clean (dev, C1-where and C3-partial)
p1 = {}
for m in ("U1", "U2", "U2mf"):
    sub = sm[(sm.node.isin(["C1-where", "C3-partial"])) & (sm.system.isin(DEV)) & (sm.tier == "id") & (sm.cond == "clean") & (sm.method == m)]
    mn, mean = float(sub.coverage.min()), float(sub.coverage.mean())
    ok = mn >= 0.87 and 0.88 <= mean <= 0.95
    p1[m] = {"min": mn, "mean": mean, "n": len(sub), "pass": ok,
             "worst": sub.sort_values("coverage").head(3)[["node", "system", "state", "coverage"]].to_dict("records")}
verdict["P1"] = all(v["pass"] for v in p1.values())
p1t = {}
for m in ("U1", "U2", "U2mf"):
    sub = sm[(sm.node.isin(["TEST-C1", "TEST-C3"])) & (sm.tier == "id") & (sm.cond == "clean") & (sm.method == m)]
    if len(sub):
        p1t[m] = {"min": float(sub.coverage.min()), "mean": float(sub.coverage.mean()), "n": len(sub),
                  "worst": sub.sort_values("coverage").head(3)[["node", "system", "state", "coverage"]].to_dict("records")}

# P2 adaptivity
p2 = {}
for node in ("C1-where", "C3-partial"):
    for tier in TIERS:
        wins = [s for s in DEV if cell(node, s, tier, "clean", "U2", "IS") < cell(node, s, tier, "clean", "U1", "IS")]
        p2[(node, tier, "IS")] = wins
    wins = [s for s in DEV if cell(node, s, "id", "clean", "U2", "cov_hi") > cell(node, s, "id", "clean", "U1", "cov_hi")]
    p2[(node, "id", "cov_hi")] = wins
verdict["P2"] = all(len(v) >= 4 for v in p2.values())

# P3 observability awareness
p3a = {}
for s in DEV:
    sub = sm[(sm.node == "C1-where") & (sm.system == s) & (sm.tier == "id") & (sm.cond == "clean") & (sm.method == "U2")]
    if (~sub.measured).any() and sub.measured.any():
        p3a[s] = (float(sub[~sub.measured].width.mean()), float(sub[sub.measured].width.mean()))
p3a_n = sum(1 for u, m in p3a.values() if u > m)
p3b = {}
for node in ("C1-where", "C3-partial"):
    sub = sm[(sm.node == node) & (sm.system == "quad_tank") & (sm.tier == "id") & (sm.cond == "clean") & (sm.method == "U2")].set_index("state")
    if len(sub):
        w = sub.width
        p3b[node] = {"widths": [float(w[j]) for j in range(4)], "pass": bool(min(w[2], w[3]) > max(w[0], w[1]))}
# per-state lora RMSE (normalised), from the U1 rows
from scipy.stats import spearmanr
rm = {f"{r.node}|{r.system}|{int(r.state)}": float(r.rmse) for _, r in
      sm[(sm.tier == "id") & (sm.cond == "clean") & (sm.method == "U1")].iterrows()}
pairs = []
for node in ("C1-where", "C3-partial"):
    for s in DEV:
        sub = sm[(sm.node == node) & (sm.system == s) & (sm.tier == "id") & (sm.cond == "clean") & (sm.method == "U2")]
        for _, r in sub.iterrows():
            k = f"{node}|{s}|{int(r.state)}"
            if k in rm:
                pairs.append((r.width, rm[k]))
rho3 = float(spearmanr([a for a, _ in pairs], [b for _, b in pairs]).correlation) if len(pairs) > 3 else np.nan
verdict["P3"] = p3a_n >= min(4, len(p3a)) and all(v["pass"] for v in p3b.values()) and rho3 >= 0.6

# P4 shift
fail_cells, restored, oracle_ok = [], [], []
for node in NODES:
    systems = DEV if node in ("C1-where", "C3-partial") else TEST
    for s in systems:
        for tier in TIERS:
            for cond in CONDS:
                if tier == "id" and cond == "clean":
                    continue
                c2 = cell(node, s, tier, cond, "U2")
                if not np.isfinite(c2):
                    continue
                c3o = cell(node, s, tier, cond, "U3oracle")
                oracle_ok.append(abs(c3o - 0.9) <= 0.02)
                if c2 < 0.85:
                    c3 = cell(node, s, tier, cond, "U3")
                    wr = cell(node, s, tier, cond, "U3", "width") / cell(node, s, tier, cond, "U2", "width")
                    fail_cells.append((node, s, tier, cond, c2, c3, wr))
                    restored.append(c3 >= 0.85 and wr <= 3)
n_shift = len(oracle_ok)
p4 = {"n_shift_cells": n_shift, "n_fail": len(fail_cells), "n_restored": int(sum(restored)),
      "aci_restores": bool(len(fail_cells) and sum(restored) >= 0.5 * len(fail_cells)),
      "oracle_within_002": float(np.mean(oracle_ok)) if oracle_ok else np.nan}
verdict["P4_aci_restores"] = p4["aci_restores"]
verdict["P4_oracle"] = p4["oracle_within_002"] >= 0.95

# P5 filters
p5 = {}
for node in NODES:
    systems = DEV if node in ("C1-where", "C3-partial") else TEST
    for tier in TIERS:
        for cond in CONDS:
            for fn in ("ekf", "ukf", "ekf_oracle"):
                for s in systems:
                    c, a = cell(node, s, tier, cond, fn), cell(node, s, tier, cond, fn, "anees")
                    if not np.isfinite(c):
                        continue
                    calib = 0.85 <= c <= 0.95 and 0.5 <= a <= 2
                    cu = cell(node, s, tier, cond, "U2")
                    p5.setdefault((node, tier, cond, fn), []).append((s, c, a, calib, abs(cu - 0.9) < abs(c - 0.9), cu))

# P6 floor
p6 = []
for node in ("C1-where", "C3-partial", "TEST-C1", "TEST-C3"):
    for m in ("U1", "U2"):
        sub = sm[(sm.node == node) & (sm.tier == "id") & (sm.cond == "clean") & (sm.method == m)]
        for _, r in sub.iterrows():
            if r.system in floor:
                fl = floor[r.system][int(r.state)]
                hw = r.width / 2
                p6.append((node, r.system, int(r.state), m, hw, fl, hw / (1.645 * fl), r.coverage))
p6_viol = [x for x in p6 if x[6] < 1 and x[7] >= 0.9]
verdict["P6"] = len(p6_viol) == 0 if p6 else None

# ================================================================= report
P(f"# Uncertainty bands for HyperKKL (`lora`): results\n")
P(f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `analysis/uncertainty/score.py` from the stage-2 outputs in "
  f"`experiments/uncertainty/raw/` (rules: `PREREG.md`, theory: `THEORY.md`). Nothing hand-edited. Seeds 0-2 per cell; "
  f"`U4`/`U4c` are one ensemble of the 3 seeds. Widths are full widths `2w/sigma_j`; coverage target 0.90.\n")
P("## Verdicts (pre-registered criteria)\n")
P("| criterion | verdict | key numbers |\n|---|---|---|")
P(f"| P1 validity on id/clean (dev) | **{'PASS' if verdict['P1'] else 'FAIL'}** | " + "; ".join(
    f"{m}: mean {v['mean']:.3f}, min {v['min']:.3f} (n={v['n']})" for m, v in p1.items()) + " |")
P(f"| P2 adaptivity (U2 < U1 interval score; U2 > U1 high-error coverage, >= 4/5) | **{'PASS' if verdict['P2'] else 'FAIL'}** | " + "; ".join(
    f"{n} {t} {k}: {len(v)}/5" for (n, t, k), v in p2.items()) + " |")
P(f"| P3 observability awareness | **{'PASS' if verdict['P3'] else 'FAIL'}** | (a) unmeasured > measured U2 width on {p3a_n}/{len(p3a)}; "
  f"(b) quad_tank upper > lower: " + ", ".join(f"{k} {'yes' if v['pass'] else 'no'}" for k, v in p3b.items()) + f"; (c) Spearman(width, RMSE) = {fmt(rho3, 2)} |")
P(f"| P4 shift: ACI (deployable) restores failing cells | **{'PASS' if verdict['P4_aci_restores'] else 'FAIL'}** | U2 fails (< 0.85) in {p4['n_fail']}/{n_shift} shift cells; U3 restores {p4['n_restored']} |")
P(f"| P4 U3-oracle long-run coverage within 0.02 | **{'PASS' if verdict['P4_oracle'] else 'FAIL'}** | {100 * p4['oracle_within_002']:.0f}% of cells (need 95%) |")
P(f"| P6 no band narrower than the information floor with coverage >= 0.9 | **{'PASS' if verdict['P6'] else ('n/a' if verdict['P6'] is None else 'FAIL')}** | {len(p6_viol)} violations in {len(p6)} (node, system, state, method) cells; floor: {floor_src} |")
P("")
P("P5 (filters) is a count, not pass/fail, and is in Section 5.\n")

# ---- Section 1: id/clean coverage table
P("## 1. Validity on id/clean (conformal guarantee regime)\n")
P("Coverage averaged over states (min over states in parentheses), seed mean.\n")
meth1 = ["U1", "U2", "U2mf", "U3", "U3oracle", "U4", "U4c", "ekf", "ukf", "ekf_oracle", "ekf-c"]
for node in NODES:
    systems = DEV if node in ("C1-where", "C3-partial") else TEST
    P(f"**{node}**\n")
    P("| system | " + " | ".join(meth1) + " |\n|---|" + "---|" * len(meth1))
    for s in systems:
        cells = []
        for m in meth1:
            sub = sm[(sm.node == node) & (sm.system == s) & (sm.tier == "id") & (sm.cond == "clean") & (sm.method == m)]
            cells.append(f"{sub.coverage.mean():.3f} ({sub.coverage.min():.2f})" if len(sub) else "-")
        P(f"| {s} | " + " | ".join(cells) + " |")
    P("")
if p1t:
    P("Held-out test plants, id/clean: " + "; ".join(f"{m}: mean {v['mean']:.3f}, min {v['min']:.3f} (n={v['n']}), worst "
      + ", ".join(f"{w['node']}/{w['system']}/x{w['state']}={w['coverage']:.3f}" for w in v["worst"]) for m, v in p1t.items()) + "\n")
P("Worst dev cells (P1): " + "; ".join(f"{m}: " + ", ".join(f"{w['node']}/{w['system']}/x{w['state']}={w['coverage']:.3f}" for w in v["worst"]) for m, v in p1.items()) + "\n")

# ---- Section 2: per-state widths, observability
P("## 2. Per-state bands on id/clean: observability awareness and the information floor\n")
P("Full width `2w/sigma` (seed mean). `floor` = `2 x 1.645 x` the full-past Bayesian PCRB std from the information study "
  "(the narrowest 90% band any estimator could have with Gaussian posteriors; THEORY.md section 3). "
  "`rmse` = the lora settled RMSE of that state (normalised). m = measured.\n")
for node in NODES:
    systems = DEV if node in ("C1-where", "C3-partial") else TEST
    P(f"**{node}**\n")
    P("| system | state | m | rmse | floor | U1 width | U2 width (cov) | U2 width / floor | EKF width (cov) | U2 rho(width, abs err) |\n|---|---|---|---|---|---|---|---|---|---|")
    for s in systems:
        sub = sm[(sm.node == node) & (sm.system == s) & (sm.tier == "id") & (sm.cond == "clean")]
        for j in sorted(sub.state.unique()):
            g = lambda m, c: sub[(sub.method == m) & (sub.state == j)][c].mean()
            fl = 2 * 1.645 * floor[s][j] if s in floor else np.nan
            P(f"| {s} | x{j} | {'y' if sub[sub.state == j].measured.iloc[0] else ''} | {fmt(rm.get(f'{node}|{s}|{j}', np.nan))} | {fmt(fl)} | {fmt(g('U1', 'width'))} | "
              f"{fmt(g('U2', 'width'))} ({fmt(g('U2', 'coverage'), 2)}) | {fmt(g('U2', 'width') / fl, 1)} | {fmt(g('ekf', 'width'))} ({fmt(g('ekf', 'coverage'), 2)}) | {fmt(g('U2', 'rho_w_e'), 2)} |")
    P("")

# ---- Section 3: shift
P("## 3. Coverage under shift and faults (state mean, averaged over systems; seed mean)\n")
meth3 = ["U1", "U2", "U3", "U3oracle", "U3qt*", "U3qt_oracle*", "U4c", "ekf", "ukf", "ekf_oracle", "ekf-c"]
for grp, nodes, systems in (("dev (C1-where + C3-partial)", ["C1-where", "C3-partial"], DEV), ("test (TEST-C1 + TEST-C3)", ["TEST-C1", "TEST-C3"], TEST)):
    P(f"**{grp}** - coverage; width of U2 / U3 in brackets\n")
    P("| tier | cond | " + " | ".join(meth3) + " | U2 width | U3 width (frac. infinite) | U3qt* width |\n|---|---|" + "---|" * (len(meth3) + 3))
    for tier in TIERS:
        for cond in CONDS:
            sub = st[(st.node.isin(nodes)) & (st.system.isin(systems)) & (st.tier == tier) & (st.cond == cond)]
            if not len(sub):
                continue
            g = lambda m, c="coverage": sub[sub.method == m.rstrip("*")][c].mean()
            P(f"| {tier} | {cond} | " + " | ".join(fmt(g(m)) for m in meth3) + f" | {fmt(g('U2', 'width'))} | {fmt(g('U3', 'width'))} ({fmt(g('U3', 'cap_frac'), 2)} inf) | {fmt(g('U3qt', 'width'))} |")
    P("")
P("`*` = exploratory (PREREG amendment A3, not part of any verdict). Widths of U3 / U3-oracle include infinite steps capped at 3x the largest U2 band.\n")
P(f"**Failing cells** (U2 coverage < 0.85; {len(fail_cells)} of {n_shift} shift cells), with the deployable ACI (U3) and, exploratory, U3qt:\n")
P("| node | system | tier | cond | U2 cov | U3 cov | U3/U2 width | U3qt* cov | U3qt*/U2 width | U3-oracle cov | ekf_oracle cov |\n|---|---|---|---|---|---|---|---|---|---|---|")
qt_rest = 0
for c in sorted(fail_cells, key=lambda c: c[4]):
    cq = cell(c[0], c[1], c[2], c[3], "U3qt"); wq = cell(c[0], c[1], c[2], c[3], "U3qt", "width") / cell(c[0], c[1], c[2], c[3], "U2", "width")
    qt_rest += int(cq >= 0.85 and wq <= 3)
    P(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]:.3f} | {c[5]:.3f} | {c[6]:.2f} | {cq:.3f} | {wq:.2f} | {cell(c[0], c[1], c[2], c[3], 'U3oracle'):.3f} | {fmt(cell(c[0], c[1], c[2], c[3], 'ekf_oracle'))} |")
P("")
P(f"Exploratory: U3qt brings {qt_rest}/{len(fail_cells)} failing cells to >= 0.85 within 3x the U2 width (U3: {p4['n_restored']}).\n")
p4["qt_restored_exploratory"] = qt_rest
P("")

# ---- Section 4: efficiency
P("## 4. Efficiency and conditional coverage (clean; state mean, mean over systems)\n")
P("| node | tier | method | coverage | width | median width | interval score | coverage in top-20% error episodes | rho(width, abs err) |\n|---|---|---|---|---|---|---|---|---|")
for node in NODES:
    systems = DEV if node in ("C1-where", "C3-partial") else TEST
    for tier in TIERS:
        sub = st[(st.node == node) & (st.system.isin(systems)) & (st.tier == tier) & (st.cond == "clean")]
        for m in ["U1", "U2", "U2mf", "U3", "U3qt", "U4", "U4c", "ekf", "ekf-c"]:
            q = sub[sub.method == m]
            if len(q):
                P(f"| {node} | {tier} | {m} | {q.coverage.mean():.3f} | {q.width.mean():.4f} | {q.width_med.mean():.4f} | {q.IS.mean():.4f} | {q.cov_hi.mean():.3f} | {fmt(q.rho_w_e.mean(), 2)} |")
P("")
P("P2 detail (systems where U2 beats U1): " + "; ".join(f"{n} {t} {k}: {', '.join(v) or 'none'}" for (n, t, k), v in p2.items()) + "\n")

# ---- Section 5: filters
P("## 5. EKF / UKF covariance calibration (P5)\n")
P("Per cell: coverage of the Gaussian 90% band `1.645 sqrt(P_jj)` (state mean), ANEES = mean NEES / n_x (1 = calibrated). "
  "`calibrated` = coverage in [0.85, 0.95] and ANEES in [0.5, 2]. `U2 closer` = number of systems where HyperKKL-U2's coverage is closer to 0.90.\n")
P("| node | tier | cond | filter | calibrated systems | U2 closer to 0.90 | median coverage | median ANEES | median U2 coverage |\n|---|---|---|---|---|---|---|---|---|")
p5sum = {}
for (node, tier, cond, fn), v in p5.items():
    ncal, ncl = sum(x[3] for x in v), sum(x[4] for x in v)
    p5sum[f"{node}|{tier}|{cond}|{fn}"] = {"calibrated": ncal, "u2_closer": ncl, "n": len(v)}
    P(f"| {node} | {tier} | {cond} | {fn} | {ncal}/{len(v)} | {ncl}/{len(v)} | {np.median([x[1] for x in v]):.3f} | {np.median([x[2] for x in v]):.2g} | {np.median([x[5] for x in v]):.3f} |")
P("")
tot = {fn: [0, 0, 0] for fn in ("ekf", "ukf", "ekf_oracle")}
for (node, tier, cond, fn), v in p5.items():
    tot[fn][0] += sum(x[3] for x in v); tot[fn][1] += sum(x[4] for x in v); tot[fn][2] += len(v)
P("Totals over all cells: " + "; ".join(f"{fn}: calibrated {a}/{n}, U2 closer to 0.90 in {b}/{n}" for fn, (a, b, n) in tot.items()) + "\n")

# ---- Section 6: P6 detail
P("## 6. Information-floor sanity check (P6)\n")
if p6:
    r = np.array([x[6] for x in p6])
    P(f"Ratio half-width / (1.645 x floor std) over {len(p6)} cells: min {r.min():.2f}, median {np.median(r):.2f}, max {r.max():.2f}. "
      f"Cells below 1 (band narrower than the floor's 90% interval): {int((r < 1).sum())}, of which with coverage >= 0.9: {len(p6_viol)}.\n")
    if (r < 1).any():
        P("| node | system | state | method | half-width | floor std | ratio | coverage |\n|---|---|---|---|---|---|---|---|")
        for x in sorted([x for x in p6 if x[6] < 1], key=lambda x: x[6]):
            P(f"| {x[0]} | {x[1]} | x{x[2]} | {x[3]} | {x[4]:.4f} | {x[5]:.4f} | {x[6]:.2f} | {x[7]:.3f} |")
        P("")
else:
    P("No information floor available.\n")

# ---- checks
P("## 7. Reproduction and data checks\n")
bad = []
for k, c in checks.items():
    for kk, v in c.items():
        if kk.endswith("nrmse_lora"):
            for t, (a, b) in v.items():
                if abs(a - b) / b > 0.01:
                    bad.append((k, kk, t, a, b))
P(f"Settled NRMSE of the re-run lora observer vs the run's stored results.json (clean, 3 tiers): "
  f"{sum(len(v) for c in checks.values() for kk, v in c.items() if kk.endswith('nrmse_lora'))} comparisons, "
  f"{len(bad)} differ by > 1%." + (" Differences: " + "; ".join(f"{a}/{b}/{t}: {x:.4f} vs {y:.4f}" for a, b, t, x, y in bad[:10]) if bad else "") + "\n")
P("Ensemble (U4) available (identical episodes across seeds): " + ", ".join(f"{k}: {c.get('ensemble_same_episodes')}" for k, c in checks.items()) + "\n")
(outd / "RESULTS.md").write_text("\n".join(L))
(outd / "summary.json").write_text(json.dumps({"verdict": verdict, "P1": p1, "P1_test": p1t,
    "P2": {"|".join(k): v for k, v in p2.items()}, "P3": {"a": p3a, "b": p3b, "rho": rho3}, "P4": p4,
    "P4_fail_cells": fail_cells, "P5": p5sum, "P6_violations": p6_viol, "floor_source": floor_src}, default=str, indent=1))
print("\n".join(L[:20]))
