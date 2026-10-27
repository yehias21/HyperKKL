"""Mechanical verdicts for the archive-recipe test (STUDY.md section 6.4, pre-registered).

usage: python analyze_ar.py <SELECTION.json> <scratch tree dir> <live tree dir> <arms.yaml> <out.json> <out.md>
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
from hyperkkl.metrics import paired_relative  # noqa: E402  (the live, unmodified metrics module)

sel = json.load(open(sys.argv[1]))
TREE, LIVE = Path(sys.argv[2]), Path(sys.argv[3])
ARMS = yaml.safe_load(open(sys.argv[4]))
NODE = {"C3-partial": "AR-partial", "C3-mismatch30": "AR-mismatch30"}
TIERS = ["sequence_ood", "shape_ood", "id"]
PRIMARY = ["A7_best", "A8_best_noenc"]
COMPONENTS = ["A1_soap", "A2_adam1e-2", "A3_p1x3", "A4_gru", "A5_curv", "A6_enc"]
OURS = {"lora", "hyper", "latent", "concat", "film", "autonomous"}


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def ep(d, tier, m, key="nrmse"):
    try:
        v = d["results"][tier]["clean"]["methods"][m][key]
    except KeyError:
        return None
    return np.asarray([np.nan if a is None else a for a in v], float)


cells = []
for c in sel["selected"]:
    node, sysn = NODE[c["node"]], c["system"]
    runs = {}
    for nd in (node, node + "-x"):                 # -x: extra seeds 3-4, primary arms only
        for sd in sorted((TREE / nd / "runs" / sysn).glob("seed*")):
            d = load(sd / "results.json")
            if d is not None:
                runs[int(sd.name[4:])] = d
    cells.append({"scenario": c["node"], "node": node, "system": sysn, "role": c["role"], "runs": runs,
                  "sel_r": c["r"]})


def arm_eps(cell, arm, tier):
    out = {}
    for s, d in cell["runs"].items():
        v = ep(d, tier, ARMS[arm])
        if v is not None and np.isfinite(v).all():
            out[s] = v
    return out


def reference(cell, tier, seeds):
    """Better of default lora / hyper by seed-mean on the same episodes (favours the default)."""
    best = None
    for a in ("A0_lora", "A0_hyper"):
        e = arm_eps(cell, a, tier)
        if not all(s in e for s in seeds):
            continue
        m = float(np.mean([e[s].mean() for s in seeds]))
        if best is None or m < best[0]:
            best = (m, a, e)
    return best


def contrast(cell, arm, tier, ref_arm=None, alpha=0.05):
    ea = arm_eps(cell, arm, tier)
    seeds = sorted(ea)
    if ref_arm is None:
        r = reference(cell, tier, seeds)
        if r is None:
            return None
        _, ref_name, eb = r
    else:
        eb = arm_eps(cell, ref_arm, tier)
        ref_name = ref_arm
        seeds = [s for s in seeds if s in eb]
    seeds = [s for s in seeds if s in eb]
    if len(seeds) < 2:
        return None
    pr = paired_relative([ea[s] for s in seeds], [eb[s] for s in seeds], alpha=alpha)
    if not pr:
        return None
    lo, hi = pr["ci"]
    return {"ref": ref_name, "seeds": seeds, "point": pr["rel"], "lo": lo, "hi": hi,
            "method": pr["ci_method"], "arm_mean": float(np.mean([ea[s].mean() for s in seeds])),
            "ref_mean": float(np.mean([eb[s].mean() for s in seeds])),
            "favour": hi is not None and hi < 0, "against": lo is not None and lo > 0}


def verdict(rows, n_cells, need=5):
    out = {}
    for t in ("sequence_ood", "shape_ood"):
        f = sum(1 for r in rows[t] if r and r["favour"])
        a = sum(1 for r in rows[t] if r and r["against"])
        out[t] = {"favour": f, "against": a, "n": n_cells, "n_with_data": sum(1 for r in rows[t] if r)}
    if all(out[t]["favour"] >= need for t in out) and not any(out[t]["against"] for t in out):
        v = "SUPERIOR"
    elif all(out[t]["against"] >= need for t in out) and not any(out[t]["favour"] for t in out):
        v = "INFERIOR"
    else:
        v = "INCONCLUSIVE"
    if any(out[t]["n_with_data"] < n_cells for t in out):
        v += " (PARTIAL DATA)"
    return v, out


report = {"generated": __import__("time").strftime("%Y-%m-%d %H:%M"), "selection": sel.get("frozen_at"),
          "cells": [], "primary": {}, "soap_vs_lr": {}, "components": {}}
n = len(cells)
for arm in PRIMARY:
    rows = {t: [contrast(c, arm, t, alpha=0.025) for c in cells] for t in TIERS}
    v, cnt = verdict(rows, n)
    report["primary"][arm] = {"verdict": v, "counts": cnt, "rows": rows}
rows = {t: [contrast(c, "A1_soap", t, ref_arm="A2_adam1e-2", alpha=0.05) for c in cells] for t in TIERS}
v, cnt = verdict(rows, n)
report["soap_vs_lr"] = {"verdict": v, "counts": cnt, "rows": rows}
for arm in COMPONENTS:
    rows = {t: [contrast(c, arm, t, alpha=0.05) for c in cells] for t in TIERS}
    report["components"][arm] = {"counts": verdict(rows, n)[1], "rows": rows}

# context: best baseline per cell from the live C3 node (same episodes, seeds 0-2), absolute NRMSE
for c in cells:
    ctx = {}
    for t in TIERS:
        per = {}
        for sd in sorted((LIVE / c["scenario"] / "runs" / c["system"]).glob("seed*/results.json")):
            d = load(sd)
            if not d:
                continue
            for m, v in d["results"][t]["clean"]["methods"].items():
                if m in OURS:
                    continue
                x = [a for a in v["nrmse"] if a is not None and a == a]
                if x:
                    per.setdefault(m, []).append(float(np.mean(x)))
        if per:
            nmax = max(len(v) for v in per.values())
            full = {m: st.mean(v) for m, v in per.items() if len(v) == nmax}
            b = min(full, key=full.get)
            ctx[t] = {"best_baseline": b, "nrmse": full[b], "seeds": nmax}
        means = {}
        for arm in ARMS:
            e = arm_eps(c, arm, t)
            if e:
                means[arm] = float(np.mean([v.mean() for v in e.values()]))
        ctx[t + "_arm_means"] = means
    sc = {}
    for arm in ["A0_lora", "A0_hyper", "A7_best", "A8_best_noenc"]:
        vals = {}
        for s, d in c["runs"].items():
            try:
                b = d["results"]["sequence_ood"]["clean"]["scorecard"][ARMS[arm]]
            except KeyError:
                continue
            for k in ("nrmse_unmeasured", "nrmse_p99", "div_rate", "t_conv_median_s"):
                if b.get(k) is not None:
                    vals.setdefault(k, []).append(b[k])
        sc[arm] = {k: float(np.mean(v)) for k, v in vals.items()}
    p1 = {}
    for arm in ARMS:
        vals = {}
        for s, d in c["runs"].items():
            ti = d.get("train", {}).get(ARMS[arm], {})
            dg = ti.get("phase1_diag") or {}
            for k in ("eps_dec", "l_dec_q95"):
                if dg.get(k) is not None:
                    vals.setdefault(k, []).append(dg[k])
            if ti.get("n_params") is not None:
                vals.setdefault("n_params", []).append(ti["n_params"])
        p1[arm] = {k: float(np.mean(v)) for k, v in vals.items()}
    report["cells"].append({"scenario": c["scenario"], "system": c["system"], "role": c["role"],
                            "selection_r": c["sel_r"], "seeds": sorted(c["runs"]), "context": ctx,
                            "scorecard_seq": sc, "phase1_and_params": p1})

json.dump(report, open(sys.argv[5], "w"), indent=1, default=str)


def fmt(r):
    if not r:
        return "n/a"
    s = f"{100 * r['point']:+.0f}% [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]"
    return f"**{s}**" if r["favour"] else (f"*{s}*" if r["against"] else s)


L = [f"# Archive-recipe test: mechanical verdicts\n", f"Generated {report['generated']} Dubai from `analyze_ar.py` "
     f"(pre-registered rule: STUDY.md section 6.4). Selection frozen at {report['selection']}.\n",
     "Cell = per-seed paired relative change of the arm's mean episode NRMSE vs the better of default "
     "lora / hyper in that cell (Student-t CI over seeds; **bold** = CI excludes 0 in favour, *italic* = against). "
     "Primary arms use 97.5% CIs (Bonferroni over A7, A8); others 95%.\n"]
labels = [f"{c['system']}/{c['scenario'].replace('C3-', '')} ({c['role']})" for c in cells]
for arm in PRIMARY:
    P = report["primary"][arm]
    L.append(f"\n## {arm}: **{P['verdict']}**\n")
    L.append("counts: " + "; ".join(f"{t}: favour {v['favour']}/{v['n']}, against {v['against']}/{v['n']}"
                                    for t, v in P["counts"].items()) + "\n")
    L.append("| cell | " + " | ".join(TIERS) + " |\n|---|" + "---|" * len(TIERS))
    for i, lab in enumerate(labels):
        L.append(f"| {lab} | " + " | ".join(fmt(P["rows"][t][i]) for t in TIERS) + " |")
S = report["soap_vs_lr"]
L.append(f"\n## A1 (SOAP lr 1e-2) vs A2 (Adam lr 1e-2): **{S['verdict']}**\n")
L.append("| cell | " + " | ".join(TIERS) + " |\n|---|" + "---|" * len(TIERS))
for i, lab in enumerate(labels):
    L.append(f"| {lab} | " + " | ".join(fmt(S["rows"][t][i]) for t in TIERS) + " |")
L.append("\n## Components vs the default (95% CI, no verdict)\n")
for t in ("sequence_ood", "shape_ood"):
    L.append(f"\n### {t}\n\n| arm | " + " | ".join(labels) + " | favour / against |\n|---|" + "---|" * (len(labels) + 1))
    for arm in COMPONENTS + PRIMARY:
        rows = (report["components"].get(arm) or report["primary"][arm])["rows"][t]
        cnt = (report["components"].get(arm) or report["primary"][arm])["counts"][t]
        L.append(f"| {arm} | " + " | ".join(fmt(r) for r in rows) + f" | {cnt['favour']} / {cnt['against']} |")
L.append("\n## Absolute NRMSE (seed means) and context\n")
for c in report["cells"]:
    L.append(f"\n### {c['system']} / {c['scenario']} ({c['role']}, selection r = {c['selection_r']:.2f}, seeds {c['seeds']})\n")
    L.append("| arm | " + " | ".join(TIERS) + " | eps_dec | l_dec_q95 | params |\n|---|" + "---|" * (len(TIERS) + 3))
    for arm in ARMS:
        vals = [c["context"].get(t + "_arm_means", {}).get(arm) for t in TIERS]
        p1 = c["phase1_and_params"].get(arm, {})
        L.append(f"| {arm} | " + " | ".join("-" if v is None else f"{v:.4g}" for v in vals)
                 + f" | {p1.get('eps_dec', float('nan')):.3g} | {p1.get('l_dec_q95', float('nan')):.3g} | {p1.get('n_params', float('nan')):.0f} |")
    L.append("| best baseline (live C3 node, seeds 0-2) | " + " | ".join(
        f"{c['context'][t]['best_baseline']} {c['context'][t]['nrmse']:.4g}" if t in c["context"] else "-" for t in TIERS) + " | | | |")
    L.append("\nScorecard essentials (sequence_ood, seed means): " + "; ".join(
        f"{a}: " + ", ".join(f"{k} {v:.3g}" for k, v in d.items()) for a, d in c["scorecard_seq"].items() if d))
open(sys.argv[6], "w").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
