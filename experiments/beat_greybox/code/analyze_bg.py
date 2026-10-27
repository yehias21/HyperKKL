"""Mechanical analysis of the beat-the-grey-box campaign (rules: experiments/beat_greybox/PREREG.md).

    python analyze_bg.py [--arms A0,A1,...] [--out results.json] [--md tables.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
from hyperkkl.metrics import paired_relative  # noqa: E402

LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
NODES = ["C3-partial", "C3-mismatch30"]
LOSE = ["battery", "duffing", "quad_tank"]
WIN = ["bioreactor", "lorenz", "vdp"]
SYSTEMS = LOSE + WIN
TIERS = ["sequence_ood", "shape_ood", "id"]
GB = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
SEEDS = [0, 1, 2]
WAVE1 = ["A0", "A1", "A2", "A3", "A4", "A4-S", "A4-P", "A4-H", "D1", "D2", "D3"]
STACK = ["A1", "A2", "A3", "A4"]
SHORT = {"C3-partial": "partial", "C3-mismatch30": "mm30"}


def cells(systems=SYSTEMS, nodes=NODES):
    return [(n, s) for n in nodes for s in systems]


def cname(c):
    return f"{c[1]}/{SHORT.get(c[0], c[0])}"


_tree = {}


def tree(node, system, seed):
    k = (node, system, seed)
    if k not in _tree:
        r = json.load(open(LIVE / "experiments/tree" / node / "runs" / system / f"seed{seed}/results.json"))
        _tree[k] = {t: r["results"][t]["clean"]["methods"] for t in TIERS}
    return _tree[k]


def arm(node, system, seed, a, root=HERE / "runs"):
    f = root / node / system / f"seed{seed}" / f"{a}.json"
    return json.load(open(f)) if f.exists() else None


def per_seed(node, system, a, tier, seeds=SEEDS):
    out = []
    for s in seeds:
        r = arm(node, system, s, a)
        if r is None:
            return None
        out.append(np.asarray(r["results"][tier]["nrmse"], float))
    return out


def gb_best(node, system, tier, seeds=SEEDS):
    best, arrs = None, None
    for g in GB:
        a = [np.asarray(tree(node, system, s)[tier][g]["nrmse"], float) for s in seeds]
        m = np.mean([x.mean() for x in a])
        if best is None or m < best[1]:
            best, arrs = (g, m), a
    return best[0], arrs


def cmp(a, b):
    r = paired_relative(a, b)
    return {"rel": r["rel"], "ci": r["ci"], "better": r["better"], "worse": r["worse"]}


def fmt(r):
    if r is None:
        return "-"
    s = f"{100 * r['rel']:+.0f}% [{100 * r['ci'][0]:+.0f}, {100 * r['ci'][1]:+.0f}]"
    return f"**{s}**" if r["better"] else (f"*{s}*" if r["worse"] else s)


def analyze(arms, seeds=SEEDS):
    res = {"arms": {}, "greybox_best": {}, "seeds": seeds}
    for c in cells():
        for t in TIERS:
            g, ga = gb_best(*c, t, seeds)
            res["greybox_best"][f"{cname(c)}|{t}"] = {"method": g, "mean": float(np.mean([x.mean() for x in ga]))}
    for a in arms:
        A = {"vs_gb": {}, "vs_A0": {}, "mean": {}, "complete": True}
        for c in cells():
            for t in TIERS:
                x = per_seed(*c, a, t, seeds)
                key = f"{cname(c)}|{t}"
                if x is None:
                    A["complete"] = False
                    continue
                A["mean"][key] = float(np.mean([v.mean() for v in x]))
                A["vs_gb"][key] = cmp(x, gb_best(*c, t, seeds)[1])
                x0 = per_seed(*c, "A0", t, seeds)
                if x0 is not None and a != "A0":
                    A["vs_A0"][key] = cmp(x, x0)
        res["arms"][a] = A
    return res


def counts(A, which, cellset, tier):
    n_b = sum(1 for c in cellset if A[which].get(f"{cname(c)}|{tier}", {}).get("better"))
    n_w = sum(1 for c in cellset if A[which].get(f"{cname(c)}|{tier}", {}).get("worse"))
    return n_b, n_w


def val_geomean(a, ref="A0", seeds=SEEDS):
    logs = []
    for c in cells():
        for s in seeds:
            ra, r0 = arm(*c, s, a), arm(*c, s, ref)
            if ra is None or r0 is None:
                return None
            if not ra.get("info") or not ra["info"].get("val"):
                return None
            logs.append(math.log(ra["info"]["val"]["val_nrmse"] / r0["info"]["val"]["val_nrmse"]))
    return float(math.exp(np.mean(logs)))


def helpful(A):
    b, w = counts(A, "vs_A0", cells(), "sequence_ood")
    return b >= 4 and w <= 1, b, w


def verdict(A, lose_systems=LOSE, need=4):
    lose = cells(lose_systems); win = cells(WIN)
    b_gb, w_gb = counts(A, "vs_gb", lose, "sequence_ood")
    b_a0, w_a0 = counts(A, "vs_A0", win, "sequence_ood")
    ok = b_gb >= need and w_a0 == 0
    return {"losing_cells": [cname(c) for c in lose], "need": need,
            "beats_gb_losing": b_gb, "beaten_cells": [cname(c) for c in lose if A["vs_gb"].get(f"{cname(c)}|sequence_ood", {}).get("better")],
            "worse_than_gb_losing": w_gb, "worse_than_A0_winning": w_a0,
            "better_than_A0_winning": b_a0, "PASS": bool(ok)}


def verdict_posthoc(A):
    """Post-hoc exclusion of duffing (author decision, PREREG addendum 1): 4 losing cells, need 3."""
    return verdict(A, ["battery", "quad_tank"], 3)


def cost_table(arms, seeds=SEEDS):
    rows = {}
    for a in arms:
        P, W = [], []
        for c in cells():
            for s in seeds:
                r = arm(*c, s, a)
                if r is None:
                    continue
                i = r["info"]
                if not i or "params" not in i:
                    continue
                P.append(i["params"]); W.append((i["phase1_wall_s"], i["phase2_wall_s"],
                                                 (r.get("eval_timing") or [{}])[0].get("run_s", float("nan"))))
                st = i["steps"]
        if not P:
            continue
        key = lambda k: int(np.median([p[k] for p in P]))
        rows[a] = {"encoder": key("encoder"), "base_decoder": key("base_decoder"), "conditioner": key("conditioner"),
                   "context": key("context"), "history": key("history"),
                   "base_trainable_p2": key("base_decoder_trainable_p2"),
                   "p2_trainable": key("phase2_trainable_total"),
                   "total_trained": key("encoder") + key("base_decoder") + key("conditioner") + key("context") + key("history"),
                   "steps": st, "p1_wall_s_median": float(np.median([w[0] for w in W])),
                   "p2_wall_s_median": float(np.median([w[1] for w in W])),
                   "eval_run_s_median": float(np.nanmedian([w[2] for w in W])), "n_jobs": len(P)}
    return rows


def md_tables(res, arms, final=None):
    L = []
    for t in TIERS:
        L.append(f"\n### vs best grey-box filter, {t} (paired t-CI over seeds; bold = arm significantly better, italic = significantly worse)\n")
        hdr = "| arm | " + " | ".join(cname(c) for c in cells()) + " | losing: better/worse | winning: better/worse |"
        L.append(hdr); L.append("|" + "---|" * (len(cells()) + 3))
        for a in arms:
            A = res["arms"].get(a)
            if not A:
                continue
            row = [fmt(A["vs_gb"].get(f"{cname(c)}|{t}")) for c in cells()]
            bl, wl = counts(A, "vs_gb", cells(LOSE), t); bw, ww = counts(A, "vs_gb", cells(WIN), t)
            L.append(f"| {a} | " + " | ".join(row) + f" | {bl}/{wl} | {bw}/{ww} |")
    for t in TIERS:
        L.append(f"\n### vs A0 (current default), {t}\n")
        L.append("| arm | " + " | ".join(cname(c) for c in cells()) + " | better/worse (12) |")
        L.append("|" + "---|" * (len(cells()) + 2))
        for a in arms:
            A = res["arms"].get(a)
            if not A or a == "A0":
                continue
            row = [fmt(A["vs_A0"].get(f"{cname(c)}|{t}")) for c in cells()]
            b, w = counts(A, "vs_A0", cells(), t)
            L.append(f"| {a} | " + " | ".join(row) + f" | {b}/{w} |")
    L.append("\n### Absolute NRMSE (seed means), sequence_ood / shape_ood / id\n")
    L.append("| arm | " + " | ".join(cname(c) for c in cells()) + " |")
    L.append("|" + "---|" * (len(cells()) + 1))
    for t in TIERS:
        g = [f"{res['greybox_best'][f'{cname(c)}|{t}']['mean']:.4f}" for c in cells()]
        L.append(f"| best grey-box ({t}) | " + " | ".join(g) + " |")
        for a in arms:
            A = res["arms"].get(a)
            if not A:
                continue
            L.append(f"| {a} ({t}) | " + " | ".join(f"{A['mean'].get(f'{cname(c)}|{t}', float('nan')):.4f}" for c in cells()) + " |")
    L.append("\nBest grey-box filter per cell (sequence_ood): " + ", ".join(
        f"{cname(c)} {res['greybox_best'][f'{cname(c)}|sequence_ood']['method']}" for c in cells()))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(WAVE1))
    ap.add_argument("--out", default=str(HERE / "analysis.json"))
    ap.add_argument("--md", default=str(HERE / "tables.md"))
    a = ap.parse_args()
    arms = a.arms.split(",")
    res = analyze(arms)
    res["verdict_by_arm"] = {x: verdict(res["arms"][x]) for x in arms if res["arms"][x]["complete"]}
    res["verdict_by_arm_posthoc_no_duffing"] = {x: verdict_posthoc(res["arms"][x]) for x in arms if res["arms"][x]["complete"]}
    res["helpful_design"] = {x: dict(zip(("helpful", "better", "worse"), helpful(res["arms"][x])))
                             for x in arms if x.startswith("D") and res["arms"][x]["complete"]}
    res["val_geomean_vs_A0"] = {x: val_geomean(x) for x in arms}
    res["cost"] = cost_table(arms)
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.md).write_text(md_tables(res, arms))
    for x, v in res["verdict_by_arm"].items():
        print(x, "PRE", v["PASS"], v["beats_gb_losing"], v["beaten_cells"], "worseA0win", v["worse_than_A0_winning"],
              "| POSTHOC", res["verdict_by_arm_posthoc_no_duffing"][x]["PASS"], res["verdict_by_arm_posthoc_no_duffing"][x]["beats_gb_losing"],
              "| val-gm", res["val_geomean_vs_A0"].get(x))
    print(res["helpful_design"])


if __name__ == "__main__":
    main()
