"""Held-out TEST-C3 pass: arm vs best grey-box (merged TEST-C3 / TEST-C3-B runs) and vs A0."""
import json
import sys

import numpy as np

import analyze_bg as A

ARM = sys.argv[1] if len(sys.argv) > 1 else "A4+D3"
SETS = {"pre-registered (orig. cartpole)": [("TEST-C3", s) for s in ("cartpole", "fhn", "highway_traffic", "network", "rossler", "two_link_arm")],
        "cartpole_b replacing cartpole": [("TEST-C3-B", "cartpole_b")] + [("TEST-C3", s) for s in ("fhn", "highway_traffic", "network", "rossler", "two_link_arm")]}
def gb_arr(node, system, s, g, t):
    m = A.tree(node, system, s)[t]
    if g in m:
        return np.asarray(m[g]["nrmse"], float), "tree"
    r = A.arm(node, system, s, g)
    return (np.asarray(r["results"][t]["nrmse"], float), "refit") if r else (None, None)


def gb_best(node, system, t):
    best = None
    for g in A.GB:
        arrs = [gb_arr(node, system, s, g, t) for s in A.SEEDS]
        if any(a is None for a, _ in arrs):
            continue
        m = np.mean([a.mean() for a, _ in arrs])
        if best is None or m < best[1]:
            best = (g, m, [a for a, _ in arrs], sorted({src for _, src in arrs}))
    return best


out = {}
for t in ("sequence_ood", "shape_ood", "id"):
    rows = {}
    for c in SETS["pre-registered (orig. cartpole)"] + [("TEST-C3-B", "cartpole_b")]:
        n = f"{c[1]}|{t}"
        x = A.per_seed(*c, ARM, t); x0 = A.per_seed(*c, "A0", t)
        gbb = gb_best(*c, t)
        if x is None or gbb is None:
            rows[n] = {"missing": True}
            continue
        g, _, ga, src = gbb
        seeds, xs = src, x
        rows[n] = {"arm_mean": float(np.mean([v.mean() for v in x])), "A0_mean": float(np.mean([v.mean() for v in x0])),
                   "gb": g, "gb_mean": float(np.mean([v.mean() for v in ga])), "gb_seeds": seeds,
                   "vs_gb": A.cmp(xs, ga), "vs_A0": A.cmp(x, x0)}
    out[t] = rows
res = {"arm": ARM, "rows": out, "counts": {}}
for lab, cs in SETS.items():
    for t in ("sequence_ood", "shape_ood", "id"):
        r = [out[t][f"{c[1]}|{t}"] for c in cs]
        res["counts"][f"{lab}|{t}"] = {"beats_gb": sum(1 for x in r if x.get("vs_gb", {}).get("better")),
                                       "worse_than_gb": sum(1 for x in r if x.get("vs_gb", {}).get("worse")),
                                       "better_than_A0": sum(1 for x in r if x.get("vs_A0", {}).get("better")),
                                       "worse_than_A0": sum(1 for x in r if x.get("vs_A0", {}).get("worse")),
                                       "missing": [c[1] for c, x in zip(cs, r) if x.get("missing")]}
json.dump(res, open(f"analysis_test_{ARM}.json", "w"), indent=1)
for k, v in res["counts"].items():
    print(k, v)
for k, v in out["sequence_ood"].items():
    if not v.get("missing"):
        print(k, f"arm {v['arm_mean']:.4f} A0 {v['A0_mean']:.4f} {v['gb']} {v['gb_mean']:.4f} (seeds {v['gb_seeds']})",
              A.fmt(v["vs_gb"]), "vsA0", A.fmt(v["vs_A0"]))
    else:
        print(k, v)
