"""Recompute the v2 reference tables (seed means, per-cell order on the primary block) from runs_v2.json and check them
against results.json (written by experiments/bench_v2/code/analyze.py). No training, no research code.

    python reference/v2/reproduce_v2.py [--check]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def recompute():
    R = json.loads((HERE / "runs_v2.json").read_text())
    det = set(R["deterministic"])
    out = {}
    for cell, obs in R["cells"].items():
        fam = cell.split("/")[0]
        blk = R["primary"][fam]
        seeds = sorted({s for o, d in obs.items() if o not in det for s in d}) or ["0"]
        summ = {}
        for o, d in obs.items():
            if o in det and "0" in d:
                vals = [d["0"]["blocks"].get(blk, {}).get("nrmse_unmeasured", np.nan)] * len(seeds)
            else:
                vals = [d[s]["blocks"].get(blk, {}).get("nrmse_unmeasured", np.nan) for s in seeds if s in d]
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            if len(v):
                summ[o] = float(v.mean())
        out[cell] = {"order": sorted(summ, key=summ.get), "mean": summ}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rc = recompute()
    if not a.check:
        for c, d in rc.items():
            print(c, "best", d["order"][0], "hyperkkl rank", d["order"].index("hyperkkl") + 1 if "hyperkkl" in d["order"] else None)
        return
    ref = json.loads((HERE / "results.json").read_text())["cells"]
    bad = 0
    for c, d in rc.items():
        r = ref[c]
        if d["order"] != r["order"]:
            bad += 1; print("ORDER MISMATCH", c)
        for o, v in d["mean"].items():
            if abs(v - r["summary"][o]["mean"]) > 1e-12 * max(1.0, abs(v)):
                bad += 1; print("VALUE MISMATCH", c, o, v, r["summary"][o]["mean"])
    n = sum(len(d["mean"]) for d in rc.values())
    print("ALL MATCH" if bad == 0 else f"{bad} MISMATCHES", f"({len(rc)} cells, {n} method means)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
