"""Bundle the v2 reference results into hyperkkl_bench/reference/v2/ (run after analyze.py and extras.py).

    python make_reference_v2.py

Writes runs_v2.json: for every (cell, observer, seed) the validation-selected run (analyze.load), its lr / file,
and for every block the episode means of nrmse_unmeasured, nrmse, nrmse_transient and the divergence rate. Also copies
results.json, results_extras.json, tables.md, extras.md and the real-track run summaries.
reference/v2/reproduce_v2.py recomputes the per-cell orders and seed means from runs_v2.json and checks them against
results.json.
"""
import json
import shutil

import numpy as np

import analyze as A

HERE = A.HERE
OUT = HERE.parent.parent / "hyperkkl_bench" / "reference" / "v2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = A.load()
    runs = {}
    for (fam, plant, var, o, s), (v, blocks, f, m) in sel.items():
        d = runs.setdefault(f"{fam}/{plant}/{var}", {}).setdefault(o, {})
        d[str(s)] = {"val_score": v, "file": f, "lr": m.get("lr"),
                     "blocks": {b: {k: float(np.mean(bb[k])) for k in ("nrmse_unmeasured", "nrmse", "nrmse_transient", "diverged")}
                                for b, bb in blocks.items()}}
    (OUT / "runs_v2.json").write_text(json.dumps({"primary": A.PRIMARY, "deterministic": sorted(A.DETERMINISTIC),
                                                  "cells": runs}, indent=0))
    for f in ("results.json", "results_extras.json"):
        shutil.copy(HERE / f, OUT / f)
    for f in ("tables.md", "extras.md"):
        shutil.copy(HERE / "logs" / f, OUT / f)
    real = {}
    for f in sorted((HERE / "runs" / "real").glob("*/*/seed*.json")):
        if f.parts[-2].endswith("_smoke"):
            continue
        d = json.loads(f.read_text())
        real.setdefault(f.parts[-3], {}).setdefault(f.parts[-2], {})[str(d["seed"])] = {
            "headline": d["meta"].get("headline"), "headline_metric": d["meta"].get("headline_metric"),
            "splits": {s: r["summary"] for s, r in d["results"].items()}}
    (OUT / "real_v2.json").write_text(json.dumps(real, indent=0, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
