"""Achieved settled NRMSE of every observer per system, from the finished runs (read-only).

    .venv/bin/python experiments/information/collect_achieved.py  -> achieved.json

Sources (tiers id and sequence_ood, condition clean, mean over seeds of the per-seed mean over
episodes; per-state and unmeasured-state NRMSE from the stored scorecards):
  full knowledge:    C1-where (dev) / TEST-C1 (test): lora (= HyperKKL r=4), latent, concat, film, hyper, ekf
  partial knowledge: C3-regimes (dev) / TEST-C3 (test): lora, hyper, ekf, ukf, greybox_ekf, greybox_ukf, ...
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infolib as L  # noqa: E402

TIERS = ("id", "sequence_ood")


def collect(system: str, node: str) -> dict:
    runs = sorted((L.TREE / node / "runs" / system).glob("seed*/results.json"))
    acc = {}
    for p in runs:
        j = json.loads(p.read_text())
        for tier in TIERS:
            blk = j["results"].get(tier, {}).get("clean")
            if not blk:
                continue
            for m, sc in blk["scorecard"].items():
                d = acc.setdefault(m, {}).setdefault(tier, {"nrmse": [], "state": [], "unmeasured": [], "p95": [],
                                                            "rms": [], "rms_state": []})
                d["nrmse"].append(sc["nrmse_mean"]); d["state"].append(sc["nrmse_state"])
                d["unmeasured"].append(sc["nrmse_unmeasured"]); d["p95"].append(sc["nrmse_p95"])
                me = blk["methods"][m]                      # per-episode lists -> pooled RMS (what a CRB bounds)
                nx = len(sc["nrmse_state"])
                d["rms"].append(float(np.sqrt(np.mean(np.square(me["nrmse"])))))
                d["rms_state"].append([float(np.sqrt(np.nanmean(np.square(me[f"nrmse_x{j + 1}"])))) for j in range(nx)])
    out = {"node": node, "n_runs": len(runs), "methods": {}}
    for m, tiers in acc.items():
        out["methods"][m] = {}
        for tier, d in tiers.items():
            out["methods"][m][tier] = {
                "nrmse": float(np.mean(d["nrmse"])), "nrmse_seeds": d["nrmse"],
                "state": np.mean(np.array(d["state"]), 0).tolist(),
                "unmeasured": float(np.nanmean(d["unmeasured"])), "p95": float(np.mean(d["p95"])), "n_seeds": len(d["nrmse"]),
                "rms": float(np.mean(d["rms"])), "rms_state": np.mean(np.array(d["rms_state"]), 0).tolist()}
    return out


if __name__ == "__main__":
    res = {}
    for s in L.ALL:
        res[s] = {"full": collect(s, L.RUN_FULL[s]), "partial": collect(s, L.RUN_PARTIAL[s])}
    (HERE / "achieved.json").write_text(json.dumps(res, indent=1))
    print("written", HERE / "achieved.json")
