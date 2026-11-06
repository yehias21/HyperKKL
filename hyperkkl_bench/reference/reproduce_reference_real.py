"""Reference results of the real-data track ``battery_pf`` from the STORED research runs (NO training, NO
inference): every number is recomputed from stored per-episode predictions / per-episode scores and checked
against the aggregate the research code stored.

Sources (read-only, research repository next to this package):
  experiments/real/cache/predictions.pt                    EKF, UKF, coulomb counting, HyperKKL variants (lora = the
  experiments/real/cache/predictions_baselines_seed*.pt    paper's method, hyper, latent, concat, autonomous; 3 seeds),
                                                           grey-box EKF / UKF, Koopman KF, NLOX (3 seeds)
       -> checked against experiments/real/results.json["tables"] (seed means and per-seed means)
  experiments/baselines2/real/{gru,ncde}_seed<k>.json       per-episode settled RMSE (GRU, neural CDE; 3 seeds)
       -> checked against experiments/baselines2/real/summary.json
  experiments/beat_greybox/real/results_{A4+D3,B2-A4+D3}.json   HyperKKL A4+D3 / B2 per-seed values (reported as
                                                           stored: their per-episode predictions were not kept)
Cell value: settled SOC RMSE [% SOC] (t >= 15 min), mean over the family's episodes per seed, then over seeds.
The episodes are exactly the benchmark's (tests/test_real_battery.py checks the windows / labels / priors).

    python reference/reproduce_reference.py --real          # writes reference/real/REFERENCE_battery_pf.md + .json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments"
TOL = 1e-9
FAM25 = {"US06": ["test25/US06"], "HWFET": ["test25/HWFTa", "test25/HWFTb"], "NN": ["test25/NN"]}
FAM0 = ["Cycle_1", "Cycle_2", "LA92", "NN", "Cycle_3", "Cycle_4", "US06", "HWFET", "UDDS"]
SEEDS = (0, 1, 2)
LABEL = {"lora": "HyperKKL (lora r=4, the paper's method; research recipe A0)", "hyper": "HyperKKL dense generator (ablation)",
         "ws_best": "WS-best (lora/hyper by validation, pre-registered)", "latent": "latent input injection",
         "concat": "concat", "autonomous": "autonomous (no input)", "greybox_ukf": "grey-box UKF (ECM + residual MLP)",
         "greybox_ekf": "grey-box EKF (ECM + residual MLP)", "koopman_kf": "Koopman KF (EDMD)",
         "koopman_kf_bilinear": "Koopman KF bilinear", "nlox": "NLOX", "ekf": "EKF on the identified ECM (val-tuned)",
         "ukf": "UKF on the identified ECM (val-tuned)", "cc": "coulomb counting from the prior",
         "gru": "GRU (140k)", "ncde": "neural CDE (140k)", "A4+D3": "HyperKKL A4+D3 (beat_greybox)",
         "B2-A4+D3": "HyperKKL B2 on A4+D3 (beat_greybox)"}


def _soc_rmse(pred, truth, settle=900):
    e = np.nan_to_num((np.asarray(pred, float) - np.asarray(truth, float)) * 100.0, nan=1e4, posinf=1e4, neginf=-1e4)
    return np.sqrt(np.mean(e[:, settle:] ** 2, 1))


def run(log=print, write=True):
    import torch
    res = json.loads((EXP / "real" / "results.json").read_text())
    tables = res["tables"]
    ws = res["decision"]["ws_choice_per_seed"]
    st = torch.load(EXP / "real" / "cache" / "predictions.pt", weights_only=False)
    pred, truth = dict(st["pred"]), st["truth"]
    for p in sorted((EXP / "real" / "cache").glob("predictions_baselines_seed*.pt")):
        pred.update(torch.load(p, weights_only=False)["pred"])
    for s in SEEDS:
        for n in truth:
            if (n, ws[str(s)], s) in pred:
                pred[(n, "ws_best", s)] = pred[(n, ws[str(s)], s)]
    rm = {}

    def fam_value(method, sets):
        if (sets[0], method) in pred:                          # deterministic (filters, cc)
            return float(np.mean(np.concatenate([_soc_rmse(pred[(n, method)], truth[n]) for n in sets]))), None
        ps = [float(np.mean(np.concatenate([_soc_rmse(pred[(n, method, s)], truth[n]) for n in sets]))) for s in SEEDS]
        return float(np.mean(ps)), ps

    cells, n_chk, bad = {}, 0, []
    groups = {"test25": FAM25, "test0": {c: [f"test0/{c}"] for c in FAM0}, "val": {"Cycle_4": ["val"]}}
    methods = [m for m in tables["test25"]["US06"] if m != "hyperkkl"]
    for g, fams in groups.items():
        for f, sets in fams.items():
            for m in methods:
                try:
                    v, ps = fam_value(m, sets)
                except KeyError:
                    continue
                ref = tables[g][f][m]["rmse_settled"]
                n_chk += 1
                if abs(v - ref["mean"]) > TOL * max(1, abs(v)):
                    bad.append((g, f, m, v, ref["mean"]))
                if ps is not None and "per_seed" in ref:
                    n_chk += 1
                    if max(abs(a - b) for a, b in zip(ps, ref["per_seed"])) > TOL * 100:
                        bad.append((g, f, m, "per_seed", ps, ref["per_seed"]))
                cells.setdefault(m, {})[f"{g}/{f}"] = {"mean": v, "per_seed": ps}
    # GRU / NCDE (baselines2/real): per-episode lists
    summ = json.loads((EXP / "baselines2" / "real" / "summary.json").read_text())
    for kind in ("gru", "ncde"):
        per = {s: json.loads((EXP / "baselines2" / "real" / f"{kind}_seed{s}.json").read_text())["results"] for s in SEEDS}
        for g, fams in groups.items():
            for f, sets in fams.items():
                ps = [float(np.mean(np.concatenate([per[s][n]["rmse_settled"] for n in sets]))) for s in SEEDS]
                cells.setdefault(kind, {})[f"{g}/{f}"] = {"mean": float(np.mean(ps)), "per_seed": ps}
                if g == "test25":
                    ref = summ[kind]["fam"][f]
                    n_chk += 1
                    if abs(np.mean(ps) - ref["mean"]) > TOL or max(abs(a - b) for a, b in zip(ps, ref["per_seed"])) > TOL:
                        bad.append((kind, f, ps, ref["per_seed"]))
                elif g == "test0" and f in summ[kind].get("test0_mean", {}):
                    n_chk += 1
                    if abs(np.mean(ps) - summ[kind]["test0_mean"][f]) > TOL:
                        bad.append((kind, "test0", f, np.mean(ps), summ[kind]["test0_mean"][f]))
    # beat_greybox real arms (stored per-seed values only)
    for arm in ("A4+D3", "B2-A4+D3"):
        p = EXP / "beat_greybox" / "real" / f"results_{arm}.json"
        if p.exists():
            d = json.loads(p.read_text())["settled_rmse"]
            for f in FAM25:
                ps = d[f][arm]
                cells.setdefault(arm, {})[f"test25/{f}"] = {"mean": float(np.mean(ps)), "per_seed": ps, "stored_only": True}
    ok = not bad and n_chk > 0
    for b in bad[:10]:
        log("MISMATCH", b)
    log(f"real battery_pf: {n_chk} stored aggregates checked: {'ALL MATCH' if ok else 'MISMATCHES'}")
    if write:
        out = HERE / "real"
        out.mkdir(exist_ok=True)
        (out / "REFERENCE_battery_pf.json").write_text(json.dumps(
            {"about": __doc__, "n_checked": n_chk, "all_match": ok, "cells": cells}, indent=1))
        (out / "REFERENCE_battery_pf.md").write_text(table(cells))
    return ok, n_chk, cells


def table(cells) -> str:
    cols = ["test25/US06", "test25/HWFET", "test25/NN"]
    order = ["lora", "ws_best", "hyper", "A4+D3", "B2-A4+D3", "latent", "concat", "autonomous", "gru", "ncde",
             "greybox_ukf", "greybox_ekf", "koopman_kf", "koopman_kf_bilinear", "nlox", "ekf", "ukf", "cc"]
    L = ["# battery_pf: reference results from the stored research runs", "",
         "Settled SOC RMSE [% SOC] (t >= 15 min of each 30-min window, prior SOC 20 % off), mean over the family's",
         "episodes, then over 3 training seeds (± sd over seeds). 25 degC test families; headline = their mean.",
         "Regenerated by `python reference/reproduce_reference.py --real` (no training, no inference); every value",
         "except the beat_greybox arms (stored per-seed values only) is recomputed from stored per-episode outputs and",
         "checked against the stored aggregates.", "",
         "| method | US06 | HWFET | NN | headline | 0 degC (mean of 9 cycles) |", "|---|---|---|---|---|---|"]
    for m in order:
        if m not in cells:
            continue
        c = cells[m]
        row = []
        for k in cols:
            v = c.get(k)
            if v is None:
                row.append("")
                continue
            sd = f" ± {np.std(v['per_seed'], ddof=1):.2f}" if v.get("per_seed") else ""
            row.append(f"{v['mean']:.2f}{sd}")
        h = np.mean([c[k]["mean"] for k in cols]) if all(k in c for k in cols) else None
        t0 = [c[f"test0/{f}"]["mean"] for f in FAM0 if f"test0/{f}" in c]
        L.append(f"| {LABEL.get(m, m)} | " + " | ".join(row) + f" | {'' if h is None else f'{h:.2f}'} | "
                 f"{f'{np.mean(t0):.2f}' if len(t0) == len(FAM0) else ''} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys
    sys.exit(0 if run()[0] else 1)
