"""Regenerate the reference results table from the stored per-run JSONs (NO training, NO
inference) and check every number against the aggregate the research code stored.

Sources (read-only, in the research repository next to this package):
  experiments/tree/<node>/runs/<plant>/seed<k>/results.json     HyperKKL variants + model-based filters
       aggregate checked against experiments/tree/<node>/summary.json  (key 'all' = mean over seeds)
  experiments/baselines2/runs/<node>/<plant>/seed<k>/<b>.json    UDE / NODE / GRU / NCDE / NLOX / grey-box
  experiments/baselines2/references.json                         HyperKKL A0 / A3 / A4+D3 per-episode refs
       aggregate checked against experiments/baselines2/results.json['means']
Cell value = mean over training seeds of the per-seed mean over the 64 test episodes (both the
all-state settled NRMSE and the benchmark headline, the unmeasured settled NRMSE).

    python reference/reproduce_reference.py            # writes reference/reference_results.json + REFERENCE.md
    python reference/reproduce_reference.py --check    # only check, exit 1 on any mismatch > 1e-9
    python reference/reproduce_reference.py --real     # the real-data track battery_pf (reproduce_reference_real.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments"
TIERS = ("id", "shape_ood", "sequence_ood")
TOL = 1e-9

# knowledge level -> tree node (dev plants, test plants, cartpole_b)
TREE_NODES = {"exact": ("C1-where", "TEST-C1", "TEST-C1-B"),
              "partial": ("C3-partial", None, None),
              "mismatch": ("C3-mismatch30", None, None),
              "realistic": ("C3-realistic", "TEST-C3", "TEST-C3-B")}
B2_NODES = {"C3-partial": "partial", "C3-mismatch30": "mismatch", "TEST-C3": "realistic", "TEST-C3-B": "realistic"}
DEV = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]
TEST = ["fhn", "highway_traffic", "network", "rossler", "two_link_arm"]
TREE_METHODS = ("hyper", "lora", "latent", "film", "concat", "ekf", "ukf", "pf", "ekf_huber", "ekf_oracle",
                "ukf_oracle", "greybox_ekf", "greybox_ukf", "nlox", "koopman_kf")
B2_METHODS = ("ude5k1_ekf", "ude5k1_ukf", "ude5k_ekf", "node5k_ekf", "gru140", "gru500", "ncde140", "nlox140",
              "greybox_ekf", "A3")


def _agg(per_seed: dict) -> dict:
    """per_seed: {seed: {nrmse: [...], nrmse_unmeasured: [...], diverged: [...]}}"""
    seeds = sorted(per_seed)
    ps = {m: [float(np.mean(per_seed[s][m])) for s in seeds] for m in ("nrmse", "nrmse_unmeasured")}
    return {"nrmse": float(np.mean(ps["nrmse"])), "nrmse_unmeasured": float(np.mean(ps["nrmse_unmeasured"])),
            "nrmse_unmeasured_sd_seeds": float(np.std(ps["nrmse_unmeasured"], ddof=1)) if len(seeds) > 1 else None,
            "div": float(np.mean([np.mean(per_seed[s].get("diverged", [0])) for s in seeds])),
            "seeds": seeds, "per_seed_nrmse": ps["nrmse"]}


def tree_cells(log):
    out, n_chk, bad = {}, 0, []
    for k, nodes in TREE_NODES.items():
        for node, plants in zip(nodes, (DEV, TEST, ["cartpole_b"])):
            if node is None:
                continue
            summ = EXP / "tree" / node / "summary.json"
            S = json.loads(summ.read_text())["nrmse"]["summary"] if summ.exists() else {}
            for p in plants:
                runs = sorted((EXP / "tree" / node / "runs" / p).glob("seed*/results.json"))
                if not runs:
                    continue
                data = {}
                for f in runs:
                    r = json.loads(f.read_text())
                    for t in TIERS:
                        for m, v in r["results"][t]["clean"]["methods"].items():
                            if m in TREE_METHODS:
                                data.setdefault(m, {}).setdefault(t, {})[r["seed"]] = v
                for m, tv in data.items():
                    for t, ps in tv.items():
                        a = _agg(ps)
                        a["source"] = f"experiments/tree/{node}/runs/{p}/seed*/results.json"
                        if m.startswith("greybox"):
                            a["flawed_P1"] = True
                        out.setdefault(f"{p}/{k}", {}).setdefault(m, {})[t] = a
                        ref = S.get(p, {}).get(t, {}).get("clean", {}).get(m, {}).get("all")
                        if ref is not None:
                            n_chk += 1
                            if abs(ref[0] - a["nrmse"]) > TOL * max(1, abs(ref[0])):
                                bad.append((node, p, m, t, ref[0], a["nrmse"]))
    log(f"tree: {n_chk} cell x method x tier means checked against summary.json, {len(bad)} mismatches")
    return out, n_chk, bad


def b2_cells(log):
    res = json.loads((EXP / "baselines2" / "results.json").read_text())["means"]
    refs = json.loads((EXP / "baselines2" / "references.json").read_text())["refs"]
    data = {}
    for f in (EXP / "baselines2" / "runs").glob("*/*/seed*/*.json"):
        if f.parts[-4] == "smoke":
            continue
        d = json.loads(f.read_text())
        c = f"{d['node']}/{d['system']}"
        for t, r in d["results"].items():
            for m, v in r.items():
                data.setdefault(c, {}).setdefault(m, {}).setdefault(t, {})[d["seed"]] = v
    for c, d in refs.items():
        for m, v in d.items():
            for t, sv in v.items():
                for sd, x in sv.items():
                    data.setdefault(c, {}).setdefault(m, {}).setdefault(t, {})[int(sd)] = x
    out, n_chk, bad = {}, 0, []
    for c, md in data.items():
        node, p = c.split("/")
        k = B2_NODES[node]
        for m, tv in md.items():
            if m not in B2_METHODS:
                continue
            for t, ps in tv.items():
                a = _agg(ps)
                a["source"] = f"experiments/baselines2/runs/{c}/seed*/ (+ references.json)"
                if m.startswith("greybox"):
                    a["flawed_P1"] = True
                out.setdefault(f"{p}/{k}", {}).setdefault("b2:" + m, {})[t] = a
                ref = res.get(c, {}).get(m, {}).get(t, {}).get("mean")
                if ref is not None:
                    n_chk += 1
                    if abs(ref - a["nrmse"]) > TOL * max(1, abs(ref)):
                        bad.append((c, m, t, ref, a["nrmse"]))
    log(f"baselines2: {n_chk} cell x method x tier means checked against results.json, {len(bad)} mismatches")
    return out, n_chk, bad


def table(cells) -> str:
    cols = ["ekf", "ekf_oracle", "ukf", "pf", "b2:ude5k1_ekf", "b2:node5k_ekf", "b2:gru140",
            "b2:nlox140", "hyper", "b2:A3"]
    L = ["# Reference results (regenerated by reproduce_reference.py; no training)", "",
         "Settled **unmeasured** NRMSE on `test_sequence_ood`, clean; mean over training seeds (n in the JSON).",
         "`b2:` = experiments/baselines2 (UDE/NODE/GRU/NLOX baselines; `ude5k1` = corrected grey-box, PITFALLS P1);",
         "`hyper` = HyperKKL base recipe of the tree runs; `b2:A3` = HyperKKL A3 (beat_greybox recipe). Empty = not run.",
         "`ekf` / `ukf` / `pf` here are the research runs' UNGATED filters with nominal Q (the benchmark reference is the",
         "gated EKF, reference/runs/ekf_gated). The research `greybox_ekf` / `greybox_ukf` (one-step residual, PITFALLS P1)",
         "are kept in reference_results.json flagged `flawed_P1` and deliberately left out of this table.", "",
         "| cell | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for c in sorted(cells, key=lambda s: (s.split("/")[1], s)):
        row = []
        for m in cols:
            v = cells[c].get(m, {}).get("sequence_ood")
            row.append("" if v is None else f"{v['nrmse_unmeasured']:.4f}" + ("*" if v["div"] > 0 else ""))
        L.append(f"| {c} | " + " | ".join(row) + " |")
    L += ["", "`*` = some episodes diverged (clipped error counted)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check_only = "--check" in argv
    log = print
    if "--real" in argv:                       # real-data track battery_pf (needs torch; reads ~0.6 GB of stored predictions)
        sys.path.insert(0, str(HERE))
        import reproduce_reference_real as RRR
        ok = RRR.run(log, write=not check_only)[0]
        print("ALL MATCH" if ok else "MISMATCHES FOUND")
        return 0 if ok else 1
    t, n1, bad1 = tree_cells(log)
    b, n2, bad2 = b2_cells(log)
    cells = t
    for c, md in b.items():
        cells.setdefault(c, {}).update(md)
    for x in (bad1 + bad2)[:20]:
        print("MISMATCH", x)
    ok = not bad1 and not bad2 and n1 > 0 and n2 > 0
    if not check_only:
        (HERE / "reference_results.json").write_text(json.dumps(
            {"about": __doc__, "n_checked": n1 + n2, "all_match": ok, "cells": cells}, indent=1))
        (HERE / "REFERENCE.md").write_text(table(cells))
        print("wrote reference_results.json, REFERENCE.md")
    print("ALL MATCH" if ok else "MISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
