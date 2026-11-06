"""Benchmark tracks.

core           12 frozen plants x 4 knowledge levels, test tiers id / shape_ood / sequence_ood,
               conditions clean + outliers + burst_outage + poor_init. Headline per cell: settled
               unmeasured NRMSE, clean, mean of the three tiers. Track score: geometric mean over
               cells of (method / ekf_gated) (the standalone gated EKF is the fixed yardstick),
               plus the number of cells won, each with its episode-bootstrap CI.
input_ood      the paper's focus: same cells; scored on the OOD tiers only and on the OOD gap
               g = NRMSE(tier) / NRMSE(test_id) for shape_ood and sequence_ood. Model selection
               may use val_ood (public, different generators); never the test tiers.
deployment     resource-constrained deployment (casestudies): drone ``quad_planar`` with a
               150 us / 512 KB per-step budget (latency measured on CPU batch 1 AND the MCU
               estimate), and CSTR ``cstr_exo`` with sparse, jittered lab labels (C_A assay every
               45 +- 15 samples) instead of full-state training labels.
real           measured data of real systems (battery_pf, battery_lg, emps, silverbox, neurobem; hyperkkl_bench.real):
               fetched from the original source, never redistributed; own task + scorer per dataset.
scaling        ``fhnring`` with n_x in {4, 8, 16, 32, 64} (n_y = n_x / 4): how accuracy and cost
               scale with the state dimension.

Status in hkb-data-1.0: core, input_ood and deployment are frozen and runnable (deployment episodes
exported from the research cache by tools/export_casestudies.py into data_extra/; in cstr_exo the
training splits carry C_A only at the lab-assay samples, NaN elsewhere, so a method must cope with
missing labels). The latency budget of quad_planar is checked on the measured CPU latency; the
research MCU estimate is in resources/tracks_reference.json. scaling is frozen too
(tools/export_scaling.py; training sizes follow the pre-registered rule n_train = 600 n_x / 8).
"""

from __future__ import annotations

import math

import numpy as np

from .registry import CORE, KNOWLEDGE

TRACKS = {
    # v2 (2.0): the two families of state estimation (hyperkkl_bench.families); cells are "family/plant/level"
    "autonomous": {"kind": "v2", "family": "autonomous", "levels": ["partial", "exact", "mismatch", "realistic"],
                   "headline": "test_id/clean", "frozen": True},
    "driven": {"kind": "v2", "family": "driven", "levels": ["realistic", "partial", "exact", "mismatch"],
               "headline": "test_op_ood/clean", "frozen": True},
    # real-data tracks (hyperkkl_bench.real): measured data of real input-driven systems, fetched from the original
    # source; one task and scorer per dataset (battery: settled SOC RMSE; mechanics: settled velocity NRMSE)
    "real": {"plants": ["battery_pf", "battery_lg", "emps", "silverbox", "neurobem", "calce"], "knowledge": ["real"], "frozen": True,
             "kind": "real", "required": ["battery_pf"]},
    "core": {"plants": CORE, "knowledge": list(KNOWLEDGE), "splits": ["test_id", "test_shape_ood", "test_sequence_ood"],
             "conditions": ["clean", "outliers", "burst_outage", "poor_init"], "frozen": True},
    "input_ood": {"plants": CORE, "knowledge": list(KNOWLEDGE), "splits": ["test_id", "test_shape_ood", "test_sequence_ood"],
                  "conditions": ["clean"], "frozen": True, "model_selection_split": "val_ood"},
    "deployment": {"plants": ["quad_planar", "cstr_exo"], "knowledge": list(KNOWLEDGE),
                   "splits": ["test_id", "test_shape_ood", "test_sequence_ood"],
                   "conditions": ["clean", "outliers", "burst_outage", "poor_init"], "frozen": True,
                   "budgets": {"quad_planar": {"latency_us": 150.0, "memory_bytes": 512 * 1024}},
                   "labels": {"cstr_exo": {"states": [0], "every": 45, "jitter": 15, "noise": 0.05}}},
    "scaling": {"plants": [f"fhnring_n{n}" for n in (4, 8, 16, 32, 64)], "n_x": [4, 8, 16, 32, 64],
                "knowledge": list(KNOWLEDGE), "splits": ["test_id", "test_shape_ood", "test_sequence_ood"],
                "conditions": ["clean"], "frozen": True},
}


def run_track(make_estimator, track: str = "core", plants=None, knowledge=None, verbose=False, **kw) -> dict:
    """make_estimator(plant, knowledge) -> fresh estimator. Returns {plant/knowledge: Report}."""
    from .evaluate import evaluate
    t = TRACKS[track]
    if t.get("kind") == "v2":
        from . import families as F
        out = {}
        for p in plants or (F.AUTONOMOUS if t["family"] == "autonomous" else F.DRIVEN):
            for lv in knowledge or t["levels"][:1]:
                spec = f"{t['family']}/{p}/{lv}"
                try:
                    F.parse(spec)
                except ValueError:
                    continue
                out[spec] = F.evaluate_v2(make_estimator(spec, lv), spec, verbose=verbose, **kw)
        return out
    if t.get("kind") == "real":
        from . import real
        return {f"{p}/real": real.evaluate(make_estimator(p, "real"), p, verbose=verbose, **kw) for p in plants or t["plants"]}
    if not t["frozen"]:
        raise NotImplementedError(f"track {track!r} has no frozen episodes in this data release (see tracks.py)")
    out = {}
    for p in plants or t["plants"]:
        b = t.get("budgets", {}).get(p, {})
        for k in knowledge or t["knowledge"]:
            out[f"{p}/{k}"] = evaluate(make_estimator(p, k), p, k, splits=t["splits"],
                                       conditions=t["conditions"], verbose=verbose,
                                       budget_us=b.get("latency_us"), budget_bytes=b.get("memory_bytes"), **kw)
    return out


def cell_headline(rep) -> float:
    vals = [rep.value(s, "clean") for s in ("test_id", "test_shape_ood", "test_sequence_ood") if s in rep.results]
    return float(np.mean(vals))


def track_score(reports: dict, reference: dict) -> dict:
    """reports / reference: {cell: Report} on the same cells. Geometric-mean ratio and wins."""
    common = sorted(set(reports) & set(reference))
    r = np.array([cell_headline(reports[c]) / cell_headline(reference[c]) for c in common])
    lr = np.log(r)
    se = lr.std(ddof=1) / math.sqrt(len(lr)) if len(lr) > 1 else float("nan")
    return {"cells": len(common), "geo_mean_ratio": float(np.exp(lr.mean())),
            "geo_mean_ratio_ci95_over_cells": [float(np.exp(lr.mean() - 1.96 * se)), float(np.exp(lr.mean() + 1.96 * se))],
            "wins": int((r < 1).sum()), "per_cell_ratio": dict(zip(common, r.tolist()))}


def reference_reports(baseline: str = "ekf_gated", track: str = "core") -> dict:
    """The stored reference reports (reference/runs/<baseline>/<plant>__<knowledge>.json in the
    checkout) of a track's cells, as {plant/knowledge: Report}: the yardstick of track_score."""
    from pathlib import Path
    from .evaluate import Report
    from .paths import CHECKOUT
    d = Path(CHECKOUT) / "reference" / "runs" / baseline
    t = TRACKS[track]
    out = {}
    for p in t["plants"]:
        for k in t.get("knowledge", []):
            f = d / f"{p}__{k}.json"
            if f.exists():
                out[f"{p}/{k}"] = Report.load(f)
    return out
