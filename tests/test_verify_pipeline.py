"""Verification of measurement, evaluation and the experiment infrastructure.

    .venv/bin/python -m pytest tests/test_verify_pipeline.py -q -rxs

No simulation, training or GPU: evaluation data, observers, jobs and worker processes are
synthetic, and every experiment-tree path is redirected into tmp_path (experiments/ is
never touched). Tests marked xfail document known limitations (bootstrap coverage with
few seeds) and print the measured numbers.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hyperkkl import cache as C
from hyperkkl import config as Cf
from hyperkkl import data as Dm
from hyperkkl import design as D
from hyperkkl import evaluate as Ev
from hyperkkl import job as J
from hyperkkl import metrics as M
from hyperkkl import sched as Sch
from hyperkkl import signals as S
from hyperkkl import tree as Tr
from hyperkkl.__main__ import main as cli
from hyperkkl.nets import BaseMap
from hyperkkl.observers import Observer
from hyperkkl.systems import Plant, build_plant

ROOT = Path(__file__).resolve().parent.parent
FORK = mp.get_context("fork")


# ===========================================================================
# metrics.episode_metrics
# ===========================================================================

def _manual_metrics(x, xh, sigma, settle):
    """Every field of episode_metrics from its written definition, in plain numpy loops."""
    x, xh, s = np.asarray(x, float), np.asarray(xh, float), np.asarray(sigma, float)
    E, N1, nx = x.shape
    out = {k: np.zeros(E) for k in ["nrmse", "nrmse_transient", "max_nerr", "t_conv_steps"]
           + [f"{p}_x{j + 1}" for p in ("rmse", "mae", "nrmse") for j in range(nx)]}
    out["diverged"] = np.zeros(E, bool)
    for e in range(E):
        en = np.zeros((N1, nx))
        div = False
        for k in range(N1):
            for j in range(nx):
                v = (xh[e, k, j] - x[e, k, j]) / s[j]
                if not np.isfinite(xh[e, k, j]) or abs(v) > 100:
                    div = True
                en[k, j] = 100.0 if np.isnan(v) else min(max(v, -100.0), 100.0)
        out["diverged"][e] = div
        st, tr = en[settle:], en[:settle]
        out["nrmse"][e] = math.sqrt((st ** 2).mean())
        out["nrmse_transient"][e] = math.sqrt((tr ** 2).mean())
        out["max_nerr"][e] = max(math.sqrt((st[k] ** 2).mean()) for k in range(len(st)))
        for j in range(nx):
            out[f"rmse_x{j + 1}"][e] = math.sqrt(((st[:, j] * s[j]) ** 2).mean())
            out[f"mae_x{j + 1}"][e] = np.abs(st[:, j] * s[j]).mean()
            out[f"nrmse_x{j + 1}"][e] = math.sqrt((st[:, j] ** 2).mean())
        ne = [math.sqrt((en[k] ** 2).mean()) for k in range(N1)]
        above = [k for k in range(N1) if ne[k] > 0.05]
        out["t_conv_steps"][e] = (max(above) + 1) if above else 0
    return out


def _episodes(N1=41, settle=20):
    sigma = np.array([0.5, 2.0])
    rng = np.random.RandomState(0)
    x = rng.randn(9, N1, 2)
    e = 0.05 * rng.randn(9, N1, 2) * sigma
    k = np.arange(N1)
    e[1] = np.exp(-k / 5.0)[:, None] * sigma           # decays: t_conv = 15
    e[2, 3, 0] = np.nan                                 # NaN in the transient
    e[3, 30, 1] = np.inf                                # inf in the steady window
    e[4, 10, 0] = 150 * sigma[0]                        # > 100 sigma once (transient)
    e[5] = 0.0
    e[5, 25, 1] = 99 * sigma[1]                         # 99 sigma: large but not diverged
    e[6, settle:] = np.nan                              # whole steady window NaN
    e[7] = 0.0
    e[7, -1, 0] = 1.0 * sigma[0]                        # above the band at the last sample
    e[8] = 0.0                                          # perfect
    return x, x + e, sigma, settle


def test_episode_metrics_match_manual_definitions():
    x, xh, sigma, settle = _episodes()
    got = M.episode_metrics(torch.tensor(x), torch.tensor(xh), sigma, settle)
    ref = _manual_metrics(x, xh, sigma, settle)
    assert set(got) == set(ref)
    for k in ref:
        assert np.allclose(got[k], ref[k], rtol=1e-5, atol=1e-6), (k, got[k], ref[k])
    assert got["diverged"].tolist() == [False, False, True, True, True, False, True, False, False]
    assert got["t_conv_steps"][1] == 15 and got["t_conv_steps"][7] == 41 and got["t_conv_steps"][8] == 0
    # a NaN estimate is NOT a perfect estimate in any metric (it used to give rmse/mae = 0)
    assert got["nrmse"][6] == 100.0
    assert np.allclose([got["rmse_x1"][6], got["rmse_x2"][6]], 100 * sigma)
    assert np.allclose([got["mae_x1"][6], got["mae_x2"][6]], 100 * sigma)
    # per-state rmse is exactly nrmse_x * sigma for every episode (one clipping convention)
    for j in range(2):
        assert np.allclose(got[f"rmse_x{j + 1}"], got[f"nrmse_x{j + 1}"] * sigma[j], rtol=1e-6)


def test_t_conv_definition():
    N1 = 50
    x = torch.zeros(3, N1, 1)
    xh = torch.full((3, N1, 1), 0.01)
    xh[0, :10] = 1.0                       # converges at 10
    xh[1, :10] = 1.0
    xh[1, 30] = 1.0                        # re-excursion: must re-converge after 30
    m = M.episode_metrics(x, xh, [1.0], 25)
    assert m["t_conv_steps"].tolist() == [10, 31, 0]


def test_episode_metrics_settle_zero_and_dtype():
    x = torch.zeros(2, 11, 2)
    m = M.episode_metrics(x, x + 0.1, [1.0, 1.0], 0)
    assert np.allclose(m["nrmse"], 0.1) and np.all(m["nrmse_transient"] == 0.0)
    assert m["diverged"].dtype == bool and m["nrmse"].dtype == np.float32


# ===========================================================================
# bootstrap / hierarchical / paired_relative
# ===========================================================================

def _hier_ref(per_seed, n_boot=2000, seed=0, alpha=0.05):
    """The original (loop) two-level bootstrap, kept as the reference implementation."""
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in per_seed]
    groups = [g for g in groups if len(g)]
    point = float(np.mean([g.mean() for g in groups]))
    rng = np.random.RandomState(seed)
    est = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(groups), len(groups))
        est[b] = np.mean([groups[i][rng.randint(0, len(groups[i]), len(groups[i]))].mean() for i in pick])
    return point, np.percentile(est, 100 * alpha / 2), np.percentile(est, 100 * (1 - alpha / 2))


def _paired_ref(a_per_seed, b_per_seed, n_boot=2000, seed=0):
    pairs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in zip(a_per_seed, b_per_seed)]
    rng = np.random.RandomState(seed)

    def rel(sel):
        ma = np.mean([pairs[i][0][j].mean() for i, j in sel])
        mb = np.mean([pairs[i][1][j].mean() for i, j in sel])
        return (ma - mb) / max(mb, 1e-12)

    point = rel([(i, np.arange(len(p[0]))) for i, p in enumerate(pairs)])
    est = [rel([(i, rng.randint(0, len(pairs[i][0]), len(pairs[i][0])))
                for i in rng.randint(0, len(pairs), len(pairs))]) for _ in range(n_boot)]
    return point, *np.percentile(est, [2.5, 97.5])


def test_bootstrap_edge_cases_and_coverage():
    assert all(np.isnan(M.bootstrap([])))
    assert M.bootstrap([np.nan, 3.0]) == (3.0, 3.0, 3.0)          # non-finite dropped
    rng = np.random.RandomState(0)
    R, hit = 400, 0
    for r in range(R):
        mu, lo, hi = M.bootstrap(1.0 + 0.5 * rng.randn(64), n_boot=1000, seed=r)
        hit += lo <= 1.0 <= hi
    print(f"\n[coverage] bootstrap n=64: {hit / R:.3f}")
    assert 0.91 <= hit / R <= 0.98


def test_hierarchical_point_is_mean_of_seed_means_and_matches_reference():
    g = [np.array([1.0, 1.0, 1.0, 1.0]), np.array([3.0]), np.array([np.nan, 2.0])]
    assert M.hierarchical(g)[0] == pytest.approx((1 + 3 + 2) / 3)        # not the pooled mean 1.5
    assert all(np.isnan(M.hierarchical([[], [np.nan]])))
    assert M.hierarchical([[0.7]]) == (0.7, 0.7, 0.7)
    # >= 2 seeds: Student-t interval on the seed means; 1 seed: episode bootstrap
    from scipy.stats import t as student_t
    rng = np.random.RandomState(1)
    for S_, n in [(3, 64), (5, 8), (5, 64)]:
        groups = [0.3 + 0.1 * rng.randn() + 0.2 * rng.rand(n) for _ in range(S_)]
        m = np.array([g.mean() for g in groups])
        h = student_t.ppf(0.975, S_ - 1) * m.std(ddof=1) / np.sqrt(S_)
        assert M.hierarchical(groups) == pytest.approx((m.mean(), m.mean() - h, m.mean() + h))
    one = [0.3 + 0.2 * rng.rand(30)]
    assert M.hierarchical(one, n_boot=4000) == pytest.approx(M.bootstrap(one[0], 4000))


def _coverage_hier(S_, tau, sd=0.5, n=32, R=300, n_boot=600, seed=0):
    rng = np.random.RandomState(seed)
    hit = 0
    for r in range(R):
        b = tau * rng.randn(S_)
        _, lo, hi = M.hierarchical([1.0 + b[s] + sd * rng.randn(n) for s in range(S_)], n_boot=n_boot, seed=r)
        hit += lo <= 1.0 <= hi
    return hit / R


def test_hierarchical_coverage_two_level():
    c10 = _coverage_hier(10, 0.2)
    c0 = _coverage_hier(5, 0.0)
    print(f"\n[coverage] hierarchical S=10 tau=0.2 sd=0.5: {c10:.3f}; S=5 tau=0 (no seed effect): {c0:.3f}")
    assert c10 >= 0.90
    assert c0 >= 0.93                      # conservative when there is no between-seed variance


@pytest.mark.parametrize("S_", [3, 5])
def test_hierarchical_coverage_few_seeds(S_):
    c = _coverage_hier(S_, 0.2)
    print(f"\n[coverage] hierarchical S={S_} tau=0.2 sd=0.5: {c:.3f}")
    assert c >= 0.93


def _coverage_paired(S_, tau, n=32, R=300, n_boot=600, seed=0):
    true = 0.9 * math.exp(0.05 ** 2 / 2) - 1
    rng = np.random.RandomState(seed)
    hit = 0
    for r in range(R):
        d = tau * rng.randn(S_)
        A, B = [], []
        for s in range(S_):
            b = np.exp(0.3 * rng.randn(n))
            A.append(b * (0.9 + d[s]) * np.exp(0.05 * rng.randn(n)))
            B.append(b)
        ci = M.paired_relative(A, B, n_boot=n_boot, seed=r)["ci"]
        hit += ci[0] <= true <= ci[1]
    return hit / R


def test_paired_relative_sign_pairing_and_strictness():
    b = [np.linspace(1, 2, 10) + s for s in range(3)]
    d = M.paired_relative([0.5 * x for x in b], b)
    assert d["rel"] == pytest.approx(-0.5) and d["ci"] == pytest.approx([-0.5, -0.5])
    assert d["better"] and not d["worse"] and d["n_seeds"] == 3
    d = M.paired_relative([2 * x for x in b], b)
    assert d["rel"] == pytest.approx(1.0) and d["worse"] and not d["better"]
    # >= 2 seeds: t-interval on the per-seed relative changes
    from scipy.stats import t as student_t
    rng = np.random.RandomState(0)
    B = [np.exp(0.5 * rng.randn(64)) for _ in range(5)]
    A = [x * 0.95 * (1 + 0.01 * rng.randn(64)) for x in B]
    r = np.array([(a.mean() - b_.mean()) / b_.mean() for a, b_ in zip(A, B)])
    h = student_t.ppf(0.975, 4) * r.std(ddof=1) / np.sqrt(5)
    d = M.paired_relative(A, B)
    assert d["ci_method"] == "t-over-seeds" and d["better"]
    assert d["rel"] == pytest.approx(r.mean()) and d["ci"] == pytest.approx([r.mean() - h, r.mean() + h])
    # 1 seed: paired episode bootstrap; pairing (same resampled episodes) narrows the CI
    paired = M.paired_relative(A[:1], B[:1])
    shuffled = M.paired_relative([rng.permutation(A[0])], B[:1])
    assert paired["ci_method"].startswith("bootstrap") and paired["better"]
    assert paired["rel"] == pytest.approx(shuffled["rel"])
    w = lambda q: q["ci"][1] - q["ci"][0]
    print(f"\n[paired, 1 seed] CI width paired {w(paired):.4f} vs pairing destroyed {w(shuffled):.4f}")
    assert w(shuffled) > 3 * w(paired)
    # strict pairing: mismatched episodes or seeds raise; empty seeds (absent family) are skipped
    with pytest.raises(ValueError):
        M.paired_relative([np.ones(4)], [np.ones(5)])
    with pytest.raises(ValueError):
        M.paired_relative([np.ones(4)] * 2, [np.ones(4)] * 3)
    assert M.paired_relative([[], np.ones(3)], [[], 2 * np.ones(3)])["n_seeds"] == 1
    assert M.paired_relative([[]], [[]]) == {}


def test_paired_relative_coverage():
    c10, c0 = _coverage_paired(10, 0.02), _coverage_paired(5, 0.0)
    print(f"\n[coverage] paired S=10 tau=0.02: {c10:.3f}; S=5 tau=0: {c0:.3f}")
    assert c10 >= 0.88 and c0 >= 0.93


@pytest.mark.parametrize("S_", [3, 5])
def test_paired_relative_coverage_few_seeds(S_):
    c = _coverage_paired(S_, 0.05)
    print(f"\n[coverage] paired S={S_} tau=0.05: {c:.3f}")
    assert c >= 0.93


def test_bound_terms_arithmetic():
    b = M.bound_terms(0.01, 0.2, 3.0, 1.0, 1.5, l_phi=0.5, l_c=2.0, delta_c=0.1)
    assert b["margin"] == pytest.approx(1.0) and b["drive"] == pytest.approx(0.2)
    assert b["context_term"] == pytest.approx(0.2) and b["bound"] == pytest.approx(0.01 + 3 * 0.2 + 0.2)
    b = M.bound_terms(0.01, 0.2, 3.0, 2.0, 1.0, l_phi=0.5)                # kappa l_phi = alpha
    assert b["margin"] == 0.0 and b["drive"] == math.inf and b["bound"] == math.inf


# ===========================================================================
# config
# ===========================================================================

def _leaves(d, prefix):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def _reordered(d):
    return {k: _reordered(v) if isinstance(v, dict) else v for k, v in reversed(list(d.items()))}


def _perturb(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, str):
        return v + "_x"
    if isinstance(v, list):
        return v + [v[0]] if v else ["x"]
    if v is None:
        return 1
    return {"x": 1}


def test_load_precedence_and_merge():
    base = yaml.safe_load(open(Cf.CONFIG_DIR / "base.yaml"))
    sysc = yaml.safe_load(open(Cf.CONFIG_DIR / "systems" / "vdp.yaml"))
    cfg = Cf.load("vdp")
    assert cfg["system"] == {**base["system"], **sysc} and cfg["phase2"] == base["phase2"]   # base.yaml system defaults (knowledge) under the system YAML
    cfg = Cf.load("vdp", [{"phase2.lr": 1.0, "system.alpha": 9.0}, {"phase2.lr": 2.0}])
    assert cfg["phase2"]["lr"] == 2.0 and cfg["system"]["alpha"] == 9.0               # later wins
    assert cfg["phase2"]["steps"] == base["phase2"]["steps"]
    ov = {"phase2": {"context": {"kind": "gru"}}, "eval.tiers": ["shape_ood"]}
    cfg = Cf.load("vdp", ov)
    assert cfg["phase2"]["context"] == {**base["phase2"]["context"], "kind": "gru"}  # deep merge
    assert cfg["eval"]["tiers"] == ["shape_ood"]                                     # lists replace
    assert ov == {"phase2": {"context": {"kind": "gru"}}, "eval.tiers": ["shape_ood"]}   # not mutated
    assert Cf.apply_dotted({}, {"a.b.c": 1}) == {"a": {"b": {"c": 1}}}
    a = {"x": {"y": [1]}}
    m = Cf.deep_merge(a, {})
    m["x"]["y"].append(2)
    assert a == {"x": {"y": [1]}}                                                    # no aliasing


def test_parse_value():
    assert Cf.parse_value("1e-3") == 0.001 and isinstance(Cf.parse_value("1e-3"), float)
    assert Cf.parse_value("-2E+4") == -20000.0 and Cf.parse_value("1.0e-3") == 0.001
    assert Cf.parse_value("[1e-3, 2]") == [0.001, 2]
    assert Cf.parse_value("140000") == 140000 and Cf.parse_value("true") is True
    assert Cf.parse_value("sfm") == "sfm" and Cf.parse_value("[a, b]") == ["a", "b"]
    assert Cf.parse_value("e5") == "e5"


def test_digest_stability_and_sensitivity():
    cfg = Cf.load("vdp")
    secs = ("system", "data", "phase1", "phase2", "eval", "conditions")
    d0 = Cf.digest(cfg, *secs)
    assert Cf.digest(Cf.load("vdp"), *secs) == d0
    assert Cf.digest(_reordered(cfg), *secs) == d0                     # key order irrelevant
    assert Cf.digest(cfg, *reversed(secs)) == d0                       # section order irrelevant
    n = 0
    for sec in secs:
        for key, v in _leaves(cfg[sec], f"{sec}."):
            assert Cf.digest(Cf.apply_dotted(cfg, {key: _perturb(v)}), *secs) != d0, key
            n += 1
    print(f"\n[digest] {n} leaves perturbed one at a time: every digest changed")
    assert Cf.digest(Cf.apply_dotted(cfg, {"nlox.lr": 1.0}), *secs) == d0   # other sections
    assert Cf.digest(cfg, "system", extra=[0]) != Cf.digest(cfg, "system", extra=[1])
    # documented quirk: 1 and 1.0 hash differently (a cache miss, never a wrong hit)
    assert Cf.digest({"a": 1}, "a") != Cf.digest({"a": 1.0}, "a")


# ===========================================================================
# experiment tree (redirected into tmp_path)
# ===========================================================================

@pytest.fixture
def exp(tmp_path, monkeypatch):
    E = tmp_path / "experiments"
    monkeypatch.setattr(Tr, "EXP", E)
    monkeypatch.setattr(Tr, "TREE", E / "tree")
    monkeypatch.setattr(Tr, "QUEUE", E / "queue")
    monkeypatch.setattr(Sch, "PENDING", E / "queue" / "pending")
    monkeypatch.setattr(Sch, "RUNNING", E / "queue" / "running")
    monkeypatch.setattr(Sch, "FAILED", E / "queue" / "failed")
    monkeypatch.setattr(Sch, "REPORT_ASYNC", False)      # synchronous: never touch the real tree
    for d in ("pending", "running", "failed"):            # the worker creates these at start
        (E / "queue" / d).mkdir(parents=True)
    return E


def _mk_tree():
    Tr.new_node("R0", None, "root", set={"phase2.lr": 1.0, "eval.n_episodes": 8},
                methods=["autonomous", "lora"], baselines=["ekf"], systems=["vdp", "duffing"],
                seeds=[0], compare={"reference": "autonomous", "condition": "clean"})
    Tr.new_node("A", "R0", "child", set={"phase2.lr": 2.0, "phase2.steps": 5}, seeds=[0, 1, 2], priority=3)
    Tr.new_node("B", "R0", "sibling", set={"phase1.pde": False}, seeds=[0, 1])
    Tr.new_node("A1", "A", "grandchild", set={"phase2.steps": 7}, methods=["lora", "lora@gru"])


FAMS = ["f1", "f1", "f2", "f2", "f2", "f1"]


def _payload(node, system, seed, vals, code="c0", cfg=None, diverged=None, fams=FAMS):
    methods = {m: {"nrmse": list(map(float, v)), "nrmse_transient": [2.0 * float(x) for x in v],
                   "diverged": list((diverged or {}).get(m, [False] * len(v)))} for m, v in vals.items()}
    return {"node": node, "system": system, "seed": seed, "code": code, "wall_s": 120.0,
            "phase1": {"diag": {"eps_dec": 0.01 * (seed + 1)}, "wall_s": 1.0},
            "train": {m: {"n_params": 1000 + i} for i, m in enumerate(vals)},
            "cost": {m: {"us_cpu_b1": 10.0, "us_gpu_b64": 1.0} for m in vals},
            "results": {t: {"clean": {"families": list(fams), "methods": methods}} for t in ("shape_ood", "sequence_ood")},
            "state_coverage": {"shape_ood": {"outside_box": 0.01 * (seed + 1), "outside_knn": 0.02},
                               "sequence_ood": {"outside_box": 0.0, "outside_knn": 0.04}},
            "config": cfg if cfg is not None else Cf.load(system, Tr.resolved_overrides(node)), "scales": {}}


def _write(node, system, seed, payload):
    d = Tr.run_dir(node, system, seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(payload))


def _ref_vals(seed):
    return np.array([1 + 0.1 * seed + 0.01 * i for i in range(6)])


def _fill_A(code_of_seed=lambda k: "c0", cfg_of_seed=lambda k: None):
    for k in range(3):
        r = _ref_vals(k)
        _write("A", "vdp", k, _payload("A", "vdp", k, {"autonomous": r, "lora": 0.5 * r}, code_of_seed(k),
                                       cfg_of_seed(k), diverged={"lora": [k == 0] + [False] * 5}))
        _write("A", "duffing", k, _payload("A", "duffing", k, {"autonomous": r, "lora": 1.2 * r}, code_of_seed(k)))


def test_resolved_overrides_inherited_jobs(exp):
    _mk_tree()
    ov = Tr.resolved_overrides("A1")
    assert ov == [{"phase2.lr": 1.0, "eval.n_episodes": 8, "methods": ["autonomous", "lora"], "baselines": ["ekf"]},
                  {"phase2.lr": 2.0, "phase2.steps": 5},
                  {"phase2.steps": 7, "methods": ["lora", "lora@gru"]}]                    # root first
    cfg = Cf.load("vdp", ov)
    assert cfg["phase2"]["lr"] == 2.0 and cfg["phase2"]["steps"] == 7                    # child wins
    assert cfg["methods"] == ["lora", "lora@gru"] and cfg["baselines"] == ["ekf"] and cfg["eval"]["n_episodes"] == 8
    assert Tr.inherited("A1", "seeds") == [0, 1, 2] and Tr.inherited("A1", "systems") == ["vdp", "duffing"]
    assert Tr.inherited("A1", "compare")["reference"] == "autonomous"
    assert Tr.inherited("A1", "nothing", 5) == 5 and Tr.inherited("R0", "seeds") == [0]
    assert Tr.jobs_of("A1") == [("vdp", 0), ("vdp", 1), ("vdp", 2), ("duffing", 0), ("duffing", 1), ("duffing", 2)]
    assert Tr.jobs_of("B") == [("vdp", 0), ("vdp", 1), ("duffing", 0), ("duffing", 1)]
    with pytest.raises(FileExistsError):
        Tr.new_node("A", "R0", "dup")
    with pytest.raises(KeyError):
        Tr.new_node("Z", "nope", "orphan")


def test_queue_idempotent_and_status_transitions(exp):
    _mk_tree()
    pend = lambda: sorted(p.name for p in (Tr.QUEUE / "pending").glob("*.json"))
    assert Tr.refresh_status("A") == "planned"
    assert Tr.queue("A") == 6 and len(pend()) == 6
    assert Tr.queue("A") == 0 and len(pend()) == 6                     # no rewrite, no duplicate
    assert Tr.refresh_status("A") == "queued"
    job, path = Sch._next_job()
    assert job["node"] == "A" and path.parent == Sch.RUNNING
    assert Tr.queue("A") == 0                                            # running job not re-queued
    assert Tr.refresh_status("A") == "running"
    # five finish, one is killed without a traceback
    jobs = [(job, path)] + [Sch._next_job() for _ in range(5)]
    for (j, p), rc in zip(jobs, [0, 0, 0, 0, 0, -9]):
        if rc == 0:
            _write("A", j["system"], j["seed"], _payload("A", j["system"], j["seed"], {"autonomous": np.ones(6)}))
        Sch._finish(j, p, rc)
    pr = Tr.progress("A")
    assert pr == {"total": 6, "done": 5, "failed": 1, "running": 0, "pending": 0}
    assert Tr.load_node("A")["status"] == "failed"
    bad = jobs[-1][0]
    assert (Sch.FAILED / jobs[-1][1].name).exists()
    assert "exit code -9" in (Tr.run_dir("A", bad["system"], bad["seed"]) / "error.txt").read_text()
    assert (Tr.TREE / "A" / "report.md").exists() and (Tr.EXP / "TREE.md").exists()
    # re-queue: only the failed job, its error and failed/ entry are cleared
    assert Tr.queue("A") == 1 and pend() == [jobs[-1][1].name]
    assert not (Tr.run_dir("A", bad["system"], bad["seed"]) / "error.txt").exists()
    assert not (Sch.FAILED / jobs[-1][1].name).exists()
    j, p = Sch._next_job()
    _write("A", j["system"], j["seed"], _payload("A", j["system"], j["seed"], {"autonomous": np.ones(6)}))
    Sch._finish(j, p, 0)
    assert Tr.load_node("A")["status"] == "done"
    # --force re-queues finished jobs, and the node is NOT reported done while they run
    assert Tr.queue("A") == 0
    assert Tr.queue("A", force=True) == 6
    assert Tr.refresh_status("A") == "queued"
    Sch._next_job()
    assert Tr.refresh_status("A") == "running"


def test_summarize_report_headline_render_verdict(exp):
    _mk_tree()
    _fill_A()
    S_ = Tr.summarize("A")
    assert S_["references"] == ["autonomous"]
    row = S_["summary"]["vdp"]["shape_ood"]["clean"]
    assert S_["summary"]["vdp"]["sequence_ood"]["clean"] == row
    assert row["autonomous"]["all"][0] == pytest.approx(1.125) and row["autonomous"]["all"][3] == 3
    assert row["autonomous"]["f1"][0] == pytest.approx(1.12) and row["autonomous"]["f2"][0] == pytest.approx(1.13)
    assert row["lora"]["all"][0] == pytest.approx(0.5625)
    seeds_f1 = [0.5 * _ref_vals(k)[[0, 1, 5]] for k in range(3)]
    assert row["lora"]["f1"][:3] == pytest.approx(list(M.hierarchical(seeds_f1, n_boot=1000)))
    assert row["lora"]["_diverged"] == pytest.approx(1 / 18) and row["autonomous"]["_diverged"] == 0.0
    for fam in ("f1", "f2", "all"):
        pv = S_["paired"]["vdp"]["shape_ood"]["clean"]["autonomous"]["lora"][fam]
        assert pv["rel"] == pytest.approx(-0.5) and pv["ci"] == pytest.approx([-0.5, -0.5]) and pv["better"]
        pd = S_["paired"]["duffing"]["sequence_ood"]["clean"]["autonomous"]["lora"][fam]
        assert pd["rel"] == pytest.approx(0.2) and pd["worse"]
    ex = S_["extra"]["vdp"]
    assert ex["params"] == {"autonomous": 1000, "lora": 1001} and ex["phase1"]["eps_dec"] == pytest.approx(0.02)
    assert ex["code"] == ["c0"] and len(ex["config"]) == 1 and ex["wall_s"] == 120.0

    Tr.refresh_status("A")
    md = Tr.report("A").read_text()
    assert "WARNING" not in md
    assert "## Scoreboard (shape_ood / clean, all families)" in md
    assert "## Scoreboard (sequence_ood / clean, all families)" in md
    assert md.index("## Scoreboard (shape_ood") < md.index("## Scoreboard (sequence_ood") < md.index("## vdp")
    assert ex["state_coverage"]["shape_ood"]["outside_box"] == pytest.approx(0.02)       # mean over seeds
    assert ("- state coverage shape_ood: 2.0% of eval states outside the training box, 2.0% beyond "
            "the training kNN q99") in md
    assert "- state coverage sequence_ood: 0.0% of eval states outside the training box, 4.0%" in md
    assert "| lora | 0.5625 | 1.3500 |" in md                                  # scoreboard means
    assert "| lora | 2/0 (-50%*) | 0/2 (+20%*) |" in md                         # scoreboard paired
    assert "| lora | **-50.0% [-50, -50]** | **-50.0% [-50, -50]** | **-50.0% [-50, -50]** |" in md
    lora_row = [l for l in md.splitlines() if l.startswith("| lora | **0.5600 [")][0]          # vdp: f1 column
    assert "| **0.5625 [" in lora_row and lora_row.endswith("| 0.06 | 1001 | 10.00 |")      # all, div, params, µs
    sj = json.loads((Tr.TREE / "A" / "summary.json").read_text())
    assert list(sj) == ["nrmse"]
    hl = Tr.headline("A")
    assert hl == f"vdp: lora {0.5625:.3g} < autonomous {1.125:.3g}; duffing: autonomous {1.125:.3g} < lora {1.35:.3g}"
    Tr.set_verdict("A", "supported", "lora halves the error on vdp")
    with pytest.raises(AssertionError):
        Tr.set_verdict("A", "great", "x")
    Tr.report("A")
    tree_md = Tr.render().read_text()
    block = tree_md[tree_md.index("```") + 4: tree_md.index("```", tree_md.index("```") + 3)]
    assert block.splitlines() == ["○ R0  [0/2] root", "├── ○ B  [0/4] sibling",
                                  "└── ● A  [6/6] child ✔", "    └── ○ A1  [0/6] grandchild"]
    assert "- **verdict (supported)**: lora halves the error on vdp" in tree_md
    assert f"- headline (shape_ood): {hl}" in tree_md
    assert f"- headline (sequence_ood): {Tr.headline('A', 'sequence_ood')}" in tree_md
    assert "verdict supported" in (Tr.EXP / "JOURNAL.md").read_text()


def test_report_flags_mixed_code_and_configs(exp):
    _mk_tree()
    cfg1 = Cf.load("vdp", Tr.resolved_overrides("A") + [{"phase2.lr": 9.0}])
    _fill_A(code_of_seed=lambda k: "c1" if k == 2 else "c0", cfg_of_seed=lambda k: cfg1 if k == 1 else None)
    S_ = Tr.summarize("A")
    assert S_["extra"]["vdp"]["code"] == ["c0", "c1"] and len(S_["extra"]["vdp"]["config"]) == 2
    md = Tr.report("A").read_text()
    assert "WARNING: these runs come from 2 code versions (c0, c1); seeds of vdp ran with different configs" in md


def test_report_survives_nan_and_headline_orders_nan_last(exp):
    _mk_tree()
    for k in range(3):
        r = _ref_vals(k)
        _write("A", "vdp", k, _payload("A", "vdp", k, {"autonomous": r, "bad": np.full(6, np.nan)}))
    S_ = Tr.summarize("A")
    assert np.isnan(S_["summary"]["vdp"]["shape_ood"]["clean"]["bad"]["all"][0])
    md = Tr.report("A").read_text()
    assert "| bad | – [–, –] |" in md and "nan" not in md.lower().replace("nanosec", "")
    assert Tr.headline("A").startswith("vdp: autonomous")


def test_cross_compare_pairing_and_mismatch_errors(exp):
    _mk_tree()
    _fill_A()
    for k in range(2):                                           # B: same episodes, seeds 0, 1
        r = _ref_vals(k)
        _write("B", "vdp", k, _payload("B", "vdp", k, {"lora": 0.4 * r, "lora@window:window=20": 0.45 * r},
                                       cfg=Cf.load("vdp", Tr.resolved_overrides("A"))))
    out = Tr.cross_compare("B:lora", "A:lora")
    assert list(out) == ["vdp"] and out["vdp"]["seeds"] == [0, 1] and out["vdp"]["same_code"]
    assert out["vdp"]["rel"] == pytest.approx(0.4 / 0.5 - 1)
    assert out["vdp"]["a"] == pytest.approx(0.4 * np.mean([_ref_vals(k).mean() for k in range(2)]))
    out = Tr.cross_compare("B:lora@window:window=20", "A:lora")      # ':' inside a method name
    assert out["vdp"]["rel"] == pytest.approx(0.45 / 0.5 - 1)
    # different families -> different episodes
    _write("B", "vdp", 1, _payload("B", "vdp", 1, {"lora": np.ones(6)}, fams=FAMS[::-1]))
    with pytest.raises(ValueError, match="episodes differ"):
        Tr.cross_compare("B:lora", "A:lora")
    # same family list, different plant (alpha_scale changes dt, hence the episodes)
    bad = Cf.load("vdp", Tr.resolved_overrides("A") + [{"system.alpha_scale": 0.5}])
    _write("B", "vdp", 1, _payload("B", "vdp", 1, {"lora": np.ones(6)}, cfg=bad))
    with pytest.raises(ValueError, match="system"):
        Tr.cross_compare("B:lora", "A:lora")
    # different training config, same episodes: allowed, and a code mismatch is reported
    ok = Cf.load("vdp", Tr.resolved_overrides("A") + [{"phase1.pde": False}])
    _write("B", "vdp", 1, _payload("B", "vdp", 1, {"lora": np.ones(6)}, cfg=ok, code="c9"))
    assert Tr.cross_compare("B:lora", "A:lora")["vdp"]["same_code"] is False


def test_cli_new_queue_status_compare(exp, capsys):
    _mk_tree()
    cli(["new", "X", "--parent", "R0", "--title", "cli node", "--set", "phase2.lr=1e-3",
         "phase2.context.kind=gru", "--seeds", "0", "1", "--reference", "lora"])
    n = Tr.load_node("X")
    assert n["set"] == {"phase2.lr": 0.001, "phase2.context.kind": "gru"} and n["seeds"] == [0, 1]
    assert n["compare"] == {"reference": "lora"} and n["priority"] == 0
    cli(["queue", "X"])
    assert len(list((Tr.QUEUE / "pending").glob("X__*.json"))) == 4
    cli(["status"])
    cli(["tree"])
    out = capsys.readouterr().out
    assert "X                            queued" in out and "◔ X  [0/4] cli node" in out
    _fill_A()
    for k in range(2):
        _write("X", "vdp", k, _payload("X", "vdp", k, {"lora": 0.25 * _ref_vals(k)},
                                       cfg=Cf.load("vdp", Tr.resolved_overrides("A"))))
    cli(["compare", "X:lora", "A:lora"])
    out = capsys.readouterr().out
    assert "change -50.0%" in out and "better" in out and "seeds [0, 1]" in out


# ===========================================================================
# scheduler
# ===========================================================================

def _dead_pid():
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def _pending_file(name, prio, queued, **extra):
    Sch.PENDING.mkdir(parents=True, exist_ok=True)
    node, system, seed = name.split("__")
    (Sch.PENDING / f"{name}.json").write_text(json.dumps(
        {"node": node, "system": system, "seed": int(seed), "priority": prio, "queued": queued, **extra}))


def test_next_job_priority_order(exp):
    Sch.RUNNING.mkdir(parents=True, exist_ok=True)
    _pending_file("X__vdp__0", 0, "2026-10-24T10:00:00")
    _pending_file("Y__vdp__0", 5, "2026-10-24T10:02:00")
    _pending_file("Z__vdp__0", 5, "2026-10-24T10:01:00")
    _pending_file("W__vdp__0", 10, "2026-10-24T10:03:00")
    (Sch.PENDING / "corrupt.json").write_text("{not json")
    order = []
    while True:
        j, p = Sch._next_job()
        if j is None:
            break
        order.append(j["node"])
        assert p.exists() and p.parent == Sch.RUNNING
    assert order == ["W", "Z", "Y", "X"]                      # priority desc, then FIFO
    assert [p.name for p in Sch.PENDING.glob("*.json")] == ["corrupt.json"]


def _claimer(out, t0):
    while time.time() < t0:
        pass
    names = []
    while True:
        j, p = Sch._next_job()
        if j is None:
            break
        names.append(p.name)
    Path(out).write_text("\n".join(names))


def test_claim_is_atomic_across_processes(exp):
    Sch.RUNNING.mkdir(parents=True, exist_ok=True)
    for i in range(300):
        _pending_file(f"N__vdp__{i}", i % 3, "")
    outs = [exp / f"claims{i}.txt" for i in range(6)]
    t0 = time.time() + 1.0
    ps = [FORK.Process(target=_claimer, args=(str(o), t0)) for o in outs]
    [p.start() for p in ps]
    [p.join(60) for p in ps]
    claimed = [n for o in outs for n in o.read_text().split()]
    print(f"\n[claim] per-process claims: {[len(o.read_text().split()) for o in outs]}")
    assert len(claimed) == 300 and len(set(claimed)) == 300
    assert sum(bool(o.read_text()) for o in outs) >= 2        # the processes really competed
    assert not list(Sch.PENDING.glob("*.json")) and len(list(Sch.RUNNING.glob("*.json"))) == 300


def test_finish_survives_bookkeeping_errors(exp, capsys):
    Sch.RUNNING.mkdir(parents=True, exist_ok=True)
    p = Sch.RUNNING / "ghost__vdp__0.json"
    p.write_text("{}")
    Sch._finish({"node": "ghost", "system": "vdp", "seed": 0}, p, 0)      # node does not exist
    assert "bookkeeping for ghost failed" in capsys.readouterr().out and not p.exists()


def test_alive():
    assert Sch._alive(os.getpid()) and not Sch._alive(_dead_pid())
    assert not Sch._alive(None) and not Sch._alive(0) and not Sch._alive("x")


def test_worker_restart_recovery(exp):
    _mk_tree()
    Sch.RUNNING.mkdir(parents=True, exist_ok=True)
    now = time.time()
    files = {"dead": ("A", "vdp", 0, _dead_pid()), "alive": ("A", "vdp", 1, os.getpid()),
             "finished": ("A", "vdp", 2, _dead_pid()), "legacy": ("A", "duffing", 0, None)}
    for key, (node, system, seed, pid) in files.items():
        j = {"node": node, "system": system, "seed": seed, "priority": 3, "queued": ""}
        if pid is not None:
            j["pid"] = pid
        f = Sch.RUNNING / f"{node}__{system}__{seed}.json"
        f.write_text(json.dumps(j))
        os.utime(f, (now - 100, now - 100))
    _write("A", "vdp", 2, _payload("A", "vdp", 2, {"autonomous": np.ones(6)}))     # newer than claim
    (exp / "STOP").touch()
    Sch.worker([0], per_gpu=1, poll=0.0)                   # STOP + idle: recovery only, then exit
    pend = sorted(p.name for p in Sch.PENDING.glob("*.json"))
    run = sorted(p.name for p in Sch.RUNNING.glob("*.json"))
    assert pend == ["A__duffing__0.json", "A__vdp__0.json"]          # dead and legacy: re-queued
    assert run == ["A__vdp__1.json"]                                  # alive: left alone
    assert not list(Sch.FAILED.glob("*.json"))                        # finished: not re-run


class _FakeProc:
    started = []

    def __init__(self, args, cwd=None, env=None, stdout=None, stderr=None):
        self.node, self.system = args[args.index("--node") + 1], args[args.index("--system") + 1]
        self.seed = int(args[args.index("--seed") + 1])
        self.pid, self.returncode, self.n = 4_000_000 + len(self.started), None, 0
        _FakeProc.started.append((self.node, self.system, self.seed, env["CUDA_VISIBLE_DEVICES"]))

    def poll(self):
        self.n += 1
        if self.n < 2:
            return None
        if (self.system, self.seed) == ("duffing", 2):
            self.returncode = -9                               # killed: writes nothing
        else:
            _write(self.node, self.system, self.seed,
                   _payload(self.node, self.system, self.seed, {"autonomous": _ref_vals(self.seed)}))
            self.returncode = 0
        return self.returncode


def test_worker_loop_with_fake_jobs(exp, monkeypatch):
    _mk_tree()
    assert Tr.queue("A") == 6
    _FakeProc.started = []
    monkeypatch.setattr(Sch.subprocess, "Popen", _FakeProc)
    pid_seen = {}
    real_sleep, n = time.sleep, [0]

    def fake_sleep(_):
        n[0] += 1
        assert n[0] < 100, "worker did not drain the queue"
        for p in Sch.RUNNING.glob("*.json"):
            pid_seen[p.name] = json.loads(p.read_text()).get("pid")
        if not list(Sch.PENDING.glob("*.json")):
            (exp / "STOP").touch()

    monkeypatch.setattr(Sch.time, "sleep", fake_sleep)
    Sch.worker([0, 1], per_gpu=2, poll=0.0)
    assert [s[3] for s in _FakeProc.started[:4]] == ["0", "1", "0", "1"]    # interleaved slots
    assert len(_FakeProc.started) == 6 and len(set(s[:3] for s in _FakeProc.started)) == 6
    assert all(v is not None and v >= 4_000_000 for v in pid_seen.values()) and len(pid_seen) == 6
    assert not list(Sch.RUNNING.glob("*.json")) and not list(Sch.PENDING.glob("*.json"))
    assert [p.name for p in Sch.FAILED.glob("*.json")] == ["A__duffing__2.json"]
    assert Tr.progress("A") == {"total": 6, "done": 5, "failed": 1, "running": 0, "pending": 0}
    assert Tr.load_node("A")["status"] == "failed"
    assert (Tr.TREE / "A" / "report.md").exists() and (exp / "TREE.md").exists()


# ===========================================================================
# cache
# ===========================================================================

def _cache_user(path, cnt, t0, q, sleep_s):
    while time.time() < t0:
        pass

    def build():
        with open(cnt, "a") as f:
            f.write(f"{os.getpid()}\n")
        time.sleep(sleep_s)
        return {"pid": os.getpid()}

    q.put(C.cached(Path(path), build, poll_s=0.05)["pid"])


def _race(d, n=6, sleep_s=0.5):
    path, cnt = d / "a.pt", d / "count.txt"
    q = FORK.Queue()
    t0 = time.time() + 0.5
    ps = [FORK.Process(target=_cache_user, args=(str(path), str(cnt), t0, q, sleep_s)) for _ in range(n)]
    [p.start() for p in ps]
    [p.join(60) for p in ps]
    return [q.get(timeout=5) for _ in ps], cnt.read_text().split(), path


def test_cache_builds_once_with_concurrent_waiters(tmp_path):
    pids, builds, path = _race(tmp_path)
    assert len(builds) == 1 and len(set(pids)) == 1 and path.exists()
    assert torch.load(path, weights_only=False)["pid"] == pids[0]
    assert not (tmp_path / "a.pt.lock").exists()                       # removed once built


def test_cache_stale_lock_is_recovered_once(tmp_path):
    (tmp_path / "a.pt.lock").write_text(str(_dead_pid()))             # left by a killed job
    pids, builds, path = _race(tmp_path)
    print(f"\n[cache] stale lock, 6 concurrent waiters: {len(builds)} build(s)")
    assert len(builds) == 1 and len(set(pids)) == 1 and path.exists()


def test_cache_builder_killed_mid_build(tmp_path):
    path, cnt = tmp_path / "a.pt", tmp_path / "count.txt"
    q = FORK.Queue()
    builder = FORK.Process(target=_cache_user, args=(str(path), str(cnt), 0, q, 30.0))
    builder.start()
    for _ in range(200):
        if cnt.exists() and cnt.read_text():
            break
        time.sleep(0.05)
    waiters = [FORK.Process(target=_cache_user, args=(str(path), str(cnt), 0, q, 0.3)) for _ in range(3)]
    [w.start() for w in waiters]
    time.sleep(0.5)
    os.kill(builder.pid, signal.SIGKILL)
    builder.join()
    [w.join(60) for w in waiters]
    pids = [q.get(timeout=5) for _ in waiters]
    builds = cnt.read_text().split()
    assert len(builds) == 2 and str(builder.pid) == builds[0]       # killed one + exactly one more
    assert len(set(pids)) == 1 and str(pids[0]) == builds[1] and path.exists()


def test_cache_build_error_releases_lock(tmp_path):
    path = tmp_path / "b.pt"

    def boom():
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError):
        C.cached(path, boom)
    assert C.cached(path, lambda: 42) == 42 and C.cached(path, boom) == 42


# ===========================================================================
# job
# ===========================================================================

def test_method_cfg_parsing():
    cfg = Cf.load("vdp")
    ctx = cfg["phase2"]["context"]
    assert J.method_cfg(cfg, "lora") == ("lora", cfg) and J.method_cfg(cfg, "lora")[1] is cfg
    assert J.method_cfg(cfg, "autonomous") == ("none", cfg)
    k, mc = J.method_cfg(cfg, "lora@gru")
    assert k == "lora" and mc["phase2"]["context"] == {**ctx, "kind": "gru"} and cfg["phase2"]["context"] == ctx
    k, mc = J.method_cfg(cfg, "lora@window:window=20,dim=8")
    assert k == "lora" and mc["phase2"]["context"] == {**ctx, "kind": "window", "window": 20, "dim": 8}
    assert J.method_cfg(cfg, "latent@current")[1]["phase2"]["context"]["kind"] == "current"
    k, mc = J.method_cfg(cfg, "lora@sfm:hidden=1e2")
    assert mc["phase2"]["context"]["hidden"] == 100.0


def _keys(cfg, methods, seeds, system="vdp"):
    mh = C.module_hash(C.MODEL_MODULES)
    out = {}
    for s in seeds:
        for m in methods:
            kind, mc = J.method_cfg(cfg, m)
            out[(m, s)] = f"{system}_p2_{kind}_{Cf.digest(mc, 'system', 'data', 'phase1', 'phase2', extra=[s, kind, mh])}"
    return out


def test_cache_keys_distinct_per_method_context_seed():
    cfg = Cf.load("vdp")
    methods = ["autonomous", "latent", "concat", "film", "lora", "hyper", "lora@gru", "lora@current",
               "lora@window:window=20", "lora@window:window=40", "lora@window:window=20,dim=8", "latent@current"]
    keys = _keys(cfg, methods, [0, 1])
    assert len(set(keys.values())) == len(keys)
    # identical models share: 'lora@sfm' is 'lora' when sfm is the default context
    assert _keys(cfg, ["lora@sfm"], [0])[("lora@sfm", 0)] == keys[("lora", 0)]
    # evaluation-only changes re-train nothing; training changes re-train
    ev = Cf.apply_dotted(cfg, {"eval.conditions": ["clean", "noise10"], "baselines": ["ekf", "pf"]})
    assert _keys(ev, methods, [0]) == {k: v for k, v in keys.items() if k[1] == 0}
    tr = Cf.apply_dotted(cfg, {"phase2.lr": 1.0})
    assert not set(_keys(tr, methods, [0]).values()) & set(keys.values())
    k1 = lambda c: Cf.digest(c, "system", "data", "phase1", extra=[0, C.module_hash(C.MODEL_MODULES)])
    assert k1(cfg) == k1(tr) == k1(ev) and k1(cfg) != k1(Cf.apply_dotted(cfg, {"phase1.lr": 1.0}))


def test_code_hash_covers_exactly_result_modules(tmp_path, monkeypatch):
    code = "import sys, hyperkkl.job; print(' '.join(sorted(m.split('.', 1)[1] for m in sys.modules if m.startswith('hyperkkl.'))))"
    imported = set(subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.split())
    bookkeeping = {"tree"}            # its only effect on a job is the config, stored in full
    assert set(J.RESULT_MODULES) == imported - bookkeeping, imported
    pkg = tmp_path / "pkg"
    shutil.copytree(J.PKG, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(J, "PKG", pkg)
    h0 = J.code_hash()
    for f in sorted(pkg.glob("*.py")):
        src = f.read_text()
        f.write_text(src + "\n# edit\n")
        changed = J.code_hash() != h0
        f.write_text(src)
        assert changed == (f.stem in J.RESULT_MODULES), f.name
    assert J.code_hash() == h0


class _FakeSet:
    n_rejected = 0

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_run_job_wiring_with_fakes(exp, monkeypatch):
    Tr.new_node("N", None, "n", methods=["autonomous", "lora", "lora@gru"], baselines=["ekf", "nlox"],
                systems=["vdp"], seeds=[0])
    calls, paths = [], []
    monkeypatch.setattr(J, "snapshot_code", lambda ch: calls.append("snapshot") or "experiments/code/x.tar.gz")
    monkeypatch.setattr(J.Dm, "system_scales", lambda cfg, device: {
        "design_check": {"ok": True}, "t_burn": 1.0, "memory_s": 2.0, "x_std": [1, 1], "y_std": [1]})
    monkeypatch.setattr(J.Dm, "training_data", lambda cfg, seed, device: calls.append("data") or
                        {"p1": _FakeSet(3), "train": _FakeSet(5), "val": _FakeSet(2)})
    monkeypatch.setattr(J, "_cached_model", lambda path, build: paths.append(Path(path).name) or build())
    monkeypatch.setattr(J, "phase1", lambda cfg, *a: calls.append("phase1") or {"diag": {"eps_dec": 0.01}, "wall_s": 1.0})
    monkeypatch.setattr(J, "phase2", lambda kind, mc, *a: calls.append(("phase2", kind, mc["phase2"]["context"]["kind"]))
                        or {"n_params": 5, "cond": None, "ctx": None})
    monkeypatch.setattr(J, "nlox", lambda cfg, *a: calls.append("nlox") or {"n_params": 7})
    monkeypatch.setattr(J, "build_observers", lambda *a: {})
    monkeypatch.setattr(J, "state_coverage", lambda *a: calls.append("coverage") or {"shape_ood": {"outside_box": 0.0}})
    monkeypatch.setattr(J, "evaluate", lambda *a, **k: {"shape_ood": {"clean": {"families": ["a"], "methods": {},
                                                                    "_traj": {"x": np.zeros((1, 2, 2))}}}})
    monkeypatch.setattr(J, "_costs", lambda *a: {})
    pay = J.run_job("N", "vdp", 0, "cpu")
    assert calls[0] == "snapshot" and calls[1:3] == ["data", "phase1"]       # snapshot before any work
    assert [c for c in calls if isinstance(c, tuple)] == [("phase2", "none", "sfm"), ("phase2", "lora", "sfm"),
                                                         ("phase2", "lora", "gru")]
    assert paths[0].startswith("vdp_p1_") and paths[-1].startswith("vdp_nlox_")
    assert len(set(paths)) == 5 and paths[1].startswith("vdp_p2_none_")
    rd = Tr.run_dir("N", "vdp", 0)
    on_disk = json.loads((rd / "results.json").read_text())
    assert on_disk["code"] == J.code_hash() == pay["code"] and on_disk["code_snapshot"] == "experiments/code/x.tar.gz"
    assert on_disk["config"]["seed"] == 0 and on_disk["config"]["methods"] == ["autonomous", "lora", "lora@gru"]
    assert "_traj" not in on_disk["results"]["shape_ood"]["clean"] and (rd / "traj.npz").exists()
    assert "coverage" in calls and on_disk["state_coverage"] == {"shape_ood": {"outside_box": 0.0}}
    assert set(on_disk["train"]) == {"autonomous", "lora", "lora@gru", "nlox"}
    assert not (rd / "results.json.tmp").exists()


# ===========================================================================
# evaluate
# ===========================================================================

NE = 40


def _design():
    A, B = D.butterworth_AB(2, 1.0, 1)
    return Dm.Design(A, B, 0.05, NE, 1.0, D.latent_step(A, B, 0.05))


def _synthetic_set(tier, shift):
    n = {"id": 5, "shape_ood": 7}[tier]
    tid = {"id": 1, "shape_ood": 2}[tier]
    t = torch.arange(NE + 1, dtype=torch.float32)
    A = torch.tensor([10.0 * tid + e + 1 + 100 * shift for e in range(n)])
    y = A[:, None, None].expand(n, NE + 1, 1).clone()
    u = ((t / NE)[None, :, None] * (1 + torch.arange(n, dtype=torch.float32))[:, None, None]).clone()
    z = torch.zeros(n, NE + 1, 2)
    fams = [f"{['fa', 'fb', 'fc'][e % 3]}_{tier}" for e in range(n)]
    return Dm.EpisodeSet(torch.zeros(n, NE + 1, 2), u, y, y.clone(), z, z.clone(), fams, burn=0, dt=0.05)


class _Echo(Observer):
    """xhat_0 = x0_hat, xhat_k = [y_k, u_k]: with x = 0 the error IS what the observer saw."""

    def reset(self, B, x0_hat, u0):
        self.xh = x0_hat.clone()

    def estimate(self):
        return self.xh

    def advance(self, y_k, u_k, y_k1, u_k1):
        self.xh = torch.cat([y_k1, u_k1], 1)


CONDS = ["clean", "noise10", "outliers", "input_bias", "input_delay", "input_noise", "poor_init",
         "plant_drift10", "plant_drift30"]


def _eval_cfg():
    return Cf.load("vdp", {"data.sensor": {"noise": 0.0, "quant": 0.0}, "system.u_scale": 2.0,
                           "system.ic_box": [[-1, 1], [0, 4]], "eval.tiers": ["id", "shape_ood"],
                           "eval.conditions": CONDS, "eval.chunk": 4, "eval.save_traj": True})


def test_evaluate_maps_every_block_back(monkeypatch):
    cfg, des = _eval_cfg(), _design()
    scales = {"x_std": [1.0, 2.0], "y_std": [0.5]}
    sets, calls, sens, inp, inits = {}, [], [], [], []

    def fake_eval_data(cfg_, tier, plant_shift=0.0, device="cpu"):
        calls.append((tier, plant_shift))
        return sets.setdefault((tier, plant_shift), _synthetic_set(tier, plant_shift))

    orig_ws, orig_ci, orig_ig = Dm.EpisodeSet.with_sensor, Dm.corrupt_input, Ev.initial_guess
    monkeypatch.setattr(Dm, "eval_data", fake_eval_data)
    monkeypatch.setattr(Dm.EpisodeSet, "with_sensor",
                        lambda self, d, s, ys, seed, **kw: sens.append((s, seed)) or orig_ws(self, d, s, ys, seed, **kw))
    monkeypatch.setattr(Dm, "corrupt_input", lambda u, spec, ur, seed: inp.append((spec, ur, seed)) or orig_ci(u, spec, ur, seed))
    monkeypatch.setattr(Ev, "initial_guess", lambda c, n, kind, seed, dev: inits.append((kind, seed)) or orig_ig(c, n, kind, seed, dev))
    out = Ev.evaluate(cfg, des, scales, {"echo": _Echo(), "pf": _Echo()}, "cpu", log=lambda *a: None)

    assert sorted(calls) == sorted({(t, s) for t in ("id", "shape_ood") for s in (0.0, 0.1, 0.3)})   # once each
    assert set(out) == {"id", "shape_ood"} and all(set(out[t]) == set(CONDS) for t in out)
    settle = int(0.5 * NE)
    for tier in ("id", "shape_ood"):
        seed = Dm.split_seed("vdp", tier, "dev", 0)
        for c in CONDS:
            spec = cfg["conditions"][c]
            es = sets[(tier, float(spec.get("plant_shift", 0.0)))]
            n = len(es)
            y, u = es.y_clean.clone(), es.u.clone()
            if "sensor" in spec:
                merged = {"noise": 0.0, "quant": 0.0, **spec["sensor"]}
                assert (merged, seed + 1) in sens
                y = Dm.measure(es.y_clean.double(), merged, scales["y_std"], seed + 1).float()
            if c == "input_bias":
                u = u + 0.1 * 2.0                                      # relative to u_scale
            elif c == "input_delay":
                u = u[:, [max(k - 5, 0) for k in range(NE + 1)]]
            elif c == "input_noise":
                u = orig_ci(es.u, {"noise": 0.1}, 2.0, seed + 2)
                assert ({"noise": 0.1}, 2.0, seed + 2) in inp
            box = np.array([[-1.0, 1.0], [0.0, 4.0]])
            if c == "poor_init":
                x0 = orig_ig(cfg, n, "poor", seed + 3, "cpu")
                assert ("poor", seed + 3) in inits
                assert _vals(x0[:, 0]) <= {-0.9, 0.9} and _vals(x0[:, 1]) <= {0.2, 3.8}
            else:
                x0 = torch.tensor(box.mean(1), dtype=torch.float32).expand(n, 2)
            xh = torch.cat([y, u], -1)
            xh[:, 0] = x0
            exp_m = M.episode_metrics(torch.zeros_like(xh), xh, scales["x_std"], settle)
            blk = out[tier][c]
            assert blk["families"] == es.families
            for name in ("echo", "pf"):                                  # pf runs in chunks of 1
                for k, v in exp_m.items():
                    assert np.allclose(blk["methods"][name][k], v, rtol=1e-5, atol=1e-6), (tier, c, name, k)
    tr = out["shape_ood"]["clean"]["_traj"]
    assert np.allclose(tr["true"], 0) and tr["echo"].shape == (4, NE + 1, 2)
    assert np.allclose(tr["echo"][:, 1:, 0], sets[("shape_ood", 0.0)].y_clean[:4, 1:, 0].numpy())
    assert "_traj" not in out["id"]["clean"] and "_traj" not in out["shape_ood"]["noise10"]


def _vals(t):
    return set(np.round(t.numpy().astype(float), 5).tolist())


def test_initial_guess():
    cfg = Cf.load("vdp", {"system.ic_box": [[-1, 1], [0, 4]]})
    c = Ev.initial_guess(cfg, 6, "center", 0, "cpu")
    assert c.dtype == torch.float32 and torch.allclose(c, torch.tensor([[0.0, 2.0]]).expand(6, 2))
    p = Ev.initial_guess(cfg, 200, "poor", 7, "cpu")
    assert _vals(p[:, 0]) == {-0.9, 0.9} and _vals(p[:, 1]) == {0.2, 3.8}
    assert torch.equal(p, Ev.initial_guess(cfg, 200, "poor", 7, "cpu"))
    assert not torch.equal(p, Ev.initial_guess(cfg, 200, "poor", 8, "cpu"))
    with pytest.raises(ValueError):
        Ev.initial_guess(cfg, 2, "far", 0, "cpu")


def _maps(nz):
    T0 = BaseMap(2, nz, 16, 2, "tanh", (torch.zeros(2), torch.ones(2)), (torch.zeros(nz), torch.ones(nz)))
    Ti = BaseMap(nz, 2, 16, 2, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(2), torch.ones(2)))
    return {"T0": T0, "T0inv": Ti}


def test_build_observers_wiring():
    cfg = Cf.load("vdp", {"system.mismatch": 0.3, "baselines": ["ekf", "ukf", "pf", "kkl_mb"],
                          "eval.pf_particles": 64})
    des = Dm.make_design(cfg["system"])
    nz = des.A.shape[0]
    x_std, y_std = np.array([1.1, 0.9]), np.array([1.3])
    base = _maps(nz)
    dec, inj, ctx = torch.nn.Identity(), object(), object()
    trained = {"lora": {"cond": dec, "ctx": ctx}, "latent@current": {"cond": inj, "ctx": ctx},
               "autonomous": {"cond": dec, "ctx": None}}
    om, dec2 = (lambda z, c: z * c), torch.nn.Identity()
    obs = Ev.build_observers(cfg, des, {"x_std": x_std.tolist(), "y_std": y_std.tolist()}, base, trained,
                             {"omega": om, "dec": dec2}, "cpu")
    assert set(obs) == {"lora", "latent@current", "autonomous", "nlox", "ekf", "ukf", "pf", "kkl_mb"}
    assert obs["lora"].dec is dec and obs["lora"].inj is None and obs["lora"].ctx is ctx and obs["lora"].z0_map is base["T0"]
    lat = obs["latent@current"]
    assert lat.inj is inj and lat.dec.kind == "none" and lat.dec.base is base["T0inv"] and lat.z0_map is base["T0"]
    nl = obs["nlox"]
    assert isinstance(nl.ctx, Ev.RawInput) and nl.z0_map is None and nl.dec is dec2
    z, c = torch.ones(2, nz), torch.full((2, 1), 3.0)
    assert torch.equal(nl.inj(z, c), om(z, c))
    nominal, real = build_plant(cfg["system"]).params(), build_plant(cfg["system"], perturb=0.3).params()
    for b in ("ekf", "ukf", "pf"):
        f = obs[b]
        assert type(f.m).__name__ == "SaturatingActuator" and f.m.params() == nominal != real   # nominal model
        assert np.allclose(f.q2, (0.005 * x_std) ** 2 * des.dt)
        assert np.allclose(f.r2, (0.03 * y_std) ** 2)
        assert np.allclose(f.p02, (6.0 / np.sqrt(12.0)) ** 2)
    assert obs["pf"].Np == 64
    assert obs["kkl_mb"].model.params() == nominal and obs["kkl_mb"].T0 is base["T0"] and obs["kkl_mb"].Ti is base["T0inv"]


class _Still(Plant):
    """xdot = 0: integrate() increments are then pure process noise."""
    nx = 2
    PARAMS = {}

    def f(self, x, u):
        return torch.zeros_like(x)


def test_filter_noise_model_matches_data_generation():
    cfg = Cf.load("vdp", {"baselines": ["ekf"]})
    full = Dm.make_design(cfg["system"])
    des = Dm.Design(full.A, full.B, full.dt, 200, full.alpha, full.step)
    x_std, y_std = np.array([1.1, 0.9]), np.array([1.3])
    ekf = Ev.build_observers(cfg, full, {"x_std": x_std.tolist(), "y_std": y_std.tolist()}, _maps(full.A.shape[0]),
                             {}, None, "cpu")["ekf"]
    rng = np.random.RandomState(0)
    sigs = [S.sample_signal("zero", "train", rng, des.dt, des.T) for _ in range(256)]
    nt = torch.get_num_threads()
    torch.set_num_threads(2)                     # tiny ops: many threads only add contention
    try:
        x, _, _ = Dm.integrate(_Still(sensors=[0]), des, np.zeros((256, 2)), sigs,
                               float(cfg["data"]["noise_proc"]) * x_std, seed=0)
    finally:
        torch.set_num_threads(nt)
    inc_var = (x[:, 1:] - x[:, :-1]).reshape(-1, 2).var(0).numpy()
    y = Dm.measure(torch.zeros(256, 201, 1, dtype=torch.float64), cfg["data"]["sensor"], y_std, 0)
    ics = Dm._sobol(np.asarray(cfg["system"]["ic_box"], float), 4096, 0)
    print(f"\n[noise] process var/step data {inc_var} vs EKF Q {ekf.q2}; meas std data {float(y.std()):.4f} vs "
          f"EKF R^.5 {np.sqrt(ekf.r2)}; IC std data {ics.std(0)} vs P0^.5 {np.sqrt(ekf.p02)}")
    assert np.allclose(inc_var, ekf.q2, rtol=0.03)
    assert np.allclose(float(y.std()), np.sqrt(ekf.r2), rtol=0.03)
    assert np.allclose(ics.std(0), np.sqrt(ekf.p02), rtol=0.02) and np.allclose(ics.mean(0), 0.0, atol=0.02)


def test_costs():
    class Boom(Observer):
        def reset(self, *a):
            raise RuntimeError("boom")

    out = Ev.costs({"echo": _Echo(), "boom": Boom()}, nx=2, ny=1, device="cpu")
    assert out["echo"]["us_b1"] > 0 and out["echo"]["us_b64"] > 0 and "boom" in out["boom"]["error"]
