"""Tests for the results layer: the decision-rule engine against a hand-computed example,
and failed-filter (climatology) marking.

    .venv/bin/python -m pytest analysis/test_results_layer.py -q
"""

import numpy as np
import pytest

from analysis import rules as R
from hyperkkl.metrics import paired_relative


def _seeds(means, n_ep=8, jitter=0.0, seed=0):
    rng = np.random.RandomState(seed)
    return [m * (1 + jitter * rng.randn(n_ep)) for m in means]


def test_paired_relative_hand_example():
    # 3 seeds; method a is 20%, 10%, 30% worse than b on every episode of its seed
    b = [np.full(4, 1.0), np.full(4, 2.0), np.full(4, 0.5)]
    a = [np.full(4, 1.2), np.full(4, 2.2), np.full(4, 0.65)]
    p = paired_relative(a, b)
    assert p["rel"] == pytest.approx(0.2)                 # mean(0.2, 0.1, 0.3)
    # t-interval by hand: sd = 0.1, t_{0.975,2} = 4.302653, half = 4.302653 * 0.1 / sqrt(3)
    half = 4.302653 * 0.1 / np.sqrt(3)
    assert p["ci"][0] == pytest.approx(0.2 - half, abs=1e-5)
    assert p["ci"][1] == pytest.approx(0.2 + half, abs=1e-5)
    # half = 0.2484 -> CI [-0.048, 0.448] includes 0: +20% on 3 seeds is NOT significant
    assert not p["worse"] and not p["better"]
    assert R.within_ci(p) and R.no_loss(p)
    # tighter seeds (19%, 20%, 21%): CI [0.175, 0.225] excludes 0 -> significantly worse
    a2 = [np.full(4, 1.19), np.full(4, 2.4), np.full(4, 0.605)]
    p2 = paired_relative(a2, b)
    assert p2["worse"] and not R.no_loss(p2)


def test_vote_hand_example():
    """Hand example: rule '>= 4/6 systems', six systems:
    a: better (CI excl. 0), b: better, c: better, d: n.s., e: better, f: missing."""
    systems = list("abcdef")
    per = {"a": True, "b": True, "c": True, "d": False, "e": True}
    v = R.vote(per, systems, 4)
    assert (v.k, v.n, v.label) == (4, 5, R.PASS)
    assert v.missing == ["f"]
    # 3 met, 2 not met, 1 missing: 3 + 1 = 4 >= 4 -> the missing system decides
    v = R.vote({"a": True, "b": True, "c": True, "d": False, "e": False}, systems, 4)
    assert v.label == R.INCONCLUSIVE
    # 2 met, 3 not met, 1 missing: 2 + 1 < 4 -> FAIL already
    v = R.vote({"a": True, "b": True, "c": False, "d": False, "e": False}, systems, 4)
    assert v.label == R.FAIL
    # all six present, 3/6 -> FAIL
    v = R.vote({s: s in "abc" for s in systems}, systems, 4)
    assert (v.label, v.frac) == (R.FAIL, "3/6")
    # veto clause ('and significantly worse on none'): 5/6 met but one veto -> FAIL
    v = R.vote({s: s != "f" for s in systems}, systems, 4, veto={"f": True})
    assert v.label == R.FAIL and v.vetoed == ["f"]
    # veto clause with a missing system: a PASS count is not final
    v = R.vote({s: True for s in "abcde"}, systems, 4, veto={s: False for s in "abcde"})
    assert v.label == R.INCONCLUSIVE
    assert R.vote({}, systems, 4).label == R.PENDING


def test_rule_from_paired_comparisons():
    """End-to-end on synthetic seeds: learned beats filter on 4 systems (-30%, consistent),
    ties on 1, loses on 1 -> 4/6 PASS."""
    systems = ["s1", "s2", "s3", "s4", "s5", "s6"]
    effect = {"s1": 0.7, "s2": 0.7, "s3": 0.7, "s4": 0.7, "s5": 1.0, "s6": 1.3}
    per = {}
    for i, s in enumerate(systems):
        b = _seeds([1.0, 1.1, 0.9], jitter=0.01, seed=i)
        rng = np.random.RandomState(100 + i)
        a = [x * effect[s] * (1 + 0.01 * rng.randn(len(x))) for x in b]
        per[s] = R.better(paired_relative(a, b))
    v = R.vote(per, systems, 4)
    assert v.met == ["s1", "s2", "s3", "s4"] and v.label == R.PASS


def test_combine():
    assert R.combine([R.PASS, R.PASS]) == R.PASS
    assert R.combine([R.PASS, R.FAIL]) == R.FAIL
    assert R.combine([R.PASS, R.INCONCLUSIVE]) == R.INCONCLUSIVE
    assert R.combine([R.PENDING, R.PENDING]) == R.PENDING


def test_tost():
    p = {"rel": 0.02, "ci": [-0.05, 0.08], "n_seeds": 5}
    t = R.tost(p)
    assert t["equivalent"] and t["noninferior"] and not t["underpowered"]
    p = {"rel": 0.02, "ci": [-0.25, 0.29], "n_seeds": 3}          # half-width 0.27 >= 0.10
    t = R.tost(p)
    assert not t["equivalent"] and t["underpowered"]
    p = {"rel": -0.3, "ci": [-0.5, -0.1], "n_seeds": 3}           # better: non-inferior, not equivalent
    t = R.tost(p)
    assert t["noninferior"] and not t["equivalent"]
    assert R.tost({"rel": 0.0, "ci": [0, 0], "n_seeds": 1})["underpowered"]


def test_failed_filter_marking():
    means = {"ekf": 3.6, "ukf": 0.9, "pf": 0.05, "kkl_mb": float("nan"), "ekf_huber": 1.2, "lora": 0.02}
    clim = 1.0
    assert R.failed(3.6, clim) and R.failed(float("nan"), clim) and not R.failed(0.9, clim)
    best, bad, present = R.best_working(means, ("ekf", "ukf", "pf", "kkl_mb", "ekf_huber", "ekf_oracle"), clim)
    assert best == "pf"
    assert sorted(bad) == ["ekf", "ekf_huber", "kkl_mb"]
    assert present == ["ekf", "ukf", "pf", "kkl_mb", "ekf_huber"]          # ekf_oracle absent: not counted
    # every filter failed -> no working filter
    best, bad, _ = R.best_working({"ekf": 2.0, "ukf": 1.5}, ("ekf", "ukf"), 1.0)
    assert best is None and len(bad) == 2
    # the overall best filter may be FAILED while a working one exists: compare against the working one
    assert R.argmin({"ekf": 0.8, "ukf": 1.4}, ("ekf", "ukf")) == "ekf"


def test_knee():
    rel = {1: 2.0, 2: 1.0, 4: 0.3, 8: 0.04, 16: 0.06, 32: 0.01}
    assert R.knee(list(rel), rel) == 32          # 16 is +6% -> knee after it
    rel[16] = 0.02
    assert R.knee(list(rel), rel) == 8


def test_climatology_definition():
    """The constant mean predictor scores sqrt(mean_j var_j / sigma_j^2) (= 1 when sigma is the std)."""
    rng = np.random.RandomState(0)
    X = rng.randn(40, 201, 2) * np.array([2.0, 0.5])
    sig = np.array([2.0, 0.5])
    S = X[:, 100:]
    mu = S.reshape(-1, 2).mean(0)
    clim = np.mean(np.sqrt((((S - mu) / sig) ** 2).mean((1, 2))))
    assert clim == pytest.approx(1.0, abs=0.02)


def test_observer_groups():
    from analysis.store import DATA_DRIVEN, FILTERS, KKL, group_of
    assert "nlox" in DATA_DRIVEN and "nlox" not in FILTERS            # nlox is a same-data baseline, not a filter
    assert not set(KKL) & set(FILTERS) and not set(DATA_DRIVEN) & set(FILTERS) and not set(KKL) & set(DATA_DRIVEN)
    assert [group_of(m) for m in ("hyper", "ekf_oracle", "greybox_ekf", "koopman_kf_bilinear", "autonomous")] == \
        ["KKL", "model-based", "data-driven", "data-driven", "KKL"]


def test_data_driven_pending_until_merged():
    """Comparison (b): systems without merged baselines are missing (not losses), so the vote stays open."""
    systems = list("abcdef")
    v = R.vote({"a": False, "b": False}, systems, 4)       # 2 merged systems, both lost
    assert v.label == R.INCONCLUSIVE and v.missing == list("cdef")
    v = R.vote({"a": False, "b": False, "c": False}, systems, 4)
    assert v.label == R.FAIL                              # 0 + 3 missing < 4: cannot reach 4/6 any more


def test_regimes_pending_without_runs():
    """C3-regimes: the rule engine returns PENDING (never FAIL) while the node has no finished runs."""
    from analysis.evidence import analyse_regimes
    from analysis.store import Store
    st = Store()
    r = analyse_regimes(st)
    if not st.runs("C3-regimes"):
        assert r["label"] == R.PENDING
        assert all(row["vote"].label in (R.PENDING,) for row in r["data"]["rows"].values()) or not r["data"]["rows"]


def test_single_method_no_selection():
    """Author decision 2026-10-26: in SINGLE mode HyperKKL (= lora) is used even where hyper is better."""
    from analysis import evidence as E
    lora = [np.full(8, 1.0), np.full(8, 1.1), np.full(8, 0.9)]
    hyper = [x * 0.5 for x in lora]                      # the ablation is much better on this system
    lat = [x * 2.0 for x in lora]
    conc = [x * 2.0 for x in lora]
    vals = {"lora": lora, "hyper": hyper, "latent": lat, "concat": conc}
    means = {m: float(np.mean([v.mean() for v in vs])) for m, vs in vals.items()}
    pairs = {(a, b): paired_relative(vals[a], vals[b]) for a in vals for b in vals if a != b}
    T = {"shape_ood": {"s1": {"means": means, "pairs": pairs, "vals": vals, "n_seeds": 3}}}
    old = E.SINGLE["on"]
    try:
        E.SINGLE["on"] = True
        RR = E.c1_rules(T, ["s1"], need=1)
        assert RR["weight_space"]["shape_ood"]["pick"]["s1"] == "lora"
        assert "hyper" not in RR["location"]["shape_ood"]
        assert E.kkl_candidates() == ("lora",)
        E.SINGLE["on"] = False
        RR = E.c1_rules(T, ["s1"], need=1)
        assert RR["weight_space"]["shape_ood"]["pick"]["s1"] == "hyper"      # the old best-of pick
    finally:
        E.SINGLE["on"] = old


def test_rules_robust_without_dense_ablation():
    """The dense ablation ('hyper') is optional: C1 rules and evidence lines work when it is absent."""
    from analysis import evidence as E
    base = [np.full(8, 1.0), np.full(8, 1.1), np.full(8, 0.9)]
    vals = {"lora": base, "latent": [x * 2 for x in base], "concat": [x * 2 for x in base],
            "film": [x * 1.5 for x in base]}
    means = {m: float(np.mean([v.mean() for v in vs])) for m, vs in vals.items()}
    pairs = {(a, b): paired_relative(vals[a], vals[b]) for a in vals for b in vals if a != b}
    T = {t: {"s1": {"means": means, "pairs": pairs, "vals": vals, "n_seeds": 3}} for t in ("shape_ood", "sequence_ood")}
    for single in (True, False):
        old = E.SINGLE["on"]
        try:
            E.SINGLE["on"] = single
            RR = E.c1_rules(T, ["s1"], need=1)
            lines = E._c1_evidence(T, RR, ["s1"], ("shape_ood", "sequence_ood"))
            assert lines and RR["weight_space"]["shape_ood"]["vote"].label == R.PASS
        finally:
            E.SINGLE["on"] = old


def test_exclusion_switch():
    """EXCLUDED_SYSTEMS is applied by one switch: off -> 6 dev plants (pre-registered), on -> 5 (headline)."""
    from analysis import store as S
    old = S.EXCLUDE["on"]
    try:
        S.EXCLUDE["on"] = False
        assert "duffing" in S.dev() and len(S.dev()) == 6
        assert any(s == "duffing" for s, _ in S.Store.expected("C1-where"))
        S.EXCLUDE["on"] = True
        assert "duffing" not in S.dev() and len(S.dev()) == 5
        assert not any(s == "duffing" for s, _ in S.Store.expected("C1-where"))
        assert S.DEV == S.dev() and "duffing" in S.DEV_ALL
    finally:
        S.EXCLUDE["on"] = old
