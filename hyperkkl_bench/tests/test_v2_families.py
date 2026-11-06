"""v2 (2.0): the two families, operating-profile tiers, inclusion check 4a, hidden test, API compatibility."""
import hashlib
import inspect
import json
import math

import numpy as np
import pytest

import hyperkkl_bench as hkb
from hyperkkl_bench import families as F
from hyperkkl_bench.plants import profiles_v2 as PV


def _have_v2():
    try:
        F.data_root_v2()
        return True
    except FileNotFoundError:
        return False


needs_v2 = pytest.mark.skipif(not _have_v2(), reason="v2 data not present (data_v2/)")


# ------------------------------------------------------------------------------------------------ no data needed
def test_version():
    assert hkb.BENCH_VERSION.startswith("2.0.0")
    from hyperkkl_bench.version import DATA_V2_VERSION
    assert DATA_V2_VERSION == "hkb-data-2.0"


def test_family_rules():
    assert set(F.AUTONOMOUS) == {"vdp", "fhn", "rossler", "lorenz", "lorenz96", "network", "lorenz96_n10"}
    for p in ("battery", "bioreactor", "cartpole_b", "quad_tank", "highway_v2", "two_link_arm_v2", "duffing"):
        with pytest.raises(ValueError, match="rest at u = 0|not an autonomous"):
            F.parse(f"autonomous/{p}/partial")
    for p in ("lorenz", "rossler"):
        F.parse(f"autonomous/{p}/partial")
        with pytest.raises(ValueError, match="no realistic cell"):
            F.parse(f"autonomous/{p}/realistic")
    with pytest.raises(ValueError):
        F.parse("driven/vdp")
    assert F.HEADLINE_TIER == {"driven": "test_op_ood", "autonomous": "test_id"}
    assert "input_cmd_vs_applied" in F.CONDITIONS["driven"]["test_op_id"]
    assert not any(c.startswith("input_") for t in F.CONDITIONS["autonomous"].values() for c in t)


def test_profiles_v2_physical():
    a = PV.SPECS["two_link_arm_v2"]["act"]
    g_max = (1.0 * 0.5 + 1.0 * 1.5) * 9.81                                   # arm stretched horizontally: 19.6 N m
    assert (a.lo, a.hi) == (-12.0, 12.0) and a.hi < g_max                    # cannot hold the arm horizontal statically
    assert (PV.SPECS["highway_v2"]["lo"], PV.SPECS["highway_v2"]["hi"]) == (-0.25, 0.2)
    src = inspect.getsource(PV.op_astat)
    assert "0.004" not in src and "0.012" not in src                          # unverifiable A-stat rates removed
    for p in ("battery", "two_link_arm_v2", "highway_v2", "bioreactor", "vdp"):
        tr = {o.__name__ for o in PV.operators(p, "train")}
        od = {o.__name__ for o in PV.operators(p, "ood")}
        assert tr and od and not (tr & od)
        for split in ("train", "ood"):
            pr = PV.generate(p, split, 0)
            box = PV.SPECS[p]
            assert box["lo"] - 1e-9 <= pr.u.min() and pr.u.max() <= box["hi"] + 1e-9
    assert set(PV.UNCITED) >= {"vdp", "duffing", "cartpole_b", "two_link_arm_v2", "bioreactor"}


def test_tracks_registered():
    from hyperkkl_bench.tracks import TRACKS
    assert TRACKS["autonomous"]["headline"] == "test_id/clean"
    assert TRACKS["driven"]["headline"] == "test_op_ood/clean"


@pytest.mark.slow
def test_autonomy_check_rule():
    from hyperkkl_bench.autonomy import check_autonomous
    from hyperkkl_bench.data import metadata
    m = metadata("vdp")
    assert check_autonomous(m["system_config"], float(m["dt"]), float(m["T"]), perturb=(0.0,), n_ic=8)["pass"]
    m = metadata("duffing")
    assert not check_autonomous(m["system_config"], float(m["dt"]), float(m["T"]), perturb=(0.0,), n_ic=8)["pass"]


# ------------------------------------------------------------------------------------------------ with data
@needs_v2
def test_load_spec_and_recipes():
    tr = hkb.load("driven/vdp/realistic", "train")
    mx = hkb.load("driven/vdp/realistic", "train", recipe="mixed")
    assert len(tr) == 600 and len(mx) == 1200 and mx.x.shape[0] == 1200
    te = hkb.load("driven/vdp/realistic", "test_op_ood")
    assert te.cmd is not None and te.cmd.shape == te.u.shape
    assert set(te.families) <= {o.__name__ for o in PV.operators("vdp", "ood")}
    a = hkb.load("autonomous/vdp/partial", "test_ic_ood")
    assert float(np.abs(a.u).max()) == 0.0
    m = F.metadata_v2("autonomous/vdp/partial")
    box = np.asarray(m["ic_box"], float)
    c, h = box.mean(1), (box[:, 1] - box[:, 0]) / 2
    r = np.abs((a.x0 - c) / h).max(1)
    assert (r >= 0.8 - 1e-5).all()                                  # held-out IC shell, disjoint from the core
    sub = hkb.load("driven/vdp/realistic", "test_op_id", n=10, seed=1)
    assert len(sub) == 10 and sub.cmd.shape[0] == 10


@needs_v2
def test_v1_call_unchanged():
    es = hkb.load("vdp", "realistic", "test_id")
    assert es.split == "test_id" and es.x.shape[0] == 64


@needs_v2
def test_metadata_hashes_match_files():
    m = F._meta("driven", "vdp")
    root = F.data_root_v2()
    for v, d in m["variants"].items():
        for s in ("train", "test_op_ood"):
            f = root / "data" / "driven" / "vdp" / v / f"{s}.parquet"
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            assert h == d["files"][s]["sha256"], (v, s)


@needs_v2
def test_conditions_v2():
    es = hkb.load("driven/vdp/realistic", "test_op_id", n=6)
    U0, Y0, X0, _ = F.apply_condition("clean", es, "driven/vdp/realistic")
    U1, Y1, _, _ = F.apply_condition("outliers", es, "driven/vdp/realistic")
    U2, _, _, _ = F.apply_condition("outliers", es, "driven/vdp/realistic")
    assert np.array_equal(U0, es.u) and np.array_equal(U1, U2)
    frac = float((np.abs(Y1 - Y0) > 1e-9).mean())
    assert 0.005 < frac < 0.05
    Uc, _, _, _ = F.apply_condition("input_cmd_vs_applied", es, "driven/vdp/realistic")
    assert np.allclose(Uc, es.cmd)                                   # the observer is told the command
    Uq, _, _, _ = F.apply_condition("input_quant", es, "driven/vdp/realistic")
    assert np.abs(Uq - es.u).max() <= 2.0 / 4096 + 1e-6
    with pytest.raises(ValueError):
        F.apply_condition("input_quant", hkb.load("autonomous/vdp/partial", "test_id", n=2), "autonomous/vdp/partial")


@needs_v2
def test_evaluate_v2_and_hidden_roundtrip(tmp_path):
    from hyperkkl_bench.baselines import PriorMean
    rep = hkb.evaluate_v2(PriorMean(), "autonomous/vdp/partial", n_episodes=4)
    s = rep["results"]["test_id"]["clean"]["summary"]["nrmse_unmeasured"]
    assert 0.3 < s < 3.0                                              # x_std-normalised: the 'knows nothing' level
    files = F.predict_v2(PriorMean(), "autonomous/vdp/partial", tmp_path)
    assert len(files) == 2 and (tmp_path / "SUBMISSION.json").exists()
    from hyperkkl_bench.paths import labels_root
    if not (labels_root() / "v2").exists():
        pytest.skip("private labels not present")
    sc = F.score_hidden_v2(tmp_path)
    assert set(sc) == {"autonomous/vdp/partial/hidden_id", "autonomous/vdp/partial/hidden_ic_ood"}
    pub = hkb.load("autonomous/vdp/partial", "hidden_id")
    assert pub.x is None                                              # public hidden files carry no labels
