"""Real-data tracks: fetch / checksum logic, preprocessing equality with the research pipeline (battery_pf),
loaders, scorers, the evaluate path."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import hyperkkl_bench.real as hr
import importlib
F = importlib.import_module("hyperkkl_bench.real.fetch")
from hyperkkl_bench.real import registry as RG
from hyperkkl_bench.paths import REPO

PREP = REPO / "experiments" / "real" / "cache" / "prep.pt"
PRED = REPO / "experiments" / "real" / "cache" / "predictions.pt"


def _prepared(d):
    return (RG.processed_dir(d) / "MANIFEST.json").exists()


needs = lambda d: pytest.mark.skipif(not _prepared(d), reason=f"{d} not fetched / prepared (hyperkkl-bench fetch/prepare {d})")
needs_research = pytest.mark.skipif(not PREP.exists(), reason="research run experiments/real not present")


# --------------------------------------------------------------------------- fetch / checksums
def _fake_spec(tmp_path, content: bytes, sha: str, monkeypatch, zip_member=None):
    src = tmp_path / "src" / "f.bin"
    src.parent.mkdir()
    src.write_bytes(content)
    s = {"files": [{"path": "f.bin", "url": src.as_uri(), "sha256": sha, "size": len(content)}]}
    monkeypatch.setitem(RG.DATASETS, "fake", "battery_pf")
    monkeypatch.setattr(F, "spec", lambda d: s)
    return src


def test_fetch_checksum_ok(tmp_path, monkeypatch):
    c = b"real data bytes"
    _fake_spec(tmp_path, c, hashlib.sha256(c).hexdigest(), monkeypatch)
    r = F.fetch("fake", root=tmp_path / "root", log=lambda m: None)
    assert not r["failed"] and (tmp_path / "root" / "fake" / "raw" / "f.bin").read_bytes() == c
    r2 = F.fetch("fake", root=tmp_path / "root", log=lambda m: None)          # second call: present, not downloaded
    assert "present" in r2["status"]["f.bin"]


def test_fetch_checksum_mismatch_deletes(tmp_path, monkeypatch):
    _fake_spec(tmp_path, b"tampered", "0" * 64, monkeypatch)
    with pytest.raises(RuntimeError):
        F.fetch("fake", root=tmp_path / "root", log=lambda m: None)
    assert not (tmp_path / "root" / "fake" / "raw" / "f.bin").exists()


def test_fetch_from_local_and_dry_run(tmp_path, monkeypatch):
    c = b"abc"
    src = _fake_spec(tmp_path, c, hashlib.sha256(c).hexdigest(), monkeypatch)
    r = F.fetch("fake", root=tmp_path / "r1", dry_run=True, log=lambda m: None)
    assert r["dry_run"] and "reachable" in r["status"]["f.bin"] and not (tmp_path / "r1" / "fake" / "raw" / "f.bin").exists()
    r = F.fetch("fake", root=tmp_path / "r2", from_local=src.parent, log=lambda m: None)
    assert "copied" in r["status"]["f.bin"]


def test_safe_extract_rejects_traversal(tmp_path):
    import zipfile
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        F._safe_extract(z, tmp_path / "out")


def test_specs_complete():
    for d in hr.list_real():
        s = RG.spec(d)
        for k in ("license", "citation", "redistribution", "landing"):
            assert s["source"].get(k), (d, k)
        for f in s["files"]:
            assert len(f["sha256"]) == 64 and f["url"].startswith("https://"), (d, f["path"])


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("HKB_OFFLINE") == "1", reason="offline")
def test_fetch_dry_run_network():
    """URL reachable (one file per dataset; GET of the first byte from the original host)."""
    for d in hr.list_real():
        f = RG.spec(d)["files"][0]
        ok, st = F.reachable(f["url"])
        assert ok, (d, f["url"], st)


# --------------------------------------------------------------------------- battery_pf vs research pipeline
@needs("battery_pf")
@needs_research
def test_battery_windows_equal_research():
    import sys
    import torch
    sys.path.insert(0, str(REPO))
    from hyperkkl import realdata_run as RR
    prep = torch.load(PREP, weights_only=False)
    tr = hr.load("battery_pf", "train")
    es = prep["sets"]["train"]
    for j in (0, 3, 4):
        assert np.array_equal(es.x[:, :, j].numpy(), tr.x[:, :, j])
    assert np.array_equal(es.u[:, :, 0].numpy(), tr.u) and np.array_equal(es.y.numpy(), tr.y)
    assert np.array_equal(es.x[:, :, 1:3].numpy(), tr.x_fill[:, :, 1:3])          # placeholders = model replay
    assert np.isnan(tr.x[:, :, 1:3]).all()
    for rk, bk in (("val", "val"), ("test25/US06", "test_US06"), ("test25/NN", "test_NN"), ("test0/US06", "test0_US06")):
        Y, U, S, X0, idx, sgn = RR.expand(prep["sets"][rk])
        b = hr.load("battery_pf", bk)
        assert np.array_equal(Y.numpy(), b.y) and np.array_equal(U[:, :, 0].numpy(), b.u)
        assert np.array_equal(S.numpy(), b.x[:, :, 0]) and np.array_equal(X0.numpy(), b.prior), rk
    parts = [RR.expand(prep["sets"][k]) for k in ("test25/HWFTa", "test25/HWFTb")]
    b = hr.load("battery_pf", "test_HWFET")
    assert np.array_equal(np.concatenate([p[2].numpy() for p in parts]), b.x[:, :, 0])


@needs("battery_pf")
def test_battery_counts_and_frozen():
    fr = RG.spec("battery_pf")["frozen"]
    m = json.loads((RG.processed_dir("battery_pf") / "MANIFEST.json").read_text())
    want = {"val": 58, "test_US06": 18, "test_HWFET": 67, "test_NN": 58, "train": 502}
    for k, v in want.items():
        assert m["files"][f"{k}.npz"]["n_episodes"] == v
    assert fr["n_episodes"]["test25/US06"] == 18
    assert m["identification_max_rel_diff"] in (None, 0.0) or m["identification_max_rel_diff"] < 1e-9


@pytest.mark.slow
@needs("battery_pf")
@pytest.mark.skipif(not PRED.exists(), reason="research predictions not present")
def test_battery_ekf_reproduces_research():
    import torch
    from hyperkkl_bench.baselines import EKF
    from hyperkkl_bench.real import battery_pf as B
    P = torch.load(PRED, weights_only=False)
    es = hr.load("battery_pf", "test_US06").subset(np.arange(3))
    f = EKF(gate=False, hold_missing=False)
    f.fit(None, None, B.context())
    xh, _ = f.run_batch(es.u.astype(float), es.y.astype(float), es.prior.astype(float), np.tile(es.prior_std, (3, 1)))
    assert np.abs(xh[:, :, 0] - P["pred"][("test25/US06", "ekf")][:3]).max() < 1e-9


# --------------------------------------------------------------------------- loaders + scorers
@pytest.mark.parametrize("d", ["battery_pf", "battery_lg", "emps", "silverbox", "neurobem", "calce"])
def test_loader_contract(d):
    if not _prepared(d):
        pytest.skip(f"{d} not prepared")
    ctx = hr.context(d)
    for s in hr.splits(d):
        es = hr.load(d, s)
        E, N1 = es.u.shape[:2]
        assert es.y.shape[:2] == (E, N1) and es.x.shape[:2] == (E, N1) and len(es.meta) == E
        lab = ctx["labelled"]
        assert np.isfinite(es.x[..., lab]).all()
        ok = np.ones(es.u.shape[:2], bool) if es.score_mask is None else ~es.score_mask
        assert np.array_equal(es.y[ok], es.x[..., ctx["sensors"]][ok])
        if s != "train":
            assert es.prior.shape == (E, es.x.shape[-1])
        sub = es.subset([0])
        assert sub.prior is None or sub.prior.shape[0] == 1


def test_battery_scorer_units():
    from hyperkkl_bench.real import battery_pf as B
    soc = np.linspace(1, 0.5, 1801)[None].repeat(2, 0)
    m = B.soc_metrics(soc + 0.01, soc)
    assert np.allclose(m["rmse_settled"], 1.0) and np.allclose(m["rmse_all"], 1.0)
    m = B.soc_metrics(np.where(np.arange(1801) < 900, soc + 0.3, soc), soc)
    assert np.allclose(m["rmse_settled"], 0.0) and np.allclose(m["t_conv_min"], 15.0)
    m = B.soc_metrics(np.full_like(soc, np.nan), soc)
    assert (m["rmse_settled"] > 1e3).all()                               # non-finite -> clipped large error


def test_mech_velocity_label_and_scorer():
    from hyperkkl_bench.real.mech import zero_phase_velocity
    dt = 1e-3
    t = np.arange(0, 2, dt)
    y = np.sin(2 * np.pi * 3 * t)
    _, v = zero_phase_velocity(y, dt, 100.0)
    assert np.abs(v[100:-100] - 2 * np.pi * 3 * np.cos(2 * np.pi * 3 * t[100:-100])).max() < 1e-2 * 2 * np.pi * 3


@needs("emps")
def test_mech_scorer_on_labels():
    from hyperkkl_bench.real import emps as E
    es = hr.load("emps", "test_pulses")
    r = E.score(es.x.astype(float), es)
    assert r["summary"]["nrmse_velocity"] == 0.0 and r["summary"]["nrmse_position"] == 0.0
    r = E.score(np.full(es.x.shape, np.nan), es)
    assert r["summary"]["nrmse_velocity"] == 100.0 and r["summary"]["diverged"] == 1.0


@needs("emps")
def test_emps_outage_split():
    es = hr.load("emps", "test_pulses_outage")
    m = es.score_mask
    assert m is not None and m.sum(1).min() == 400 and not m[:, : es.settle_index].any()
    k = np.argmax(m[0])
    assert (es.y[0, k:k + 100, 0] == es.y[0, k - 1, 0]).all()               # frozen at the last value
    assert hr.context("emps")["hold_missing"] is True


@needs("emps")
def test_emps_identification_matches_published():
    i = RG.spec("emps")["frozen"]
    p, pub = i["params"], i["identification"]["published_IDIM_LS_full_record"]
    for k in ("M", "Fv", "Fc"):
        assert abs(p[k] - pub[k]) / abs(pub[k]) < 0.02, k


@needs("battery_pf")
def test_real_evaluate_path():
    from hyperkkl_bench.baselines import PriorMean
    rep = hr.evaluate(PriorMean(), "battery_pf", splits_=["test_US06", "test_HWFET", "test_NN"], n_episodes=4)
    assert set(rep.results) == {"test_US06", "test_HWFET", "test_NN"} and rep.meta["headline"] > 0
    assert rep.meta["scorer"] == "settled-soc-rmse-1"


def test_real_track_registered():
    from hyperkkl_bench.tracks import TRACKS
    assert TRACKS["real"]["plants"] == hr.list_real() and "battery_pf" in TRACKS["real"]["required"]


@pytest.mark.slow
@needs("emps")
@pytest.mark.skipif(not (REPO / "hyperkkl_bench" / "archive" / "beat_greybox_scratch" / "pkg" / "hyperkkl" / "beat.py").exists(),
                    reason="HyperKKL training code (archive/beat_greybox_scratch/pkg) not present")
def test_hyperkkl_train_end_to_end_cpu(tmp_path):
    """HyperKKL (A3) trains and predicts through the benchmark API on a real track (tiny budget, CPU)."""
    from hyperkkl_bench.baselines.hyperkkl_train import HyperKKLTrain
    est = HyperKKLTrain(seed=0, device="cpu", steps_scale=0.01, n_sim=64, work=tmp_path / "hk")
    rep = hr.evaluate(est, "emps", splits_=["test_pulses"], n_episodes=2)
    assert np.isfinite(rep.results["test_pulses"]["summary"]["nrmse_velocity"])
    assert "A3" in est.fit_info["recipe"]


# --------------------------------------------------------------------------- neurobem
def test_bfl_decoder_encodings():
    """The variable-byte and tagged encodings of the Betaflight log format, on hand-computed bytes."""
    from hyperkkl_bench.real import blackbox as BB
    r = BB._Reader(bytes([0xAC, 0x02]), 0)
    assert r.uvb() == 300
    r = BB._Reader(bytes([0x03]), 0)
    assert r.svb() == -2                                      # zigzag
    assert BB._tag2_3s32(BB._Reader(bytes([0b00011011]), 0)) == [1, -2, -1]
    assert BB._tag8_4s16(BB._Reader(bytes([0b00000001, 0xF0]), 0)) == [-1, 0, 0, 0]
    assert BB._tag8_8svb(BB._Reader(bytes([0b00000101, 0x02, 0x03]), 0), 3) == [1, 0, -2]


def test_neurobem_split_lists_fixed_and_disjoint():
    fl = RG.spec("neurobem")["flights"]
    assert len(fl) == 95
    by = {}
    for k, v in fl.items():
        by.setdefault(v["split"], set()).add(k)
    assert set(by) == {"train", "val", "test_id", "test_ood"}
    names = list(by)
    for i, a in enumerate(names):                              # every flight has exactly one split, by construction of the dict
        for b in names[i + 1:]:
            assert not by[a] & by[b]
    fam = lambda s: {fl[k]["family"] for k in by[s]}
    assert fam("test_id") <= fam("train")                      # same manoeuvre types
    assert not fam("test_ood") & (fam("train") | fam("val") | fam("test_id"))     # held out entirely
    assert fam("test_ood") == {"cpc", "satellite", "vertical_osc"}


def test_neurobem_quaternion_is_body_to_world():
    from hyperkkl_bench.real.neurobem import quat_rot, roll_pitch, tilt_deg
    a = np.deg2rad(30.0)
    q = np.array([[np.sin(a / 2), 0, 0, np.cos(a / 2)]])       # roll +30 deg about body x
    R = quat_rot(q)[0]
    assert np.allclose(R @ [0, 1, 0], [0, np.cos(a), np.sin(a)])
    g = R[2]                                                    # world up in the body frame
    assert np.allclose(roll_pitch(g)[0], 30.0) and np.allclose(roll_pitch(g)[1], 0.0)
    assert np.allclose(tilt_deg(np.array([0, 0, 2.0]), np.array([0, np.sin(a), np.cos(a)])), 30.0)


def test_neurobem_boxcar_is_causal():
    from hyperkkl_bench.real.neurobem import boxcar
    t = np.arange(0, 1, 0.001)
    x = (t > 0.5).astype(float)[:, None]
    tq = np.array([0.45, 0.5, 0.505, 0.515])
    o = boxcar(t, x, tq, 0.01)[:, 0]
    assert o[0] == 0 and o[1] == 0 and 0 < o[2] < 1 and o[3] == 1    # a step after t_k never reaches sample k


def test_neurobem_plant_interface():
    import torch
    from hyperkkl_bench.plants.systems_real import NB_G, NeuroBEMNominal
    p = NeuroBEMNominal(bT0=NB_G / 4, bT1=NB_G / 4, bT2=NB_G / 4, bT3=NB_G / 4)
    x = torch.zeros(2, 12, dtype=torch.float64)
    x[:, 5] = 1.0
    x[:, 11] = NB_G
    dx = p.f(x, torch.zeros(2, 4, dtype=torch.float64))
    assert dx.shape == (2, 12) and float(dx.abs().max()) < 1e-3     # hover at u = 0 is an equilibrium
    assert p.h(x).shape == (2, 6) and p.nu == 4


def test_vector_input_filter_and_gru():
    """Multi-input path of the bundled baselines (a scalar input keeps its old shape handling)."""
    import torch
    from hyperkkl_bench.baselines import EKF
    from hyperkkl_bench.estimator import run_episodes
    from hyperkkl_bench.plants.systems import Plant

    class P2(Plant):
        nx, nu = 2, 2
        PARAMS = {}
        DEFAULT_SENSORS = (0,)

        def f(self, x, u):
            return torch.stack([x[:, 1] + u[:, 0], u[:, 1] - x[:, 1]], 1)

    ctx = {"model": P2(), "dt": 0.01, "meta": {"t_ref": 1.0}, "q_std": [0.1, 0.1], "r_std": [0.01], "x_std": [1.0, 1.0],
           "sensors": [0], "prior_std": [1.0, 1.0], "hold_missing": False}
    f = EKF(gate=False)
    f.fit(None, None, ctx)
    U = np.zeros((3, 11, 2))
    Y = np.zeros((3, 11, 1))
    X, _ = f.run_batch(U, Y, np.zeros((3, 2)), np.ones((3, 2)))
    assert X.shape == (3, 11, 2)
    f.reset(np.zeros(2))
    assert f.step(np.zeros(2), np.zeros(1))[0].shape == (2,)

    class Loop:                                                  # step path of run_episodes with a vector input
        def reset(self, x0, p0=None):
            pass

        def step(self, u, y):
            assert np.shape(u) == (2,)
            return np.zeros(2)
    assert run_episodes(Loop(), U, Y, np.zeros((3, 2)))[0].shape == (3, 11, 2)


@needs("neurobem")
def test_neurobem_alignment_and_labels():
    """Time alignment of the three clocks: every flight's raw gyro matches the published smoothed gyro after the clock fit;
    and the label attitude integrates with the measured gyro: g' = -w x g."""
    a = RG.spec("neurobem")["frozen"]["alignment"]
    assert len(a) == 95
    assert max(v["gyro_rms"] for v in a.values()) < 0.03
    assert all(0.0025 < v["slope"] < 0.0045 for v in a.values())
    es = hr.load("neurobem", "test_id").subset(np.arange(8))
    g = es.x[..., 3:6].astype(float)
    assert np.abs(np.linalg.norm(g, axis=-1) - 1).max() < 1e-3
    dg = (g[:, 1:] - g[:, :-1]) / es.dt
    pred = -np.cross(es.y[:, :-1, :3].astype(float), g[:, :-1])
    assert np.corrcoef(dg.ravel(), pred.ravel())[0, 1] > 0.8


@needs("neurobem")
def test_neurobem_episodes_by_flight():
    seen = {s: {m["rec"] for m in hr.load("neurobem", s).meta} for s in hr.splits("neurobem")}
    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not seen[a] & seen[b], (a, b)
    fl = RG.spec("neurobem")["flights"]
    for s, f in seen.items():
        assert all(fl[k]["split"] == s for k in f)
    es = hr.load("neurobem", "test_ood")
    assert es.u.shape[1:] == (1001, 4) and es.y.shape[-1] == 6 and es.x.shape[-1] == 12
    assert np.array_equal(es.y, es.x[..., 6:12])
    assert np.allclose(es.prior[:, :6], [0, 0, 0, 0, 0, 1]) and np.array_equal(es.prior[:, 6:], es.y[:, 0])
    assert es.settle_index == 500 and abs(es.dt - 0.01) < 1e-12


@needs("neurobem")
def test_neurobem_scorer_physical_units():
    from hyperkkl_bench.real import neurobem as M
    es = hr.load("neurobem", "test_id").subset(np.arange(4))
    x = es.x.astype(float)
    r = M.score(x, es)["summary"]
    assert r["nrmse_hidden"] == 0.0 and r["vel_rmse_ms"] == 0.0 and r["tilt_rmse_deg"] < 1e-2 and r["diverged"] == 0.0
    h = x.copy()
    h[..., 0] += 2.0                                            # 2 m/s on v_x only
    r = M.score(h, es)["summary"]
    assert abs(r["vel_x_ms"] - 2.0) < 1e-6 and abs(r["vel_rmse_ms"] - 2.0) < 1e-6 and r["vel_y_ms"] == 0.0
    a = np.deg2rad(5.0)
    h = x.copy()                         # rotate g by exactly 5 deg about an axis PERPENDICULAR to g (Rodrigues; a rotation
    g = h[..., 3:6].copy()               # about body x moves g by 5 deg * sqrt(1 - g_x^2) only, which is what the scorer reports)
    k = np.cross(g, [0.0, 0.0, 1.0]) + np.cross(g, [1.0, 0.0, 0.0])
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    h[..., 3:6] = g * np.cos(a) + np.cross(k, g) * np.sin(a)
    assert abs(M.score(h, es)["summary"]["tilt_rmse_deg"] - 5.0) < 1e-3
    r = M.score(np.full(x.shape, np.nan), es)["summary"]
    assert r["diverged"] == 1.0 and r["nrmse_hidden"] == 100.0
    # early samples are not scored: a transient before the settle index changes nothing
    h = x.copy()
    h[:, :500, :6] += 50.0
    r = M.score(h, es)["summary"]
    assert r["nrmse_hidden"] == 0.0 and r["diverged"] == 0.0 and r["transient_peak_std"] > 10
    h = x.copy()
    h[:, 700, 0] += 1000 * M.x_std()[0]                         # a blow-up in the settled window is divergence
    assert M.score(h, es)["summary"]["diverged"] == 1.0


# --------------------------------------------------------------------------- calce
def test_calce_spec_and_design_fixed():
    s = RG.spec("calce")
    assert len(s["files"]) == 16 and all(len(f["sha256"]) == 64 for f in s["files"])
    from hyperkkl_bench.real import calce as C
    assert C.STEP == {"DST": 8, "US06": 16, "FUDS": 24}
    assert C.OCV_BRANCH == "discharge" and C.ECM_MIN_SOC == 0.1          # chosen on val (DIAG_REAL.md)
    fr = s["frozen"]
    assert fr and "DISCHARGE branch" in fr["identification_note"]


@needs("calce")
def test_calce_charge_balance_per_cell():
    from hyperkkl_bench.real import calce as C
    Q = {c: C.cell_q_ocv(c)["Q"] for c in (7, 8)}
    for s in C.splits():
        es = C.load(s)
        q = np.array([Q[int(m["rec"][5])] for m in es.meta])
        pred = es.u[:, :-1].astype(float).sum(1) / 3600 / q
        obs = (es.x[:, 0, 0] - es.x[:, -1, 0]).astype(float)
        assert np.abs(pred - obs).max() < 1e-6, s


@needs("calce")
def test_calce_splits_by_profile_and_temperature():
    from hyperkkl_bench.real import calce as C
    fam = lambda s: {m["family"] for m in C.load(s).meta}
    assert fam("train") == {"DST"} and fam("val") == {"DST"}
    assert fam("test_US06") == {"US06"} and fam("test_FUDS") == {"FUDS"}
    tr, va = {m["rec"] for m in C.load("train").meta}, {m["rec"] for m in C.load("val").meta}
    assert not tr & va and va == {"A1-008_DST_25C"}
    assert {m["temp"] for m in C.load("test0_US06").meta} == {0.0}
