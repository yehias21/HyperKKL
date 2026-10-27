"""Data-driven baselines (hyperkkl/baselines_data.py: EDMD-with-inputs Kalman filter, grey-box
residual EKF/UKF) and the reeval path that adds baselines to finished runs (hyperkkl/reeval.py).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests/test_baselines_data.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from hyperkkl import baselines_data as BD
from hyperkkl import config as Cf
from hyperkkl import data as Dm
from hyperkkl import evaluate as Ev
from hyperkkl import job as J
from hyperkkl import reeval as RE
from hyperkkl import tree as Tr
from hyperkkl.observers import EKF, UKF, _rk4, run
from hyperkkl.systems import Plant, REGISTRY, KnownPart


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


class _Set:
    """EpisodeSet-like (x, u) holder."""

    def __init__(self, x, u):
        self.x, self.u = x, u

    def __len__(self):
        return self.x.shape[0]


# ---------------------------------------------------------------------------
# a linear toy: x+ = A x + B u + w (exact discretisation of x' = F x + g u), y = x1
# ---------------------------------------------------------------------------

F = np.array([[0.0, 1.0], [-2.0, -0.4]])
GC = np.array([0.0, 1.0])
DT = 0.05
A_D = expm(F * DT)
B_D = np.linalg.solve(F, (A_D - np.eye(2))) @ GC


def lin_episodes(E, N, seed, q_std=0.01, hold=10):
    g = np.random.RandomState(seed)
    x = g.uniform(-1, 1, size=(E, 2))
    u = np.repeat(g.uniform(-1, 1, size=(E, N // hold + 1, 1)), hold, 1)[:, :N + 1]
    xs = [x]
    for k in range(N):
        x = x @ A_D.T + u[:, k] * B_D + q_std * g.randn(E, 2)
        xs.append(x)
    return np.stack(xs, 1), u


def reference_kf(y, u, x0, P0, Q, R):
    """Textbook KF, one episode at a time (numpy float64): y_0 assimilated on the prior."""
    out = []
    for e in range(y.shape[0]):
        x, P, xs = x0[e].copy(), P0.copy(), [x0[e].copy()]
        H = np.array([[1.0, 0.0]])

        def upd(x, P, yy):
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            return x + K @ (yy - H @ x), (np.eye(2) - K @ H) @ P

        x, P = upd(x, P, y[e, 0])
        for k in range(y.shape[1] - 1):
            x = A_D @ x + B_D * u[e, k, 0]
            P = A_D @ P @ A_D.T + Q
            x, P = upd(x, P, y[e, k + 1])
            xs.append(x.copy())
        out.append(np.stack(xs))
    return np.stack(out)


def _dic_x():
    return BD.Dictionary(torch.zeros(0, 2), [0.0, 0.0], [1.0, 1.0])


def _exact_kf(kind="linear", Q=np.diag([1e-4, 1e-4]), R=np.array([0.05]), P0=np.diag([0.3, 0.3])):
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    N = torch.zeros(1, 2, 2, dtype=torch.float64)
    m = BD.LiftedModel(t(A_D), t(B_D[:, None]), N, torch.zeros(2, dtype=torch.float64), kind)
    return BD.KoopmanKF("koopman_kf", _dic_x(), m, np.diag(Q), np.sqrt(R), [0], t(P0))


@pytest.mark.parametrize("kind", ["linear", "bilinear"])
def test_koopman_kf_with_dictionary_x_is_the_exact_kf(kind):
    """Dictionary [x] and the true (A, B): both the shared-gain (linear) and the per-episode
    covariance (bilinear with N = 0) paths are the textbook Kalman filter."""
    x, u = lin_episodes(6, 120, 0)
    R = np.array([0.05])
    y = x[:, :, :1] + np.sqrt(R) * np.random.RandomState(1).randn(6, 121, 1)
    x0 = np.zeros((6, 2)) + 0.2
    Q, P0 = np.diag([1e-4, 1e-4]), np.diag([0.3, 0.3])
    ref = reference_kf(y, u, x0, P0, Q, np.diag(R))
    ob = _exact_kf(kind, Q, R, P0)
    xh = run(ob, torch.as_tensor(y), torch.as_tensor(u), torch.as_tensor(x0)).numpy()
    assert np.allclose(xh[:, 1:], ref[:, 1:], atol=1e-9), np.abs(xh - ref).max()
    assert np.allclose(xh[:, 0], x0)                       # xhat[0] is the prior (contract)


def test_edmd_recovers_known_linear_model_and_kf():
    x, u = lin_episodes(64, 200, 2)
    xv, uv = lin_episodes(16, 200, 3)
    tr = BD.transitions([(torch.as_tensor(x), torch.as_tensor(u))], 10 ** 6, 0, "cpu")
    va = BD.transitions([(torch.as_tensor(xv), torch.as_tensor(uv))], 10 ** 6, 0, "cpu")
    dic = _dic_x()
    fit = BD.fit_edmd(dic, tr, va, "linear", BD.DEFAULTS, log=lambda *a: None)
    m = fit["model"]
    assert np.allclose(m.K.numpy(), A_D, atol=3e-3), m.K
    assert np.allclose(m.G.numpy()[:, 0], B_D, atol=3e-3), m.G
    assert np.abs(m.b.numpy()).max() < 2e-3
    assert np.allclose(fit["Q"].numpy(), 1e-4, rtol=0.1)                 # residual = process noise
    # bilinear fit of a linear plant: N ~ 0
    fb = BD.fit_edmd(dic, tr, va, "bilinear", BD.DEFAULTS, log=lambda *a: None)
    assert np.abs(fb["model"].N.numpy()).max() < 5e-3
    # the KF built from the fit tracks the exact KF
    xt, ut = lin_episodes(4, 300, 4)
    yt = xt[:, :, :1] + 0.05 * np.random.RandomState(5).randn(4, 301, 1)
    P0 = np.diag([0.3, 0.3])
    kf = BD.KoopmanKF("koopman_kf", dic, m, fit["Q"], [0.05], [0], torch.as_tensor(P0))
    ex = _exact_kf("linear", np.diag(fit["Q"].numpy()), np.array([0.05 ** 2]), P0)
    a = run(kf, torch.as_tensor(yt), torch.as_tensor(ut), torch.zeros(4, 2, dtype=torch.float64))
    b = run(ex, torch.as_tensor(yt), torch.as_tensor(ut), torch.zeros(4, 2, dtype=torch.float64))
    err = lambda z: float(((z - torch.as_tensor(xt))[:, 150:]).pow(2).mean().sqrt())
    assert abs(err(a) - err(b)) < 0.1 * err(b) + 1e-4


def test_dictionary_and_kmeans():
    g = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(200, 2, generator=g, dtype=torch.float64) * 0.05 + torch.tensor(c, dtype=torch.float64)
                   for c in ([0, 0], [3, 0], [0, 3])])
    C = BD.kmeans(X, 3, 50, seed=1)
    got = sorted(map(tuple, C.round(decimals=1).tolist()))
    assert got == [(0.0, 0.0), (0.0, 3.0), (3.0, 0.0)]
    dic = BD.Dictionary(torch.tensor([[0.0, 0.0], [1.0, 0.0]]), [0.0, 0.0], [1.0, 2.0])
    x = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
    p = dic(x)
    assert p.shape == (2, 4) and torch.equal(p[:, :2], x)                 # readout = selection
    r = np.sqrt(1 + 1)                                                    # (1, 2/2) to (0, 0)
    assert np.isclose(float(p[0, 2]), r * r * np.log(r)) and float(p[1, 2]) == 0.0


# ---------------------------------------------------------------------------
# E2: grey-box residual
# ---------------------------------------------------------------------------

def duffing_episodes(plant, E, N, seed, dt=0.02, hold=25):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(E, 2, generator=g, dtype=torch.float64) * 4 - 2)
    u = (2 * torch.rand(E, N // hold + 1, 1, generator=g, dtype=torch.float64) - 1).repeat_interleave(hold, 1)[:, :N + 1]
    xs = [x]
    for k in range(N):
        x = _rk4(plant, x, u[:, k], u[:, k], u[:, k], dt)
        xs.append(x)
    return _Set(torch.stack(xs, 1), u)


def _gb_fit(model, real, steps=1500, seed=0):
    tr = BD.transitions([duffing_episodes(real, 96, 150, seed)], 10 ** 6, 0, "cpu")
    va = BD.transitions([duffing_episodes(real, 16, 150, seed + 1)], 10 ** 6, 0, "cpu")
    xs = tr[0].std(0)
    scales = {"x_std": xs.tolist(), "x_mean": tr[0].mean(0).tolist()}
    opt = dict(BD.DEFAULTS, gb_steps=steps, gb_batch=2048, gb_lr=3e-3, gb_patience=50)
    return BD.fit_greybox(model, tr, va, 0.02, scales, 1.0, seed, "cpu", opt, log=lambda *a: None)


def test_greybox_residual_zero_when_model_exact():
    real = REGISTRY["duffing"]()
    fit = _gb_fit(real, real, steps=300)
    X = torch.rand(2000, 2, dtype=torch.float64) * 3 - 1.5
    U = torch.rand(2000, 1, dtype=torch.float64) * 2 - 1
    r = fit["model"].residual(X, U)
    scale = real.f(X, U).pow(2).mean().sqrt()
    assert float(r.pow(2).mean().sqrt() / scale) < 0.02


def test_greybox_recovers_missing_duffing_term_and_ekf_differentiates_it():
    real = REGISTRY["duffing"]()
    known = KnownPart(real)                                  # missing -x1^3
    fit = _gb_fit(known, real)
    gm = fit["model"]
    X = torch.rand(4000, 2, dtype=torch.float64) * 3 - 1.5
    U = torch.rand(4000, 1, dtype=torch.float64) * 2 - 1
    miss = real.f(X, U) - known.f(X, U)
    rel = float((gm.residual(X, U) - miss).norm() / miss.norm())
    print(f"\n[greybox] relative error of the learned missing term: {rel:.3f}")
    assert rel < 0.10
    # the EKF linearises f_model + r_theta (vmap(jacrev) through the MLP)
    ekf = EKF(gm, 0.02, [0.01, 0.01], [0.03], [1.0, 1.0])
    ekf.reset(3, torch.zeros(3, 2), torch.zeros(3, 1))
    with torch.enable_grad():
        Jm = ekf.jac(X[:3].float(), U[:3].float())
    eps = 1e-3
    fd = torch.stack([(gm.f(X[:3] + eps * e, U[:3]) - gm.f(X[:3] - eps * e, U[:3])) / (2 * eps)
                      for e in torch.eye(2, dtype=torch.float64)], -1)
    assert torch.allclose(Jm.double(), fd, atol=2e-2)


# ---------------------------------------------------------------------------
# contract: causality, batching
# ---------------------------------------------------------------------------

def _vdp_fits(kinds=("koopman_kf", "koopman_kf_bilinear", "greybox_ekf", "greybox_ukf")):
    cfg = Cf.load("vdp", {"system.knowledge": "partial", "baselines": list(kinds),
                          "data_baselines": {"n_centres": 12, "gb_steps": 100, "kmeans_points": 2000,
                                             "n_fit": 20000, "gb_n_fit": 20000}})
    real = Dm.build_plant(cfg["system"])
    mk = lambda E, s: duffing_episodes(real, E, 120, s, dt=0.05)
    data = {"train": mk(24, 0), "p1": mk(8, 1), "val": mk(8, 2)}
    xs = data["train"].x.reshape(-1, 2)
    scales = {"x_std": xs.std(0).tolist(), "x_mean": xs.mean(0).tolist(), "y_std": [float(xs[:, 0].std())]}

    class Des:
        dt = 0.05
    fits = BD.fit_all(cfg, data, scales, Des, "cpu", 0, log=lambda *a: None)
    return cfg, Des, scales, fits, real


@pytest.fixture(scope="module")
def vdp_fits():
    return _vdp_fits()


@pytest.mark.parametrize("name", ["koopman_kf", "koopman_kf_bilinear", "greybox_ekf", "greybox_ukf"])
def test_causal_and_batched_equals_per_episode(vdp_fits, name):
    cfg, des, scales, fits, real = vdp_fits
    ob = BD.make_observer(name, cfg, des, scales, [0.01, 0.01], [0.05], [1.0, 1.0], fits)
    assert ob.name == name
    ep = duffing_episodes(real, 5, 80, 7, dt=0.05)
    y = (ep.x[:, :, :1] + 0.05 * torch.randn(5, 81, 1, generator=torch.Generator().manual_seed(0),
                                              dtype=torch.float64)).float()
    u, x0 = ep.u.float(), torch.zeros(5, 2)
    xh = run(ob, y, u, x0)
    assert xh.shape == (5, 81, 2) and torch.isfinite(xh).all()
    k0 = 40
    y2 = y.clone()
    y2[:, k0 + 1:] += 5.0                                      # change the future only
    u2 = u.clone()
    u2[:, k0 + 1:] -= 0.7
    xh2 = run(ob, y2, u2, x0)
    assert torch.equal(xh[:, :k0 + 1], xh2[:, :k0 + 1])
    assert not torch.allclose(xh[:, k0 + 2:], xh2[:, k0 + 2:])
    one = torch.cat([run(ob, y[i:i + 1], u[i:i + 1], x0[i:i + 1]) for i in range(5)])
    assert torch.allclose(xh, one, atol=1e-5, rtol=1e-5)


def test_fit_all_and_build_observers_wiring(vdp_fits):
    cfg, des, scales, fits, real = vdp_fits
    assert set(fits["koopman"]) == {"linear", "bilinear"} and fits["dictionary"].n == 2 + 12
    assert set(fits["info"]) == {"koopman_kf", "koopman_kf_bilinear", "greybox_ekf", "greybox_ukf"}
    gm = fits["greybox"]["model"]
    assert type(gm.model).__name__ == "SaturatingActuator" and type(gm.model.base).__name__ == "KnownPart"
    full = Dm.make_design(cfg["system"])
    sc = {"x_std": scales["x_std"], "y_std": scales["y_std"], "domain": [[-3, 3], [-4, 4]]}
    c2 = dict(cfg, baselines=["ekf", "koopman_kf", "koopman_kf_bilinear", "greybox_ekf", "greybox_ukf",
                              "greybox_ekf_oracle"])
    fits2 = dict(fits)
    obs = Ev.build_observers(c2, full, sc, None, {}, None, "cpu", fits2)
    assert set(obs) == set(c2["baselines"])
    assert isinstance(obs["greybox_ekf"], EKF) and isinstance(obs["greybox_ukf"], UKF)
    assert obs["greybox_ekf"].m is gm and np.allclose(obs["greybox_ekf"].r2, obs["ekf"].r2)
    assert np.allclose(obs["greybox_ekf"].q2, obs["ekf"].q2) and np.allclose(obs["greybox_ekf"].p02, obs["ekf"].p02)
    assert isinstance(obs["greybox_ekf_oracle"], Ev.BlockOracle) and obs["greybox_ekf_oracle"].m is gm
    assert np.allclose(obs["koopman_kf"].r2.numpy(), obs["ekf"].r2)       # nominal R, no oracle
    with pytest.raises(KeyError, match="needs a fit"):
        Ev.build_observers(dict(cfg, baselines=["koopman_kf"]), full, sc, None, {}, None, "cpu")
    # the lifted prior's x-block is the EKF's P0 (uniform on the IC box)
    P0 = fits["koopman"]["linear"]["P0"].numpy()[:2, :2]
    assert np.allclose(np.diag(P0), obs["ekf"].p02, rtol=0.05)


# ---------------------------------------------------------------------------
# reeval on a toy node in a temp experiment dir
# ---------------------------------------------------------------------------

@pytest.fixture
def exp(tmp_path, monkeypatch):
    E = tmp_path / "experiments"
    monkeypatch.setattr(Tr, "EXP", E)
    monkeypatch.setattr(Tr, "TREE", E / "tree")
    monkeypatch.setattr(Tr, "QUEUE", E / "queue")
    monkeypatch.setattr(Dm, "CACHE", E / "cache")
    monkeypatch.setattr(J, "MODEL_CACHE", E / "cache" / "models")
    monkeypatch.setattr(J, "snapshot_code", lambda ch: "experiments/code/none.tar.gz")
    return E


TOY = {"system.T": 8.0, "data.n_p1": 6, "data.n_train": 12, "data.n_val": 6, "data.n_coll": 64,
       "eval.n_episodes": 4, "eval.tiers": ["id", "sequence_ood"], "eval.conditions": ["clean", "noise10"],
       "system.knowledge": "partial",
       "data_baselines": {"n_centres": 8, "kmeans_points": 1000, "gb_steps": 50, "n_fit": 5000, "gb_n_fit": 5000}}


def test_reeval_merges_into_finished_run(exp, monkeypatch):
    Tr.new_node("T", None, "toy", set=dict(TOY), methods=[], baselines=["ekf"], systems=["vdp"], seeds=[0])
    # a finished 'run' without learned models: run_job with phase 1 stubbed out
    monkeypatch.setattr(J, "phase1", lambda *a: {"diag": {"x": 0.0}, "wall_s": 0.0, "T0": None, "T0inv": None})
    monkeypatch.setattr(J, "_costs", lambda *a: {"ekf": {"us_cpu_b1": 1.0, "us_gpu_b64": 1.0}})
    pay = J.run_job("T", "vdp", 0, "cpu")
    rd = Tr.run_dir("T", "vdp", 0)
    before = json.loads((rd / "results.json").read_text())
    new = ["koopman_kf", "greybox_ekf"]
    out = RE.reeval_node("T", new, "cpu", log=lambda *a: None)
    assert len(out) == 1 and out[0][2]["baselines"] == new
    after = json.loads((rd / "results.json").read_text())
    assert (rd / "results.json.bak").exists()
    assert json.loads((rd / "results.json.bak").read_text()) == before
    for t, cs in after["results"].items():
        for c, b in cs.items():
            assert set(b["methods"]) == {"ekf", *new} and set(b["scorecard"]) == {"ekf", *new}
            assert b["methods"]["ekf"] == before["results"][t][c]["methods"]["ekf"]      # untouched
            assert len(b["methods"]["koopman_kf"]["nrmse"]) == len(b["families"])
    rec = after["reeval"][-1]
    assert rec["code"] == J.code_hash() and rec["data_cache_hit"] is True and "time" in rec
    assert after["code"] == before["code"] and after["config"] == before["config"]
    assert set(new) <= set(after["train"]) and set(new) <= set(after["cost"])
    # same data, same fit, same numbers as fitting inside the job
    c = dict(TOY, baselines=["ekf", *new])
    Tr.new_node("T2", None, "toy2", set=c, methods=[], systems=["vdp"], seeds=[0])
    pay2 = J.run_job("T2", "vdp", 0, "cpu")
    for m in new:
        a = after["results"]["sequence_ood"]["noise10"]["methods"][m]["nrmse"]
        b = pay2["results"]["sequence_ood"]["noise10"]["methods"][m]["nrmse"]
        assert np.allclose(a, b, rtol=1e-6), m
    # idempotent; the report includes the new methods
    assert RE.reeval_node("T", new, "cpu", log=lambda *a: None)[0][2] == {"skipped": True}
    rep = (Tr.TREE / "T" / "report.md").read_text()
    assert "koopman_kf" in rep and "greybox_ekf" in rep
    with pytest.raises(ValueError, match="model-free"):
        RE.reeval_node("T", ["nlox"], "cpu", log=lambda *a: None)
