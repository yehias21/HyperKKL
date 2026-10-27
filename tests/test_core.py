"""Regression tests for the properties every result depends on.

    .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from hyperkkl import design as D
from hyperkkl import signals as S
from hyperkkl.conditioning import ALL_KINDS, build_conditioner, match_width
from hyperkkl.data import Design, integrate, run_latent
from hyperkkl.metrics import episode_metrics, hierarchical, paired_relative
from hyperkkl.nets import BaseMap, build_context
from hyperkkl.observers import KKLObserver, LatentStep, run
from hyperkkl.systems import REGISTRY, Plant, build_plant

torch.set_default_dtype(torch.float32)


class LinearPlant(Plant):
    """x' = F x + G u, y = x1 (test-only)."""
    nx = 2
    PARAMS = {}
    F = np.array([[0.0, 1.0], [-2.0, -0.3]])
    G = np.array([0.0, 1.0])

    def f(self, x, u):
        F = torch.as_tensor(self.F, dtype=x.dtype, device=x.device)
        G = torch.as_tensor(self.G, dtype=x.dtype, device=x.device)
        return x @ F.T + u * G


def _design(dt, N, nz=4, alpha=1.0):
    A, B = D.butterworth_AB(nz, alpha, 1)
    return Design(A, B, dt, N, alpha, D.latent_step(A, B, dt))


def _exact(x0, sig, T, nz=4, alpha=1.0, nseg=4000):
    """Exact (x, z) of the augmented linear system under a piecewise-constant input."""
    A, B = D.butterworth_AB(nz, alpha, 1)
    F, G, C = LinearPlant.F, LinearPlant.G, np.array([[1.0, 0.0]])
    M = np.zeros((2 + nz, 2 + nz)); M[:2, :2] = F; M[2:, :2] = B @ C; M[2:, 2:] = A
    g = np.zeros(2 + nz); g[:2] = G
    w = np.concatenate([x0, np.zeros(nz)])
    ts = list(sig.switch_times) + [T]
    for a, b, lv in zip(ts[:-1], ts[1:], sig.levels * sig.scale):
        Mg = np.zeros((3 + nz, 3 + nz)); Mg[:-1, :-1] = M; Mg[:-1, -1] = g * lv
        w = (expm(Mg * (b - a)) @ np.concatenate([w, [1.0]]))[:-1]
    return w


@pytest.mark.parametrize("family", ["square", "prbs"])
def test_labels_are_fourth_order_through_switches(family):
    T = 4.0
    rng = np.random.RandomState(0)
    sig = S.sample_signal(family, "train", rng, 0.05, T)
    x0 = np.array([[0.7, -0.2]])
    exact = _exact(x0[0], sig, T)
    errs = []
    for dt in (0.05, 0.025):
        N = int(round(T / dt))
        x, z, _ = integrate(LinearPlant(), _design(dt, N), x0, [sig], device="cpu")
        errs.append(np.abs(np.concatenate([x[0, -1].numpy(), z[0, -1].numpy()]) - exact).max())
    order = np.log2(errs[0] / errs[1])
    assert order > 3.5, (errs, order)


def test_switches_on_grid():
    rng = np.random.RandomState(1)
    for fam in S.PIECEWISE_CONSTANT - {"zero"}:
        for mode in ("train", "id"):
            s = S.sample_signal(fam, mode, rng, 0.013, 20.0)
            r = np.abs(s.switch_times / 0.013 - np.round(s.switch_times / 0.013))
            assert r.max() < 1e-9, fam
    for fam, mode in (("drive", "train"), ("drive", "sequence_ood"), ("triangle", "shape_ood"),
                      ("sawtooth", "shape_ood")):
        s = S.sample_signal(fam, mode, rng, 0.013, 20.0)
        r = np.abs(s.knots / 0.013 - np.round(s.knots / 0.013))
        assert r.max() < 1e-9 and np.all(np.diff(s.knots) > 0.5 * 0.013), fam


def test_tiers_shift_shape_or_sequence_not_amplitude():
    """Input protocol v2: held-out waveforms never appear in training, sequence_ood reuses
    a training family, and every tier stays inside the same amplitude box [-1, 1]."""
    assert not set(S.TRAIN_FAMILIES) & set(S.SHAPE_OOD_FAMILIES)
    assert set(S.SEQUENCE_OOD_FAMILIES) <= set(S.TRAIN_FAMILIES)
    assert S.TIERS == ("id", "shape_ood", "sequence_ood")
    rng = np.random.RandomState(2)
    t = np.arange(0, 50, 0.01)
    for tier in ("train",) + S.TIERS:
        for fam in S.families_for(tier):
            peak = max(np.abs(S.sample_signal(fam, tier, rng, 0.01, 50)(t)).max() for _ in range(50))
            assert peak <= 1.0 + 1e-12, (tier, fam, peak)
    with pytest.raises(ValueError):
        S.sample_signal("chirp", "train", rng, 0.01, 50)


def test_sylvester_reference():
    A, B = D.butterworth_AB(4, 1.0)
    F, C = LinearPlant.F, np.array([[1.0, 0.0]])
    P = D.sylvester_reference(F, C, A, B)
    assert np.allclose(P @ F, A @ P + B @ C, atol=1e-9)


def test_latent_filter_tracks_labels_after_washout():
    T, dt = 30.0, 0.02
    N = int(T / dt)
    des = _design(dt, N)
    rng = np.random.RandomState(3)
    sig = S.sample_signal("sinusoid", "train", rng, dt, T)
    x, z, _ = integrate(LinearPlant(), des, np.array([[0.5, 0.5]]), [sig])
    zo = run_latent(des, x[:, :, :1])
    rel = (zo[0, N // 2:] - z[0, N // 2:]).norm() / z[0, N // 2:].norm()
    assert rel < 1e-3, float(rel)


class _Id(torch.nn.Module):
    def forward(self, z, c=None):
        return z


def test_observer_alignment_and_no_lookahead():
    # A = 0-ish is not Hurwitz; use the step weights directly: z' = y, identity decoder
    class Lat:
        A = torch.zeros(1, 1)
        def __call__(self, z, yk, yk1, phi=None):
            return z + 0.05 * (yk + yk1)          # trapezoid, dt = 0.1
    obs = KKLObserver("t", Lat(), _Id())
    y = torch.full((1, 5, 1), 2.0)
    xh = run(obs, y, torch.zeros(1, 5, 1), torch.zeros(1, 1))[0, :, 0]
    assert torch.allclose(xh, torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])), xh
    y2 = y.clone(); y2[0, -1] = 99.0
    xh2 = run(obs, y2, torch.zeros(1, 5, 1), torch.zeros(1, 1))[0, :, 0]
    assert torch.allclose(xh[:-1], xh2[:-1])


@pytest.mark.parametrize("kind", ["current", "window", "gru", "sfm"])
def test_context_seq_equals_step_and_zero_collapse(kind):
    torch.manual_seed(0)
    A, _ = D.butterworth_AB(5, 1.0)
    ccfg = {"dim": 8, "hidden": 16, "window": 7, "sfm_blocks": 4}
    E = build_context(kind, ccfg, 1, 0.05, np.linalg.eigvals(A), 1.0)
    u = torch.randn(3, 60, 1)
    c_seq = E.seq(u)
    st = E.init_state(3, "cpu")
    cs = []
    for k in range(60):
        c, st = E.step(u[:, k], st)
        cs.append(c)
    assert torch.allclose(c_seq, torch.stack(cs, 1), atol=1e-4), (c_seq - torch.stack(cs, 1)).abs().max()
    assert E.seq(torch.zeros(2, 40, 1)).abs().max() == 0.0


def _base(nz=5, nx=2):
    return BaseMap(nz, nx, 32, 3, "tanh", (torch.zeros(nz), torch.ones(nz)),
                   (torch.zeros(nx), torch.ones(nx)))


@pytest.mark.parametrize("kind", [k for k in ALL_KINDS if k != "none"])
def test_conditioners_collapse_at_zero_context_and_match_budget(kind):
    torch.manual_seed(0)
    base = _base()
    w, n = match_width(kind, base, 8, 20000, 4, torch.ones(5))
    assert abs(n - 20000) / 20000 < 0.15, (kind, w, n)
    m = build_conditioner(kind, base, 8, w, 4, torch.ones(5))
    for p in m.parameters():                      # break zero-init so the test is non-trivial
        if p.requires_grad:
            p.data.normal_(0, 0.1)
    z = torch.randn(11, 5)
    if kind == "latent":
        assert m(z, torch.zeros(11, 8)).abs().max() == 0.0
    else:
        assert torch.allclose(m(z, torch.zeros(11, 8)), base(z), atol=1e-6)
        assert not torch.allclose(m(z, torch.randn(11, 8)), base(z), atol=1e-4)


def test_metrics_and_bootstrap():
    x = torch.zeros(4, 101, 2)
    xh = x + 0.1
    m = episode_metrics(x, xh, [0.1, 0.2], 50)
    assert np.allclose(m["nrmse"], np.sqrt((1 + 0.25) / 2))
    assert not m["diverged"].any()
    mu, lo, hi = hierarchical([np.ones(10), np.ones(10) * 3])
    assert mu == 2.0 and lo <= 2.0 <= hi
    d = paired_relative([np.ones(10) * 0.5] * 3, [np.ones(10)] * 3)
    assert abs(d["rel"] + 0.5) < 1e-9 and d["better"]


def test_every_plant_is_finite():
    import yaml
    from hyperkkl.config import CONFIG_DIR
    for p in sorted((CONFIG_DIR / "systems").glob("*.yaml")):
        sc = yaml.safe_load(p.read_text())
        pl = build_plant(sc)
        box = torch.tensor(sc["ic_box"], dtype=torch.float64)
        x = box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(64, len(box), dtype=torch.float64)
        f = pl.f(x, torch.randn(64, 1, dtype=torch.float64) * sc["u_scale"])
        assert torch.isfinite(f).all(), sc["name"]
        assert pl.h(x).shape == (64, len(sc["sensors"]))


def test_state_coverage_toy():
    """metrics.state_coverage: a test set drawn like the training set is covered (box ~0,
    kNN ~1 %); a shifted one is not, and the per-state box fraction finds the state."""
    from hyperkkl.metrics import state_coverage
    g = torch.Generator().manual_seed(0)
    tr = torch.rand(40, 200, 2, generator=g) * torch.tensor([1.0, 10.0])     # uniform box
    same = torch.rand(20, 200, 2, generator=g) * torch.tensor([1.0, 10.0])
    c = state_coverage(tr, same, [1.0, 10.0], n_ref=4000, n_query=2000)
    assert c["outside_box"] < 0.002 and c["outside_knn"] < 0.03, c
    sh = same.clone()
    sh[..., 0] += 0.5                                  # half of state 1 leaves [0, 1]
    c2 = state_coverage(tr, sh, [1.0, 10.0], n_ref=4000, n_query=2000)
    assert abs(c2["outside_box"] - 0.5) < 0.03 and c2["outside_box_j"][1] < 0.002, c2
    assert abs(c2["outside_knn"] - 0.5) < 0.06, c2      # beyond the edge: far from every train state
    # normalisation by x_std: a gap of 0.3 in a state of std 10 is invisible to kNN
    sh2 = same.clone()
    sh2[..., 1] = sh2[..., 1] * 0.97
    assert state_coverage(tr, sh2, [1.0, 10.0], n_ref=4000, n_query=2000)["outside_knn"] < 0.03


def test_sfm_floor_bounds_the_forgetting_rate():
    """sfm_floor = f puts every SFM block's decay rate at >= f x the latent slowest rate,
    for any parameter value (so Lemma 2's rate is guaranteed), and 0 keeps the old map."""
    import numpy as np
    import torch
    from hyperkkl.nets import SFMEncoder
    poles = np.array([-1.5 + 1.0j, -1.5 - 1.0j, -2.0])
    e0 = SFMEncoder(1, 4, 8, 0.02, poles, n_blocks=6)
    e1 = SFMEncoder(1, 4, 8, 0.02, poles, n_blocks=6, floor=0.5)
    assert float(-e0.sigma().max()) < 0.75                  # fill-in modes are slower
    with torch.no_grad():
        e1.raw_sigma.fill_(-30.0)                           # softplus -> 0: rate = floor
    assert float(-e1.sigma().max()) >= 0.75 - 1e-6
    assert abs(e1.alpha() - 0.75) < 1e-4
    e2 = SFMEncoder(1, 4, 8, 0.02, poles, n_blocks=6, floor=0.0)
    assert torch.equal(e0.sigma(), e2.sigma())
