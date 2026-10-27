"""Real-data battery experiment (hyperkkl/realdata.py): loader, split, masked loss, ECM."""

import datetime as dt
import math

import numpy as np
import pytest
import scipy.io as sio
import torch

from hyperkkl import realdata as R


def _write_mat(path, t, I, V, ah=None, T=None, start="3/20/2017 1:43:49 AM"):
    n = len(t)
    ah = -np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))]) / 3600.0 * -1 if ah is None else ah
    meas = {"TimeStamp": np.array([start] * n, dtype=object), "Voltage": V, "Current": I, "Ah": ah,
            "Wh": np.zeros(n), "Power": V * I, "Battery_Temp_degC": np.full(n, 25.6) if T is None else T,
            "Time": t, "Chamber_Temp_degC": np.full(n, 25, dtype=np.uint8)}
    sio.savemat(str(path), {"meas": meas})


def test_loader_parses_synthetic_file(tmp_path):
    # 0.1 s log, discharge of 2 A (lab sign: negative) for 10 s then rest 10 s
    t = np.arange(0, 20.0001, 0.1)
    I = np.where(t < 10, -2.0, 0.0)
    V = 4.0 + 0.01 * I
    ah = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))]) / 3600.0
    p = tmp_path / "03-20-17_01.43_25degC_US06_Pan18650PF.mat"
    _write_mat(p, t, I, V, ah)
    raw = R.load_mat(p)
    assert set(R.FIELDS) <= set(raw) and len(raw["Time"]) == len(t)
    assert R.parse_name(p) == (25.0, "US06")
    assert R.parse_name("x_n10degC_HWFET_Pan18650PF.mat") == (-10.0, "HWFET")
    rec = R.load_recording(p, Q=2.0, offset=0.6)
    assert rec.dt == 1.0 and len(rec) == 20 and rec.family == "US06"
    # ZOH bin means, discharge positive; the bin straddling the step at t = 10 s is mixed
    assert np.allclose(rec.u[:9], 2.0, atol=1e-9) and np.allclose(rec.u[11:], 0.0, atol=1e-9)
    assert abs(rec.u.sum() - 20.0) < 0.15                         # charge preserved (20 As)
    # SOC from the lab counter: 1 + Ah / Q
    assert np.allclose(rec.soc, 1.0 + rec.ah / 2.0)
    assert rec.soc[0] == pytest.approx(1.0) and rec.soc[-1] == pytest.approx(1 - 20 / 3600 / 2.0, abs=1e-4)
    # Vm: exact ZOH first-order filter of the bin-mean voltage minus V_REF
    a = math.exp(-1.0 / R.TAU_F)
    assert rec.vm[1] == pytest.approx(a * rec.vm[0] + (1 - a) * (rec.vbar[0] - R.V_REF))
    assert np.allclose(rec.dT, 25.6 - 25.6)                       # 25 + 0.6 offset
    assert rec.abs_start == dt.datetime(2017, 3, 20, 1, 43, 49)
    assert rec.abs_end == rec.abs_start + dt.timedelta(seconds=20)
    assert rec.cc_check < 1e-6


def test_loader_rejects_malformed(tmp_path):
    p = tmp_path / "bad_25degC_US06_Pan18650PF.mat"
    sio.savemat(str(p), {"meas": {"Time": np.arange(3.0), "Voltage": np.ones(3)}})
    with pytest.raises(ValueError):
        R.load_mat(p)


def _rec(name, start_h, dur_s, cycle):
    st = dt.datetime(2017, 3, 20) + dt.timedelta(hours=start_h)
    z = np.zeros(4)
    return R.Recording(name=name, temp=25.0, cycle=cycle, t=z, u=z, vbar=z, vm=z, temp_case=z, ah=z,
                       soc=z, Q=3.0, dt=1.0, t_amb_eff=25.0, abs_start=st,
                       abs_end=st + dt.timedelta(seconds=dur_s))


def test_split_has_no_leakage():
    for split in (R.SPLIT, R.SPLIT_STRICT):
        roles = [set(v) for v in split.values()]
        assert all(not (a & b) for i, a in enumerate(roles) for b in roles[i + 1:])
        assert not ({R.FAMILY.get(c, c) for c in split["test"]} & {R.FAMILY.get(c, c) for c in split["train"]})
        assert {R.FAMILY.get(c, c) for c in split["test"]} == set(R.TEST_FAMILIES)
    recs = {c: _rec(c, i * 5, 3600, c) for i, c in enumerate(sum(map(list, R.SPLIT.values()), []))}
    R.check_split(R.SPLIT, recs)                                   # disjoint: passes
    with pytest.raises(ValueError):                                # a name in two roles
        R.check_split({"train": ("UDDS",), "test": ("UDDS",)}, recs)
    recs["US06"] = _rec("US06", 0.5, 3600, "US06")                 # overlaps Cycle_1 in time
    with pytest.raises(ValueError):
        R.check_split(R.SPLIT, recs)


def test_window_starts_stay_inside():
    for n, N, s in ((4818, 1800, 300), (1802, 1800, 120), (1801, 1800, 1)):
        st = R.window_starts(n, N, s)
        assert all(x + N + 1 <= n - 1 for x in st)
    assert R.window_starts(1801, 1800, 1) == []


def test_masked_loss_ignores_unlabelled_states():
    from hyperkkl.conditioning import ConditionedDecoder
    from hyperkkl.nets import BaseMap, CurrentEncoder
    from hyperkkl.train import _decoder_loss
    torch.manual_seed(0)
    nz, nx, E, N1 = 7, 5, 3, 20
    base = BaseMap(nz, nx, 16, 2, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(nx), torch.ones(nx)))
    ctx = CurrentEncoder(1, 4, 8)
    dec = ConditionedDecoder(base, "concat", 4, 8)
    params = [p for p in list(dec.parameters()) + list(ctx.parameters()) if p.requires_grad]
    sc = {"x_std": [0.3, 0.01, 0.02, 1.0, 0.2]}
    ms = R.masked_scales(sc)
    sig = torch.as_tensor(ms["x_std"], dtype=torch.float32)
    z = torch.randn(E, N1, nz)
    x = torch.randn(E, N1, nx)

    class ES:
        pass
    es = ES(); es.z_obs, es.x, es.u = z, x, torch.randn(E, N1, 1)
    e = torch.arange(E)
    k = torch.randint(0, N1, (E, 6))

    def loss_and_grad(xx):
        es.x = xx
        for p in params:
            p.grad = None
        L = _decoder_loss(dec, ctx, es, e, k, sig)
        L.backward()
        return L.detach(), [p.grad.clone() for p in params]

    L1, g1 = loss_and_grad(x)
    x2 = x.clone(); x2[:, :, 1:3] = 1e3 * torch.randn(E, N1, 2)    # change ONLY the unlabelled states
    L2, g2 = loss_and_grad(x2)
    assert torch.equal(L1, L2) and all(torch.equal(a, b) for a, b in zip(g1, g2))
    # the loss equals the masked mean over the labelled states with their own sigma
    with torch.no_grad():
        r = torch.arange(E)[:, None]
        c = ctx.seq(es.u)[r, k].reshape(-1, 4)
        pred = dec(z[r, k].reshape(-1, nz), c); tgt = x[r, k].reshape(-1, nx)
        lab = list(R.LABELLED)
        ref = ((pred - tgt)[:, lab] / torch.tensor(sc["x_std"])[lab]).pow(2).mean()
    assert float(L1) == pytest.approx(float(ref), rel=1e-5)
    x3 = x.clone(); x3[:, :, 0] += 0.1                             # a labelled state does matter
    assert not torch.equal(loss_and_grad(x3)[0], L1)


def test_ocv_fit_and_extension():
    s = np.linspace(0, 1, 201)
    v = 3.0 + 1.2 * s - 0.3 * np.exp(-s / 0.02) + 0.05 * np.sin(3 * s)
    fit = R.fit_ocv(s, v)
    assert fit["rms"] < 2e-3
    st = torch.linspace(-0.2, 1.2, 57, dtype=torch.float64)
    vt = R.ocv_eval(st, fit["coef"], fit["w0"], fit["w1"], xp=torch).numpy()
    vn = R.ocv_eval(st.numpy(), fit["coef"], fit["w0"], fit["w1"])
    assert np.allclose(vt, vn, atol=1e-9)
    # C^1 across the edges: one-sided slopes agree; linear beyond
    for e in (0.0, 1.0):
        h = 1e-5
        f = lambda q: R.ocv_eval(np.array([q]), fit["coef"], fit["w0"], fit["w1"])[0]
        assert (f(e + h) - f(e)) / h == pytest.approx((f(e) - f(e - h)) / h, rel=2e-2)
    d = np.diff(vn[:8])
    assert np.allclose(d, d[0], rtol=1e-6)


def test_fit_ecm_recovers_parameters():
    rng = np.random.RandomState(0)
    n = 6000
    u = np.repeat(rng.uniform(-3, 8, n // 20), 20)
    Q = 3.0
    soc = 1.0 - np.concatenate([[0.0], np.cumsum(u[:-1])]) / (3600 * Q)
    ocv = lambda s: 3.0 + 1.2 * s
    true = np.log([0.03, 0.02, 20.0, 0.05, 500.0])
    soc_mid = soc - 0.5 * u / (3600 * Q)
    v = R.ecm_voltage(true, u, soc_mid, ocv, 1.0)
    rec = R.Recording(name="s", temp=25, cycle="S", t=np.arange(n) * 1.0, u=u, vbar=v, vm=v, temp_case=np.full(n, 25.0),
                      ah=(soc - 1) * Q, soc=soc, Q=Q, dt=1.0, t_amb_eff=25.0)
    fit = R.fit_ecm([rec], ocv)
    assert fit["R0"] == pytest.approx(0.03, rel=1e-3) and fit["tau2_s"] == pytest.approx(500.0, rel=1e-2)
    assert fit["v_rmse_mV"] < 1e-3


def test_plant_matches_bin_average_model():
    """The torch plant at constant temperature reproduces the RC algebra used in fit_ecm."""
    p = dict(R.Pan18650PF.PARAMS, R0=0.03, R1=0.02, C1=1000.0, R2=0.05, C2=10000.0, Ea_R=0.0, dUdT=0.0,
             o0=3.5, o1=0.6)
    pl = R.Pan18650PF(**p)
    x = torch.tensor([[0.5, 0.01, 0.02, 0.0, 0.0]], dtype=torch.float64)
    u = torch.tensor([[2.0]], dtype=torch.float64)
    f = pl.f(x, u)[0]
    assert f[0] == pytest.approx(-60 * 2.0 / (3600 * 2.9))
    assert f[1] == pytest.approx(60 * (2.0 - 0.01 / 0.02) / 1000.0)
    vt = float(pl.ocv(x[:, 0])) - 0.01 - 0.02 - 0.03 * 2.0
    assert f[4] == pytest.approx(60 * (vt - R.V_REF) / R.TAU_F)


@pytest.mark.parametrize("kind", ["ekf", "ukf"])
def test_grid_filter_equals_individual_filters(kind):
    from hyperkkl import realdata_run as RR
    from hyperkkl.observers import run
    p = dict(R.Pan18650PF.PARAMS, Q=3.0, R0=0.03, R1=0.02, C1=1000.0, R2=0.05, C2=10000.0, o0=3.3, o1=0.9)
    pl = R.Pan18650PF(sensors=(3, 4), **p)
    scales = {"x_std": [0.2, 0.02, 0.03, 0.6, 0.2]}
    grid = [{"q_soc": 1e-3, "c": 0.1, "r_vm": 0.01}, {"q_soc": 1e-2, "c": 1.0, "r_vm": 0.1}]
    torch.manual_seed(0)
    E, N1 = 3, 25
    U = 3 * torch.rand(E, N1, 1, dtype=torch.float64)
    Y = torch.stack([0.5 * torch.rand(E, N1), 0.1 * torch.randn(E, N1)], -1).double()
    X0 = torch.tensor([[0.6, 0.0, 0.0, 0.1, 0.0]] * E, dtype=torch.float64)
    f = RR.grid_filter(kind, pl, scales, grid, E)
    xb = run(f, Y.repeat(2, 1, 1), U.repeat(2, 1, 1), X0.repeat(2, 1))
    for i, g in enumerate(grid):
        xi = run(RR.make_filter(kind, pl, scales, g), Y, U, X0)
        assert torch.allclose(xb[i * E:(i + 1) * E], xi, atol=1e-10, rtol=1e-8)
    assert not torch.allclose(xb[:E], xb[E:])
