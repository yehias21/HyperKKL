"""bench_v2 unit tests.   .venv/bin/python -m pytest experiments/bench_v2/code/tests -q

CPU only. Data tests skip cells that are not built yet.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import env  # noqa: E402,F401
import torch  # noqa: E402

import profiles_v2 as PV  # noqa: E402
import v2cfg  # noqa: E402

DATA = env.DATA


# ------------------------------------------------------------------------------------------------ profiles
@pytest.mark.parametrize("plant", ["two_link_arm_v2", "highway_v2", "bioreactor"])
@pytest.mark.parametrize("split", ["train", "ood"])
def test_v2_profiles_in_box_and_rate(plant, split):
    P = PV.SPECS[plant]
    a = P["act"]
    for op in PV.operators(plant, split):
        for s in range(3):
            pr = PV.generate(plant, split, s, operator=op.__name__)
            assert np.isfinite(pr.u).all()
            assert pr.u.min() >= a.lo - 1e-9 and pr.u.max() <= a.hi + 1e-9
            if math.isfinite(a.rate):
                assert np.abs(np.diff(pr.u)).max() <= a.rate * P["dt"] + a.quant + 1e-9


def test_arm_torque_box_is_physical():
    a = PV.SPECS["two_link_arm_v2"]["act"]
    g1 = (1.0 * 0.5 + 1.0 * 1.0) * 9.81          # shoulder gravity torque of the nominal arm
    assert a.hi == 12.0 and a.lo == -12.0 and 0.5 * g1 < a.hi < g1
    # operators actually use the widened box (not the legacy +-1 N m)
    peak = max(np.abs(PV.generate("two_link_arm_v2", "train", s).u).max() for s in range(10))
    assert peak > 3.0


def test_highway_box_widened():
    P = PV.SPECS["highway_v2"]
    assert (P["lo"], P["hi"]) == (-0.25, 0.2)          # ramp flow 0.05-0.5 veh/s around 0.3


def test_astat_has_no_cited_acceleration():
    import inspect
    src = inspect.getsource(PV.op_astat)
    assert "0.004" not in src and "0.012" not in src
    assert PV.op_astat in PV.SPECS["bioreactor"]["ood"]


def test_ood_operators_differ_from_train():
    for p in ("battery", "vdp", "two_link_arm_v2", "highway_v2", "quad_tank", "bioreactor"):
        tr = {o.__name__ for o in PV.operators(p, "train")}
        od = {o.__name__ for o in PV.operators(p, "ood")}
        assert tr and od and not (tr & od), p


# ------------------------------------------------------------------------------------------------ config
def test_family_membership():
    assert set(v2cfg.FAMILY_A) == {"vdp", "fhn", "rossler", "lorenz", "lorenz96", "network", "lorenz96_n10"}
    for p in ("duffing", "battery", "bioreactor", "cartpole_b", "quad_tank", "highway_v2", "two_link_arm_v2"):
        assert p not in v2cfg.FAMILY_A
    with pytest.raises(ValueError):
        v2cfg.config("autonomous", "lorenz", "realistic")
    with pytest.raises(AssertionError):
        v2cfg.config("autonomous", "battery", "white")


def test_v2_plant_configs():
    c = v2cfg.config("driven", "two_link_arm_v2", "realistic")["system"]
    assert c["u_scale"] == 12.0 and c["plant"] == "two_link_arm_b"
    from hyperkkl import systems_bounded  # noqa: F401
    from hyperkkl.systems import build_plant
    pl = build_plant(c)
    assert abs(pl.u_bar - 24.0) < 1e-9                  # actuator saturates at 2 x the box, as every plant
    h = v2cfg.config("driven", "highway_v2", "white")["system"]
    assert h["u_bounds"] == [-0.25, 0.2] and abs(build_plant(h).u_bar - 0.5) < 1e-9


# ------------------------------------------------------------------------------------------------ autonomous context
def test_autonomous_context_constant():
    """u = 0: the HyperKKL input context (SFM encoder, as configured) is the same vector for every episode and time,
    so the hypernetwork generates ONE static decoder (DESIGN.md section 2)."""
    from hyperkkl import data as Dm
    from hyperkkl.nets import build_context
    cfg = v2cfg.config("autonomous", "vdp", "white")
    des = Dm.make_design(cfg["system"])
    p = cfg["phase2"]["context"]
    torch.manual_seed(0)
    ctx = build_context(p["kind"], p, 1, des.dt, np.linalg.eigvals(des.A), 1.0)
    u = torch.zeros(5, 400, 1)
    c = ctx.seq(u)
    assert torch.allclose(c, c[:1, :1].expand_as(c), atol=1e-6)


# ------------------------------------------------------------------------------------------------ conditions
class _Cell:
    family, plant, variant = "driven", "vdp", "realistic"

    def __init__(self):
        from hyperkkl import data as Dm
        self.cfg = v2cfg.config("driven", "vdp", "realistic")
        self.des = Dm.make_design(self.cfg["system"])
        self.t_ref = 1.0
        self.sc = {"y_std": [1.0]}


def _es(E=4, N1=200, ny=1, nx=2):
    from hyperkkl import data as Dm
    z = torch.zeros(E, N1, 3)
    return Dm.EpisodeSet(torch.zeros(E, N1, nx), torch.zeros(E, N1, 1), torch.zeros(E, N1, ny),
                         torch.randn(E, N1, ny), z, z, ["f"] * E, 0, 0.1, [{} for _ in range(E)], 0)


def test_conditions_sensor():
    import conditions_v2 as C
    cell, es = _Cell(), _es()
    _, Y, U, x0 = C.apply(cell, es, "test_op_id", "outliers")
    frac = float(((Y - es.y).abs() > 6).float().mean())
    assert 0.005 < frac < 0.05
    _, Y2, _, _ = C.apply(cell, es, "test_op_id", "outliers")
    assert torch.equal(Y, Y2)                                  # deterministic
    _, Yb, _, _ = C.apply(cell, es, "test_op_id", "burst_outage")
    k0 = int(round(0.6 * 199))
    assert torch.equal(Yb[:, k0 + 1], Yb[:, k0]) and torch.equal(Yb[:, :k0], es.y[:, :k0])
    _, _, _, xp = C.apply(cell, es, "test_op_id", "poor_init")
    box = np.asarray(cell.cfg["system"]["ic_box"], float)
    assert np.allclose(np.abs(xp.numpy() - box.mean(1)), 0.9 * (box[:, 1] - box[:, 0]) / 2, atol=1e-5)


# ------------------------------------------------------------------------------------------------ built data
def _cells():
    out = []
    for m in sorted(DATA.glob("*/*/*/MANIFEST.json")):
        out.append(m.parent)
    return out


@pytest.mark.parametrize("d", _cells() or [None])
def test_built_cell(d):
    if d is None:
        pytest.skip("no cell built")
    man = json.loads((d / "MANIFEST.json").read_text())
    fam, plant = man["family"], man["plant"]
    tr = torch.load(d / "train.pt", weights_only=False)
    # cells whose generator drops envelope-leaving episodes after 6 redraws keep fewer (rossler; DATASHEET), but the
    # manifest must record it and at least 80 % must remain
    assert man["splits"]["train"]["n"] == len(tr) and len(tr) >= 480 and torch.isfinite(tr.x).all()
    if len(tr) < 600:
        assert man["splits"]["train"]["rejected"] > 0
    if fam == "autonomous":
        for s in ("train", "train_event", "test_id", "test_ic_ood", "test_param_shift"):
            if man["splits"].get(s, {}).get("unavailable"):
                assert not (d / f"{s}.pt").exists()
                continue
            es = torch.load(d / f"{s}.pt", weights_only=False)
            assert float(es.u.abs().max()) == 0.0, s
        ood = torch.load(d / "test_ic_ood.pt", weights_only=False)
        box = np.asarray(v2cfg.config(fam, plant, man["variant"])["system"]["ic_box"], float)
        c, h = box.mean(1), (box[:, 1] - box[:, 0]) / 2
        r = np.abs((np.asarray([m["ic"] for m in ood.meta]) - c) / h).max(1)
        assert (r >= 0.8 - 1e-6).all()                         # outer shell, disjoint from the core training box
    else:
        P = PV.SPECS[plant]
        for s in ("train", "test_op_id", "test_op_ood"):
            es = torch.load(d / f"{s}.pt", weights_only=False)
            assert float(es.u.min()) >= P["lo"] - 1e-5 and float(es.u.max()) <= P["hi"] + 1e-5, s
        ood = torch.load(d / "test_op_ood.pt", weights_only=False)
        ops = {m["operator"] for m in ood.meta}
        assert ops <= {o.__name__ for o in PV.operators(plant, "ood")}
    # splits are disjoint draws: no identical trajectory shared between train and test
    te = torch.load(d / ("test_id.pt" if fam == "autonomous" else "test_op_id.pt"), weights_only=False)
    assert not any(torch.equal(te.x[0], tr.x[i]) for i in range(len(tr)))


def test_singular_innovation_fallback():
    """A singular S in one batch element must not abort the batch (network UKF crash, 2026-11-04); a regular S gives
    the plain solve up to the 1e-9 relative jitter that is applied only when some element is singular."""
    from hyperkkl.observers import safe_inv, safe_solve
    S = torch.tensor([[[2.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float64)
    b = torch.ones(2, 2, 1, dtype=torch.float64)
    x = safe_solve(S, b)
    assert torch.isfinite(x).all() and torch.allclose(x[0, :, 0], torch.tensor([0.5, 1.0], dtype=torch.float64), rtol=1e-6)
    assert torch.equal(safe_solve(S[:1], b[:1]), torch.linalg.solve(S[:1], b[:1]))      # plain path when regular
    assert torch.equal(safe_inv(S[:1]), torch.linalg.inv(S[:1]))
    safe_inv(torch.full((1, 2, 2), float("nan"), dtype=torch.float64))        # NaN S: no exception (NaN -> diverged)
