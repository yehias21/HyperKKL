"""Unit tests for the pipeline study (CPU, synthetic tensors)."""
import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperkkl import pipeline as PL  # noqa: E402
from hyperkkl.conditioning import ConditionedDecoder, build_conditioner, match_width  # noqa: E402
from hyperkkl.nets import BaseMap, n_params  # noqa: E402

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))


def _base(nz=5, nx=2, w=128, d=3):
    torch.manual_seed(0)
    st = lambda n: (torch.randn(n) * 0.1, torch.rand(n) + 0.5)
    b = BaseMap(nz, nx, w, d, "tanh", st(nz), st(nx))
    for q in b.parameters():
        q.requires_grad_(False)
    return b


def _perturb(m, scale=0.05):
    with torch.no_grad():
        for q in m.parameters():
            q.add_(scale * torch.randn_like(q))


def test_equal_compute_units():
    units = {a: PL.compute_plan(a, CFG)["units"] for a in PL.ARMS}
    assert len(set(units.values())) == 1, units
    assert units["P0"] == 6000 + 6000 + 3000
    p2 = PL.compute_plan("P2", CFG)
    assert p2["joint_steps"] * 3 == units["P0"] and p2["per_loss_evals"]["p2"] == 5000


@pytest.mark.parametrize("kind", ["lora", "hyper"])
def test_zero_input_collapse_exact_P0_P4(kind):
    """P0 and P4 use the unchanged frozen-base ConditionedDecoder: cond(z, 0) == T0+(z) bitwise,
    also after the conditioner was trained (generators perturbed)."""
    base = _base()
    z_std = torch.ones(5)
    w, _ = match_width(kind, base, 16, 140000, 4, z_std)
    cond = build_conditioner(kind, base, 16, w, 4, z_std)
    _perturb(cond)                       # perturbs only conditioner (base has no grad but add_ works) ->
    with torch.no_grad():                # restore the base so the reference is the frozen phase-1 map
        cond.base.load_state_dict(base.state_dict())
    z = torch.randn(512, 5)
    out = cond(z, torch.zeros(512, 16))
    assert torch.equal(out, base(z))
    assert not torch.equal(cond(z, torch.randn(512, 16)), base(z))


@pytest.mark.parametrize("kind", ["lora", "hyper"])
def test_P1_counts_and_collapse_error(kind):
    base = _base()
    Ti = copy.deepcopy(base)
    z_std = torch.ones(5)
    w, npar = match_width(kind, base, 16, 140000, 4, z_std)
    assert abs(npar - 140000) / 140000 < 0.02
    cond = build_conditioner(kind, Ti, 16, w, 4, z_std)
    side = [q for q in cond.parameters() if q.requires_grad]
    assert sum(q.numel() for q in side) == npar             # base excluded from the conditioner count
    for q in Ti.parameters():
        q.requires_grad_(True)
    nb = n_params(Ti)
    assert nb == n_params(base, trainable_only=False) and nb > 0
    _perturb(cond)                                          # base AND conditioner move (P1/P2 training)
    z = torch.randn(512, 5)
    c0 = torch.zeros(512, 16)
    assert torch.equal(cond(z, c0), cond.base(z))           # structural: Delta(c=0) = 0 vs its OWN base
    dev = (cond(z, c0) - base(z)).abs().max()
    assert dev > 0                                           # but the zero-input map left phase 1's: reported


def test_P3_budget_and_init():
    base = _base()
    w, npar = PL.match_uncond_width(base, 140000)
    assert abs(npar - 140000) / 140000 < 0.02
    m = PL.UncondDecoder(copy.deepcopy(base), w)
    z = torch.randn(256, 5)
    assert torch.equal(m(z), base(z))                       # zero-initialised residual: starts at phase 1


def test_collapse_helper_reports_exact():
    from hyperkkl.data import EpisodeSet
    base = _base()
    E, N = 4, 40
    es = EpisodeSet(x=torch.randn(E, N, 2), u=torch.zeros(E, N, 1), y_clean=torch.zeros(E, N, 1),
                    y=torch.zeros(E, N, 1), z=torch.randn(E, N, 5), z_obs=torch.randn(E, N, 5),
                    families=["zero"] * E, burn=10, dt=0.1)
    r = PL.collapse(ConditionedDecoder(base, "none", 0, 0), base, es, torch.ones(2), dc=0)
    assert r["exact"] and r["max_abs_dev_sigma"] == 0.0
