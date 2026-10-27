"""Truncated-history SFM context (phase2.context.history = H; experiments/memory/PREREG.md).

    xi_k = sum_{l=1}^{min(H,k)} Ad^{l-1} Bd u_{k-l}/u_ref        (inputs older than H samples -> 0)

The same truncation in seq() (training) and step() (deployment); H = None is the untouched SFM.
"""
import numpy as np
import pytest
import torch

from hyperkkl import design as D
from hyperkkl.job import method_cfg
from hyperkkl.nets import SFMEncoder, build_context, n_params

torch.manual_seed(0)
A, _ = D.butterworth_AB(5, 1.5)
POLES = np.linalg.eigvals(A)
DT = 0.1 * np.sin(np.pi / 10) / 1.5


def _sfm(history=None, floor=0.0, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    e = SFMEncoder(1, 16, 64, DT, POLES, 8, u_ref=0.7, floor=floor, history=history).to(dtype)
    with torch.no_grad():                                 # away from the symmetric initialisation
        e.w_in.add_(0.3 * torch.randn_like(e.w_in))
        e.omega.add_(0.2 * torch.randn_like(e.omega))
        e.raw_sigma.add_(0.3 * torch.randn_like(e.raw_sigma))
    return e


def _run_step(e, u):
    st = e.init_state(u.shape[0], u.device)
    out = []
    for k in range(u.shape[1]):
        c, st = e.step(u[:, k], st)
        out.append(c)
    return torch.stack(out, 1)


@pytest.mark.parametrize("H", [1, 3, 16, 50])
def test_truncation_is_exact_per_sample(H):
    """c_k of the truncated context equals c_k of the FULL context fed with the input whose
    samples older than H (relative to k) are zero."""
    full, trunc = _sfm(None), _sfm(H)
    trunc.load_state_dict(full.state_dict())
    u = torch.randn(3, 120, 1, dtype=torch.float64)
    ct = trunc.seq(u)
    for k in (0, 1, H - 1, H, H + 1, 60, 119):
        um = u.clone()
        um[:, :max(0, k - H)] = 0.0                       # keep u_{k-H..k}, zero everything older
        cf = full.seq(um)[:, k]
        assert torch.allclose(ct[:, k], cf, atol=1e-12, rtol=0), (k, (ct[:, k] - cf).abs().max())
    # and it differs from the full context where the history is longer than H
    assert (ct[:, H + 5:] - full.seq(u)[:, H + 5:]).abs().max() > 1e-6


@pytest.mark.parametrize("H", [1, 7, 64])
def test_step_equals_seq(H):
    e = _sfm(H)
    u = torch.randn(4, 150, 1, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(e.seq(u), _run_step(e, u), atol=1e-12, rtol=0)
    e32 = _sfm(H, dtype=torch.float32)
    e32.load_state_dict({k: v.float() for k, v in e.state_dict().items()})
    u32 = u.float()
    with torch.no_grad():
        d = (e32.seq(u32) - _run_step(e32, u32)).abs().max().item()
    assert d < 1e-4, d


def test_history_none_and_history_beyond_episode_are_bit_identical_to_plain_sfm():
    plain = _sfm(None)
    for H in (None, 200, 10_000):
        e = _sfm(H)
        e.load_state_dict(plain.state_dict())
        u = torch.randn(2, 200, 1, dtype=torch.float64)
        with torch.no_grad():
            assert torch.equal(e.seq(u), plain.seq(u)), H
            if H is None:
                assert torch.equal(_run_step(e, u), _run_step(plain, u))
            else:
                assert torch.allclose(_run_step(e, u), _run_step(plain, u), atol=1e-12, rtol=0)


def test_parameter_count_independent_of_history():
    counts = {H: n_params(_sfm(H)) for H in (None, 1, 32, 512, 2048)}
    assert len(set(counts.values())) == 1, counts
    assert counts[None] == 8 + 8 + 16 + 17 * 64 + 64 * 64 + 64 * 16


@pytest.mark.parametrize("H", [1, 5])
def test_jacobian_support_is_exactly_the_last_H_plus_one_samples(H):
    e = _sfm(H)
    N = 30
    u = torch.randn(N, 1, dtype=torch.float64)
    J = torch.autograd.functional.jacobian(lambda v: e.seq(v[None])[0], u)      # (N, dc, N, 1)
    Jn = J[..., 0].norm(dim=1)                                                 # (k, j)
    k, j = np.indices((N, N))
    Jn = Jn.numpy()
    assert Jn[j > k].max() <= 1e-13 * Jn.max()                                 # causal (FFT round-off)
    assert Jn[j < k - H].max() <= 1e-13 * Jn.max()                             # nothing older than H
    assert Jn[(j >= k - H) & (j <= k)].min() > 1e-6                            # every one of the last H+1 counts


def test_floor_and_history_compose():
    e = _sfm(16, floor=0.5)
    assert e.alpha() >= 0.5 * 1.5 - 1e-9
    u = torch.randn(2, 100, 1, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(e.seq(u), _run_step(e, u), atol=1e-12, rtol=0)
        assert e.seq(torch.zeros(2, 50, 1, dtype=torch.float64)).abs().max() == 0.0


def test_build_context_and_method_string():
    ccfg = {"dim": 16, "hidden": 64, "sfm_blocks": 8, "history": 64}
    e = build_context("sfm", ccfg, 1, DT, POLES, 1.0)
    assert isinstance(e, SFMEncoder) and e.history == 64
    assert build_context("sfm", {"dim": 16, "hidden": 64}, 1, DT, POLES, 1.0).history is None
    cfg = {"phase2": {"context": {"kind": "sfm", "dim": 16, "hidden": 64, "sfm_blocks": 8}}}
    kind, mc = method_cfg(cfg, "lora@sfm:history=8")
    assert kind == "lora" and mc["phase2"]["context"]["history"] == 8 and mc["phase2"]["context"]["kind"] == "sfm"
    kind, mc = method_cfg(cfg, "hyper")
    assert kind == "hyper" and "history" not in mc["phase2"]["context"]
