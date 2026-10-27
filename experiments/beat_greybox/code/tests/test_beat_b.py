"""Stage-B1 hybrid observer (hyperkkl/beat_b.py), CPU, toy linear plant."""
import numpy as np
import torch

from hyperkkl.beat_b import HybridKKL
from hyperkkl.observers import EKF, Observer, run


class Lin:
    nx, nu, ny, sensors = 2, 1, 1, [0]

    def f(self, x, u):
        A = torch.tensor([[0.0, 1.0], [-1.0, -0.3]], dtype=x.dtype, device=x.device)
        return x @ A.T + torch.cat([torch.zeros_like(u), u], -1)

    def h(self, x):
        return x[:, :1]


class Replay(Observer):
    """A 'KKL' observer that outputs a given trajectory (one sample per advance)."""
    def __init__(self, X):
        self.X = X

    def reset(self, B, x0, u0):
        self.k = 0

    def estimate(self):
        return self.X[:, self.k]

    def advance(self, *a):
        self.k += 1


def _sim(n=300, dt=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = Lin()
    x = torch.tensor([[1.0, -0.5], [0.3, 0.8]])
    u = torch.sin(0.3 * torch.arange(n + 1.0))[None, :, None].expand(2, -1, -1).contiguous()
    xs = [x]
    for k in range(n):
        x = x + dt * m.f(x, u[:, k]); xs.append(x)
    X = torch.stack(xs, 1)
    Y = X[..., :1] + 0.05 * torch.randn(X[..., :1].shape, generator=g)
    return m, X, Y, u, dt


def test_hybrid_with_huge_c_is_the_ekf():
    m, X, Y, U, dt = _sim()
    x0 = torch.zeros(2, 2)
    q, r, p0 = np.full(2, 0.01), np.full(1, 0.05), np.ones(2)
    e = run(EKF(m, dt, q, r, p0), Y, U, x0)
    h = run(HybridKKL(Replay(torch.zeros_like(X)), m, dt, q, r, p0, np.ones(2), c=1e14), Y, U, x0)
    assert torch.allclose(e, h, atol=1e-4), float((e - h).abs().max())


def test_hybrid_follows_an_accurate_kkl():
    m, X, Y, U, dt = _sim()
    x0 = torch.zeros(2, 2)
    q, r, p0 = np.full(2, 0.01), np.full(1, 0.05), np.ones(2)
    h = run(HybridKKL(Replay(X), m, dt, q, r, p0, np.full(2, 1e-4), c=1.0), Y, U, x0)
    e = run(EKF(m, dt, q, r, p0), Y, U, x0)
    err_h = float((h - X)[:, 20:].pow(2).mean().sqrt()); err_e = float((e - X)[:, 20:].pow(2).mean().sqrt())
    assert err_h < 0.2 * err_e and err_h < 1e-3, (err_h, err_e)


def test_hybrid_is_causal():
    m, X, Y, U, dt = _sim()
    q, r, p0 = np.full(2, 0.01), np.full(1, 0.05), np.ones(2)
    ob = HybridKKL(Replay(X), m, dt, q, r, p0, np.full(2, 0.1))
    a = run(ob, Y, U, torch.zeros(2, 2))
    Y2 = Y.clone(); Y2[:, 151:] += 3.0
    X2 = X.clone(); X2[:, 151:] += 3.0
    ob2 = HybridKKL(Replay(X2), m, dt, q, r, p0, np.full(2, 0.1))
    b = run(ob2, Y2, U, torch.zeros(2, 2))
    assert torch.equal(a[:, :151], b[:, :151])
