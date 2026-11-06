"""FitzHugh-Nagumo ring of N nodes (n_x = 2N) of the scaling track (copied from
experiments/scaling/code/scaleplants.py, plant part only, rev 3). Registered as fhnring{N}
(research names); the benchmark names are fhnring_n{n_x}."""
from __future__ import annotations

import numpy as np
import torch

from .systems import Plant, register

NX_GRID = (4, 8, 16, 32, 64)
GOLDEN = 0.6180339887498949


def _sig(v):
    return 1.0 / (1.0 + torch.exp(-v))


class FHNRingN(Plant):
    N_NODES = 0
    PARAMS = {"a": 0.7, "b": 0.8, "eps": 0.08, "I_ext": 0.5, "kappa": 0.5, "spread": 1.0,
              "g_syn": 0.15, "E_syn": 2.0, "th_syn": 1.0, "s_syn": 0.25}
    NOT_PERTURBED = ("I_ext",)   # the applied bias current is an operating condition, not plant physics
    UNMODELLED = ("the excitatory chemical synapse g_syn (E_syn - v_i) sum_j a_ij sig((v_j - th)/s) "
                  "between ring neighbours (model: gap-junction coupling only)")

    def __init__(self, sensors=None, **params):
        super().__init__(sensors=sensors, **params)
        self._cache = {}

    @classmethod
    def adjacency(cls) -> np.ndarray:
        N = cls.N_NODES
        A = np.zeros((N, N))
        for i in range(N):
            A[i, (i + 1) % N] += 1.0
            A[i, (i - 1) % N] += 1.0
        return A

    def _consts(self, dtype, device):
        k = (dtype, str(device))
        if k not in self._cache:
            A = torch.as_tensor(self.adjacency(), dtype=dtype, device=device)
            L = A - torch.diag(A.sum(1))
            # rev 3: golden-ratio (low-discrepancy) layout of the recovery rates around the ring, so
            # ring neighbours always differ (a monotone ramp makes long runs of near-identical
            # neighbours that phase-lock in clusters: rev 2 failed boundedness (2a) at n_x = 64)
            i = np.arange(self.N_NODES)
            pos = torch.as_tensor(2 * ((i * GOLDEN) % 1.0) - 1, dtype=dtype, device=device)
            eps = self.eps * torch.exp(self.spread * pos)
            e0 = torch.zeros(self.N_NODES, dtype=dtype, device=device); e0[0] = 1.0
            self._cache[k] = (A, L, eps, e0)
        return self._cache[k]

    def f(self, x, u):
        return self._f(x, u, True)

    def f_known(self, x, u):
        return self._f(x, u, False)

    def _f(self, x, u, synapse: bool):
        A, L, eps, e0 = self._consts(x.dtype, x.device)
        v, w = x[:, 0::2], x[:, 1::2]
        dv = v - v ** 3 / 3 - w + self.I_ext + self.kappa * v @ L + u[:, :1] * e0
        if synapse:
            s = _sig((v - self.th_syn) / self.s_syn)
            dv = dv + self.g_syn * (self.E_syn - v) * (s @ A)          # A symmetric
        dw = eps * (v + self.a - self.b * w)
        return torch.stack([dv, dw], 2).reshape(x.shape[0], self.nx)


def sensors_for(N: int) -> tuple:
    return tuple(2 * i for i in range(0, N, 2))


PLANTS = {}
for _nx in NX_GRID:
    _N = _nx // 2
    _cls = type(f"FHNRing{_N}", (FHNRingN,), {"N_NODES": _N, "nx": _nx,
                                               "DEFAULT_SENSORS": sensors_for(_N)})
    PLANTS[_nx] = register(f"fhnring{_N}")(_cls)


