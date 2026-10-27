"""Networks: base KKL maps and causal context encoders.

Base maps
    BaseMap      normalise -> MLP -> denormalise. Encoder T0: x -> z (SiLU, it is
                 differentiated by the physics loss, so it must be C-infinity; ReLU would
                 make the residual gradient wrong). Decoder T0+: z -> x (tanh).

Context encoders  c_k = E(u_0, ..., u_k)
    current   c_k = psi(u_k)                      no history
    window    c_k = psi(u_{k-L+1..k})             finite window of L samples (FIR)
    gru       c_k = P h_k,  h = GRU(u)            unbounded recurrent memory
    sfm       c_k = psi([xi_k, u_k]),  xi' = A_xi xi + B_xi u, A_xi Hurwitz by construction
              (spectral fading memory; spectrum initialised from spec(A))

Every encoder satisfies E(0) = 0 exactly: psi is bias-free and zero-preserving, and the
GRU's zero-input response is subtracted (bitwise in step() and on CUDA; GRU seq() on CPU
is ~1e-8, the zero row shares one GEMM). Every encoder exposes the SAME computation two
ways, which tests/test_core.py and tests/test_verify_models.py check agree:
    seq(u)            whole episode at once (training)
    step(u_k, state)  one sample at a time (deployment)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT = {"silu": F.silu, "tanh": torch.tanh, "gelu": F.gelu, "softplus": F.softplus}


class Affine(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-8))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width, depth, act="silu", bias=True, zero_last=False):
        super().__init__()
        dims = [d_in] + [width] * depth + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=bias) for a, b in zip(dims[:-1], dims[1:]))
        self.act = ACT[act]
        if zero_last:
            nn.init.zeros_(self.layers[-1].weight)
            if bias:
                nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class BaseMap(nn.Module):
    """y = out.denorm(MLP(in.norm(x))). Layers are exposed for conditioned forwards."""

    def __init__(self, d_in, d_out, width, depth, act, in_stats, out_stats):
        super().__init__()
        self.inp = Affine(*in_stats)
        self.out = Affine(*out_stats)
        self.mlp = MLP(d_in, d_out, width, depth, act)

    @property
    def layers(self):
        return self.mlp.layers

    @property
    def act(self):
        return self.mlp.act

    def forward(self, x):
        return self.out.denorm(self.mlp(self.inp.norm(x)))


def n_params(m: nn.Module, trainable_only=True) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------
# Context encoders
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """Interface. Inputs are divided by ``u_ref`` so every plant presents O(1) inputs."""
    dim: int

    def seq(self, u: torch.Tensor) -> torch.Tensor:          # (B, N+1, nu) -> (B, N+1, dc)
        raise NotImplementedError

    def init_state(self, B, device):
        raise NotImplementedError

    def step(self, u_k, state):                               # -> (c_k, next_state)
        raise NotImplementedError


class CurrentEncoder(ContextEncoder):
    def __init__(self, nu, dim, hidden, u_ref=1.0):
        super().__init__()
        self.dim, self.u_ref = dim, float(u_ref)
        self.psi = MLP(nu, dim, hidden, 2, "tanh", bias=False)

    def seq(self, u):
        return self.psi(u / self.u_ref)

    def init_state(self, B, device):
        return None

    def step(self, u_k, state):
        return self.psi(u_k / self.u_ref), None


class WindowEncoder(ContextEncoder):
    """Finite window of L samples: a learned causal FIR bank followed by psi."""

    def __init__(self, nu, dim, hidden, L, u_ref=1.0):
        super().__init__()
        self.dim, self.L, self.nu, self.u_ref = dim, int(L), nu, float(u_ref)
        self.fir = nn.Conv1d(nu, hidden, self.L, bias=False)
        nn.init.normal_(self.fir.weight, std=1.0 / math.sqrt(self.L * nu))
        self.psi = MLP(hidden, dim, hidden, 1, "tanh", bias=False)

    def seq(self, u):
        v = F.pad((u / self.u_ref).transpose(1, 2), (self.L - 1, 0))   # causal, zero history
        return self.psi(torch.tanh(self.fir(v).transpose(1, 2)))

    def init_state(self, B, device):
        return torch.zeros(B, self.nu, self.L, device=device, dtype=self.fir.weight.dtype)

    def step(self, u_k, buf):
        buf = torch.cat([buf[:, :, 1:], (u_k / self.u_ref)[:, :, None]], 2)
        return self.psi(torch.tanh(self.fir(buf)[:, :, 0])), buf


class GRUEncoder(ContextEncoder):
    """c = P (h(u) - h(0)): the zero-input response of the GRU is subtracted, so the
    context is exactly zero at zero input despite the GRU's internal biases."""

    def __init__(self, nu, dim, hidden, u_ref=1.0):
        super().__init__()
        self.dim, self.hidden, self.u_ref = dim, hidden, float(u_ref)
        self.rnn = nn.GRU(nu, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def seq(self, u):
        B = u.shape[0]
        both = torch.cat([u / self.u_ref, torch.zeros_like(u[:1])], 0)
        h, _ = self.rnn(both)
        return self.proj(h[:B] - h[B:])

    def init_state(self, B, device):
        z = torch.zeros(1, B, self.hidden, device=device, dtype=self.proj.weight.dtype)
        return (z, z.clone())

    def step(self, u_k, state):
        h, h0 = state
        o, h = self.rnn((u_k / self.u_ref)[:, None], h)
        o0, h0 = self.rnn(torch.zeros_like(u_k)[:, None], h0)
        return self.proj(o[:, 0] - o0[:, 0]), (h, h0)


class SFMEncoder(ContextEncoder):
    r"""Spectral fading-memory context.

        xi_{k+1} = Ad xi_k + Bd u_k   (exact ZOH of xi' = A_xi xi + B_xi u)
        c_k      = psi([xi_k, u_k])   (bias-free)

    A_xi = blockdiag([[s_b, w_b], [-w_b, s_b]]) with s_b = -softplus(.): Hurwitz for every
    parameter value, each block normal, so for l >= 1 (u in units of u_ref)
    ||d c_k / d u_{k-l}|| <= l_psi ||Bd|| e^{-alpha_xi (l-1) dt}   (Lemma 2 in THEORY.md;
    d xi_k / d u_{k-l} = Ad^{l-1} Bd, and the bound is attained, tests/test_verify_models.py).
    """

    def __init__(self, nu, dim, hidden, dt, poles, n_blocks=8, u_ref=1.0, learn=True):
        super().__init__()
        self.dim, self.nu, self.nb, self.dt, self.u_ref = dim, nu, n_blocks, float(dt), float(u_ref)
        p = np.asarray(poles)
        p = p[p.imag >= -1e-12]
        p = p[np.argsort(-p.real)]                              # slowest first
        sig, om = np.full(n_blocks, -1.0), np.zeros(n_blocks)
        for b in range(n_blocks):
            if b < len(p):
                sig[b], om[b] = p[b].real, abs(p[b].imag)
            else:                                               # slower fill-in modes
                sig[b] = p.real.max() / (1.5 ** (b - len(p) + 1))
        raw = np.log(np.expm1(np.clip(-sig, 1e-4, None)))
        self.raw_sigma = nn.Parameter(torch.tensor(raw, dtype=torch.float32), requires_grad=learn)
        self.omega = nn.Parameter(torch.tensor(om, dtype=torch.float32), requires_grad=learn)
        w_in = torch.zeros(n_blocks, 2, nu)
        w_in[:, 0] = 1.0
        self.w_in = nn.Parameter(w_in)
        self.psi = MLP(2 * n_blocks + nu, dim, hidden, 2, "tanh", bias=False)

    def sigma(self):
        return -F.softplus(self.raw_sigma)

    def alpha(self) -> float:
        return float(-self.sigma().detach().max())

    def _ad_bd(self):
        s, w, dt = self.sigma(), self.omega, self.dt
        e, c, sn = torch.exp(s * dt), torch.cos(w * dt), torch.sin(w * dt)
        Ad = torch.stack([torch.stack([e * c, e * sn], -1), torch.stack([-e * sn, e * c], -1)], -2)
        den = s ** 2 + w ** 2 + 1e-12
        Ainv = torch.stack([torch.stack([s, -w], -1), torch.stack([w, s], -1)], -2) / den[:, None, None]
        Bd = Ainv @ (Ad - torch.eye(2, device=s.device)) @ self.w_in        # (nb, 2, nu)
        return Ad, Bd

    def impulse(self, n: int) -> torch.Tensor:
        """H[l] = Ad^l Bd, shape (n, 2 nb, nu)."""
        s, w, dt = self.sigma(), self.omega, self.dt
        _, Bd = self._ad_bd()
        l = torch.arange(n, device=s.device, dtype=s.dtype)[:, None] * dt
        e, c, sn = torch.exp(s[None] * l), torch.cos(w[None] * l), torch.sin(w[None] * l)
        B0, B1 = Bd[:, 0], Bd[:, 1]
        r0 = (e * c)[..., None] * B0 + (e * sn)[..., None] * B1
        r1 = -(e * sn)[..., None] * B0 + (e * c)[..., None] * B1
        return torch.stack([r0, r1], 2).reshape(n, 2 * self.nb, self.nu)

    def filter_seq(self, u):
        """xi_k for k = 0..N (xi_0 = 0) by FFT convolution."""
        B, N1, _ = u.shape
        H = self.impulse(N1)                                    # (N1, m, nu)
        n_fft = 1 << int(math.ceil(math.log2(2 * N1)))
        U = torch.fft.rfft(u, n=n_fft, dim=1)                   # (B, F, nu)
        Hf = torch.fft.rfft(H, n=n_fft, dim=0)                  # (F, m, nu)
        s = torch.fft.irfft(torch.einsum("bfu,fmu->bfm", U, Hf), n=n_fft, dim=1)[:, :N1]
        return torch.cat([torch.zeros_like(s[:, :1]), s[:, :-1]], 1)   # xi_k uses u_{<k}

    def seq(self, u):
        un = u / self.u_ref
        return self.psi(torch.cat([self.filter_seq(un), un], -1))

    def init_state(self, B, device):
        self._adbd = None            # parameters may have changed since the last episode
        return torch.zeros(B, 2 * self.nb, device=device, dtype=self.w_in.dtype)

    def step(self, u_k, xi):
        un = u_k / self.u_ref
        c = self.psi(torch.cat([xi, un], -1))
        if torch.is_grad_enabled() or getattr(self, "_adbd", None) is None:
            Ad, Bd = self._ad_bd()
            if not torch.is_grad_enabled():   # deployment: compute once per episode, not per step
                self._adbd = (Ad, Bd)
        else:
            Ad, Bd = self._adbd
        xb = xi.view(-1, self.nb, 2)
        nxt = torch.einsum("nij,bnj->bni", Ad, xb) + torch.einsum("niu,bu->bni", Bd, un)
        return c, nxt.reshape(-1, 2 * self.nb)

    def psi_lipschitz(self) -> float:
        p = 1.0
        for layer in self.psi.layers:
            p *= float(torch.linalg.matrix_norm(layer.weight.detach(), 2))
        return p


def build_context(kind: str, ccfg: dict, nu: int, dt: float, poles, u_ref: float) -> ContextEncoder:
    dim, hid = int(ccfg["dim"]), int(ccfg["hidden"])
    if kind == "current":
        return CurrentEncoder(nu, dim, hid, u_ref)
    if kind == "window":
        return WindowEncoder(nu, dim, hid, int(ccfg["window"]), u_ref)
    if kind == "gru":
        return GRUEncoder(nu, dim, hid, u_ref)
    if kind == "sfm":
        return SFMEncoder(nu, dim, hid, dt, poles, int(ccfg.get("sfm_blocks", 8)), u_ref)
    raise ValueError(f"unknown context encoder {kind}")
