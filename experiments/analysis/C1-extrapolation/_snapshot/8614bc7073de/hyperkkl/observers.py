"""Observers behind one contract, so every comparison is like-for-like.

Contract (tests/test_observers.py locks it down). An episode has N+1 samples:

    obs.reset(B, x0_hat, u_0)       ->  xhat[0] = obs.estimate()
    for k = 0 .. N-1:
        obs.advance(y_k, u_k, y_{k+1}, u_{k+1})
        xhat[k+1] = obs.estimate()

so xhat[k] uses measurements up to and including y_k and never beyond. Every observer
starts from the same prior x0_hat (the learned KKL family via z0 = T0(x0_hat)).

Sampled-data semantics (AAAI AI reviewer: "the continuous-time interpretation of the
generated weights"): the context c_k is computed at sample times and held constant on
[t_k, t_{k+1}) (zero-order hold) -- the same hold through which the plant receives its
input (data.integrate), so every observer sees exactly the plant's input; the latent
filter integrates y exactly under linear
interpolation; a latent injection Phi(z, c_k) is integrated by the exponential midpoint
rule. The decoder is evaluated at sample times only.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch.func import jacrev, jvp, vmap


def _t(a, ref):
    return torch.as_tensor(a, dtype=ref.dtype, device=ref.device)


class LatentStep:
    """Exact latent filter step, shared by training rollouts and deployment."""

    def __init__(self, des, device):
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.Ad, self.W0, self.W1, self.Wg = (f(des.step[k]) for k in ("Ad", "W0", "W1", "Wg"))
        self.A, self.B, self.dt = f(des.A), f(des.B), float(des.dt)

    def __call__(self, z, y_k, y_k1, phi=None):
        """phi: callable z -> Phi(z) held over the step, or None."""
        lin = z @ self.Ad.T + y_k @ self.W0.T + y_k1 @ self.W1.T
        if phi is None:
            return lin
        p0 = phi(z)
        zh = z + 0.5 * self.dt * (z @ self.A.T + y_k @ self.B.T + p0)
        return lin + phi(zh) @ self.Wg.T


class Observer:
    name = "observer"
    learned = False
    uses_model = False

    def reset(self, B, x0_hat, u0):
        raise NotImplementedError

    def estimate(self):
        raise NotImplementedError

    def advance(self, y_k, u_k, y_k1, u_k1):
        raise NotImplementedError


@torch.no_grad()
def run(obs: Observer, y: torch.Tensor, u: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
    obs.reset(y.shape[0], x0_hat, u[:, 0])
    out = [obs.estimate()]
    for k in range(y.shape[1] - 1):
        obs.advance(y[:, k], u[:, k], y[:, k + 1], u[:, k + 1])
        out.append(obs.estimate())
    return torch.nan_to_num(torch.stack(out, 1), nan=1e6, posinf=1e6, neginf=-1e6)


@torch.no_grad()
def time_per_step(obs: Observer, ny, nu, nx, device, B=1, n=200) -> float:
    """Wall-clock microseconds per online step (batch B), CUDA-synchronised."""
    y = torch.zeros(B, ny, device=device)
    u = torch.zeros(B, nu, device=device)
    obs.reset(B, torch.zeros(B, nx, device=device), u)
    for _ in range(10):
        obs.advance(y, u, y, u); obs.estimate()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        obs.advance(y, u, y, u); obs.estimate()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return 1e6 * (time.perf_counter() - t0) / n


# ---------------------------------------------------------------------------
# Learned KKL family
# ---------------------------------------------------------------------------

class KKLObserver(Observer):
    """zhat' = A zhat + B y [+ Phi(zhat, c)],  xhat = T+(zhat; c).

    ``decoder(z, c)`` is a ConditionedDecoder (kind none = autonomous). ``injection``
    is a LatentInjection or None. ``z0_map`` maps x0_hat to the initial latent (T0), or
    None for z0 = 0.
    """
    learned = True

    def __init__(self, name, latent: LatentStep, decoder, ctx=None, injection=None, z0_map=None):
        self.name, self.lat, self.dec, self.ctx = name, latent, decoder, ctx
        self.inj, self.z0_map = injection, z0_map

    def reset(self, B, x0_hat, u0):
        nz = self.lat.A.shape[0]
        self.z = self.z0_map(x0_hat) if self.z0_map is not None else \
            torch.zeros(B, nz, device=x0_hat.device)
        if self.ctx is not None:
            self.state = self.ctx.init_state(B, x0_hat.device)
            self.c, self.state = self.ctx.step(u0, self.state)
        else:
            self.c = torch.zeros(B, 0, device=x0_hat.device)

    def estimate(self):
        return self.dec(self.z, self.c)

    def advance(self, y_k, u_k, y_k1, u_k1):
        phi = (lambda z: self.inj(z, self.c)) if self.inj is not None else None
        self.z = self.lat(self.z, y_k, y_k1, phi)
        if self.ctx is not None:
            self.c, self.state = self.ctx.step(u_k1, self.state)


class ModelBasedKKL(Observer):
    r"""Stationary-transformation controlled KKL with the EXACT model correction
    (Bernard & Andrieu 2018; the "[17]-[21]" baseline CDC R10 asked for):

        zhat' = A zhat + B y + J_T0(xhat) [f(xhat, u) - f(xhat, 0)],  xhat = T0+(zhat)

    Uses the learned autonomous maps and the nominal model online (one JVP per stage).
    """
    name, uses_model = "kkl_mb", True

    def __init__(self, latent: LatentStep, T0, T0inv, model):
        self.lat, self.T0, self.Ti, self.model = latent, T0, T0inv, model

    def reset(self, B, x0_hat, u0):
        self.z = self.T0(x0_hat)

    def estimate(self):
        return self.Ti(self.z)

    def _phi(self, z, u):
        with torch.enable_grad():
            x = self.Ti(z).detach()
            v = self.model.f(x, u) - self.model.f(x, torch.zeros_like(u))
            return jvp(self.T0, (x,), (v,))[1].detach()

    def advance(self, y_k, u_k, y_k1, u_k1):
        self.z = self.lat(self.z, y_k, y_k1, lambda z: self._phi(z, u_k))


class NLOX(KKLObserver):
    """Neural Luenberger observer for systems with inputs (Woelk, Morris & Tang 2026):
    zhat' = A zhat + B y + omega(zhat) u,  xhat = T_dagger(zhat), both MLPs trained
    jointly on rollouts from z0 = 0, model-free. Injected as a KKLObserver whose
    'decoder' ignores c and whose injection is omega(z) u."""
    name = "nlox"


# ---------------------------------------------------------------------------
# Model-based filters
# ---------------------------------------------------------------------------

def _rk4(model, x, u0, um, u1, dt):
    k1 = model.f(x, u0)
    k2 = model.f(x + dt / 2 * k1, um)
    k3 = model.f(x + dt / 2 * k2, um)
    k4 = model.f(x + dt * k3, u1)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _selection(sensors, nx, ref):
    H = torch.zeros(len(sensors), nx, dtype=ref.dtype, device=ref.device)
    for i, s in enumerate(sensors):
        H[i, s] = 1.0
    return H


class _Filter(Observer):
    uses_model = True

    def __init__(self, model, dt, q_std, r_std, p0_std):
        self.m, self.dt = model, float(dt)
        self.q2 = np.asarray(q_std, float) ** 2 * dt          # per-step process covariance
        self.r2 = np.asarray(r_std, float) ** 2
        self.p02 = np.asarray(p0_std, float) ** 2

    def _base_reset(self, B, x0_hat):
        self.x = x0_hat.clone()
        nx = x0_hat.shape[1]
        self.P = torch.diag(_t(self.p02, x0_hat)).expand(B, nx, nx).clone()
        self.Q = torch.diag(_t(self.q2, x0_hat).clamp(min=1e-12))
        self.R = torch.diag(_t(self.r2, x0_hat).clamp(min=1e-12))
        self.H = _selection(self.m.sensors, nx, x0_hat)
        self._fresh = True

    def estimate(self):
        return self.x

    def _first(self, y0):
        """Assimilate y_0 before the first prediction, so xhat[k] = E[x_k | y_0..y_k]
        (the exact Kalman filter on a linear-Gaussian plant; xhat[0] stays the prior)."""
        if self._fresh:
            self._update(y0)
            self._fresh = False

    def _update(self, y):
        H, P = self.H, self.P
        S = H @ P @ H.T + self.R
        K = P @ H.T @ torch.linalg.inv(S)
        self.x = self.x + (K @ (y - self.x @ H.T)[:, :, None])[:, :, 0]
        I = torch.eye(P.shape[1], dtype=P.dtype, device=P.device)
        A = I - K @ H
        self.P = A @ P @ A.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)   # Joseph form
        # a non-finite state is reported as divergence (1e6), never restarted at 0
        self.x = torch.nan_to_num(self.x, nan=1e6, posinf=1e6, neginf=-1e6)


class EKF(_Filter):
    """Continuous-discrete EKF, RK4 prediction, oracle Q and R (the true noise levels)."""
    name = "ekf"

    def reset(self, B, x0_hat, u0):
        self._base_reset(B, x0_hat)
        f1 = lambda xi, ui: self.m.f(xi[None], ui[None])[0]
        self.jac = vmap(jacrev(f1, argnums=0))

    def advance(self, y_k, u_k, y_k1, u_k1):
        self._first(y_k)
        with torch.enable_grad():
            F = self.jac(self.x, u_k)
        Phi = torch.matrix_exp(F * self.dt)
        self.x = _rk4(self.m, self.x, u_k, u_k, u_k, self.dt)          # ZOH input, as the plant
        self.P = Phi @ self.P @ Phi.transpose(1, 2) + self.Q
        self._update(y_k1)


class UKF(_Filter):
    """Unscented KF (Merwe scaled sigma points, alpha=1, beta=2, kappa=0); the
    measurement is a linear selection so the update is exact."""
    name = "ukf"

    def reset(self, B, x0_hat, u0):
        self._base_reset(B, x0_hat)
        n = x0_hat.shape[1]
        a, b, k = 1.0, 2.0, 0.0
        self.lam = a * a * (n + k) - n
        wm = torch.full((2 * n + 1,), 1.0 / (2 * (n + self.lam)), dtype=x0_hat.dtype,
                        device=x0_hat.device)
        wc = wm.clone()
        wm[0] = self.lam / (n + self.lam)
        wc[0] = wm[0] + (1 - a * a + b)
        self.wm, self.wc = wm, wc

    def advance(self, y_k, u_k, y_k1, u_k1):
        self._first(y_k)
        B, n = self.x.shape
        P = 0.5 * (self.P + self.P.transpose(1, 2)) + 1e-9 * torch.eye(n, device=self.x.device)
        L = torch.linalg.cholesky_ex((n + self.lam) * P)[0]
        sp = torch.cat([self.x[:, None], self.x[:, None] + L.transpose(1, 2),
                        self.x[:, None] - L.transpose(1, 2)], 1)          # (B, 2n+1, n)
        rep = lambda v: v[:, None].expand(-1, 2 * n + 1, -1).reshape(-1, v.shape[-1])
        X = _rk4(self.m, sp.reshape(-1, n), rep(u_k), rep(u_k), rep(u_k),
                 self.dt).view(B, 2 * n + 1, n)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        self.x = (self.wm[None, :, None] * X).sum(1)
        d = X - self.x[:, None]
        self.P = torch.einsum("s,bsi,bsj->bij", self.wc, d, d) + self.Q
        self._update(y_k1)


class ParticleFilter(_Filter):
    """Bootstrap particle filter with systematic resampling when ESS < N/2. The
    proposal's process noise is inflated by  1 + (inflate - 1) e^{-t / anneal_tau}
    (constant `inflate` when anneal_tau = 0) to avoid sample impoverishment; a large
    initial inflation that anneals to 1 is what makes it track chaotic plants. The
    schedule is tuned per system on VALIDATION episodes (job.tune_pf)."""
    name = "pf"

    def __init__(self, *a, n_particles=512, inflate=3.0, anneal_tau=0.0, seed=0, **kw):
        super().__init__(*a, **kw)
        self.Np, self.inflate, self.seed = int(n_particles), float(inflate), int(seed)
        self.anneal_tau = float(anneal_tau)

    def reset(self, B, x0_hat, u0):
        self._base_reset(B, x0_hat)
        self.g = torch.Generator(device=x0_hat.device).manual_seed(self.seed)
        n = x0_hat.shape[1]
        sd = _t(np.sqrt(self.p02), x0_hat)
        self.X = x0_hat[:, None] + sd * torch.randn(B, self.Np, n, generator=self.g,
                                                     device=x0_hat.device)
        self.logw = torch.zeros(B, self.Np, device=x0_hat.device)
        self.qs = _t(np.sqrt(self.q2), x0_hat).clamp(min=1e-9)
        self.ri = 1.0 / _t(self.r2, x0_hat).clamp(min=1e-12)
        self.k = 0
        self.x = x0_hat.clone()

    def _first(self, y0):
        if self._fresh:                                          # weight the prior by y_0
            e = y0[:, None] - self.X[:, :, list(self.m.sensors)]
            self.logw = self.logw - 0.5 * (e * e * self.ri).sum(-1)
            self._fresh = False

    def advance(self, y_k, u_k, y_k1, u_k1):
        self._first(y_k)
        B, Np, n = self.X.shape
        rep = lambda v: v[:, None].expand(-1, Np, -1).reshape(-1, v.shape[-1])
        X = _rk4(self.m, self.X.reshape(-1, n), rep(u_k), rep(u_k), rep(u_k),
                 self.dt).view(B, Np, n)
        t = self.k * self.dt
        infl = (1 + (self.inflate - 1) * math.exp(-t / self.anneal_tau)) if self.anneal_tau > 0 else self.inflate
        self.k += 1
        X = X + infl * self.qs * torch.randn(X.shape, generator=self.g, device=X.device)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        e = y_k1[:, None] - X[:, :, list(self.m.sensors)]
        self.logw = self.logw - 0.5 * (e * e * self.ri).sum(-1)
        self.logw = self.logw - self.logw.max(1, keepdim=True).values
        w = torch.softmax(self.logw, 1)
        self.x = (w[:, :, None] * X).sum(1)
        ess = 1.0 / (w * w).sum(1)
        res = ess < Np / 2
        if res.any():                                            # systematic resampling
            cdf = torch.cumsum(w[res], 1)
            cdf[:, -1] = 1.0
            r0 = torch.rand(int(res.sum()), 1, generator=self.g, device=X.device) / Np
            pos = r0 + torch.arange(Np, device=X.device)[None] / Np
            idx = torch.searchsorted(cdf, pos).clamp(max=Np - 1)
            X[res] = torch.gather(X[res], 1, idx[:, :, None].expand(-1, -1, n))
            self.logw[res] = 0.0
        self.X = X
