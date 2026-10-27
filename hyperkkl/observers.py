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

    def __init__(self, model, dt, q_std, r_std, p0_std, robust=None, gate=False,
                 huber_k=1.345, irls_iters=3, gate_p=0.999, gate_max_reject=5,
                 skip_nan=False, hold_missing=False):
        """robust / gate / skip_nan / hold_missing select the robust measurement update
        (``_robust_update``); with their defaults the update is the plain Kalman update,
        bit-identical to the filters without these options.

        robust       None | "huber": Huber M-estimation update (IRLS), threshold huber_k
        gate         chi-square innovation gate: skip the update of an episode whose
                     NIS = e' S^-1 e exceeds chi2_{ny}(gate_p). After gate_max_reject
                     consecutive rejections the next update is accepted (guards against
                     the classic gate lock-out: a bad estimate with a small P rejecting
                     every good measurement thereafter); 0 = never force.
        skip_nan     non-finite measurement components are treated as missing (default
                     False: a NaN measurement poisons the state, which run() reports as a
                     divergence -- the behaviour of the plain filters)
        hold_missing a component exactly equal to the previous sample is treated as
                     missing (the dropout condition is a sample-and-hold, data.measure);
                     do NOT combine with a quantised sensor, where repeats are genuine
        """
        self.m, self.dt = model, float(dt)
        self.q2 = np.asarray(q_std, float) ** 2 * dt          # per-step process covariance
        self.r2 = np.asarray(r_std, float) ** 2
        self.p02 = np.asarray(p0_std, float) ** 2
        if robust not in (None, "huber"):
            raise ValueError(f"robust must be None or 'huber', got {robust!r}")
        self.robust, self.gate, self.huber_k = robust, bool(gate), float(huber_k)
        self.irls_iters, self.gate_p = int(irls_iters), float(gate_p)
        self.gate_max_reject = int(gate_max_reject)
        self.skip_nan, self.hold_missing = bool(skip_nan), bool(hold_missing)
        self._plain = robust is None and not gate and not skip_nan and not hold_missing

    def _base_reset(self, B, x0_hat):
        self.x = x0_hat.clone()
        nx = x0_hat.shape[1]
        self.P = torch.diag(_t(self.p02, x0_hat)).expand(B, nx, nx).clone()
        self.Q = torch.diag(_t(self.q2, x0_hat).clamp(min=1e-12))
        self.R = torch.diag(_t(self.r2, x0_hat).clamp(min=1e-12))
        self.H = _selection(self.m.sensors, nx, x0_hat)
        self._fresh = True
        self._ylast = None
        self.n_rejected = 0                                   # gated measurement updates
        self._run_rej = torch.zeros(B, dtype=torch.long, device=x0_hat.device)

    def estimate(self):
        return self.x

    def _first(self, y0):
        """Assimilate y_0 before the first prediction, so xhat[k] = E[x_k | y_0..y_k]
        (the exact Kalman filter on a linear-Gaussian plant; xhat[0] stays the prior)."""
        if self._fresh:
            self._update(y0)
            self._fresh = False

    def _update(self, y):
        if getattr(self, "_plain", True):
            self._kf_update(y)
        else:
            self._robust_update(y)

    def _kf_update(self, y):
        H, P = self.H, self.P
        S = H @ P @ H.T + self.R
        K = P @ H.T @ torch.linalg.inv(S)
        self.x = self.x + (K @ (y - self.x @ H.T)[:, :, None])[:, :, 0]
        I = torch.eye(P.shape[1], dtype=P.dtype, device=P.device)
        A = I - K @ H
        self.P = A @ P @ A.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)   # Joseph form
        # a non-finite state is reported as divergence (1e6), never restarted at 0
        self.x = torch.nan_to_num(self.x, nan=1e6, posinf=1e6, neginf=-1e6)

    def _robust_update(self, y):
        r"""Robust measurement update, batched over episodes (no loop over episodes).

        Huber (robust="huber"): the posterior mean is the M-estimate

            x+ = argmin_x  (x - x-)' P-^{-1} (x - x-) + 2 sum_i rho_k((y_i - H_i x) / sigma_i)

        with Huber's rho_k (quadratic for |r| <= k, linear beyond), i.e. the measurement
        block of the regression-form Huber filter (Karlgaard & Schaub 2007) with the
        prior block left quadratic. It is solved by iteratively reweighted least squares
        from x- (irls_iters Kalman updates with R_eff = R / w, w_i = min(1, k / |r_i|),
        r_i the R-standardised residual of the current iterate); each IRLS step is a
        Kalman update, so a measurement with every |r_i| <= k gives the plain KF update.
        The covariance is the Joseph form with the final R_eff (= (P^-1 + H' R_eff^-1 H)^-1).

        Weights enter as rows sqrt(w) H and sqrt(w) e with R unchanged (algebraically
        R / w), so w = 0 (gated / missing) is exact rather than R = inf.

        Gate: NIS = e' S^-1 e with S = H P- H' + R over the valid components; an
        episode with NIS > chi2_{n_valid}(gate_p) keeps its prior (x-, P-), unless it
        was rejected gate_max_reject times in a row. Gating is applied before (and is
        independent of) the Huber reweighting.
        """
        H, P = self.H, self.P
        B, n = self.x.shape
        dt_, dev = P.dtype, P.device
        ny = H.shape[0]
        I = torch.eye(n, dtype=dt_, device=dev)
        rd = torch.diagonal(self.R)                                     # (ny,)
        sig = rd.sqrt()
        valid = torch.ones(B, ny, dtype=torch.bool, device=dev)
        if self.skip_nan:
            valid = valid & torch.isfinite(y)
        if self.hold_missing and self._ylast is not None:
            valid = valid & (y != self._ylast)
        self._ylast = y.clone()
        yv = torch.where(valid, y, torch.zeros_like(y)) if (self.skip_nan or self.hold_missing) else y
        e0 = torch.where(valid, yv - self.x @ H.T, torch.zeros_like(yv))
        m0 = valid.to(dt_)
        if self.gate:
            Hm = m0[:, :, None] * H                                     # (B, ny, n)
            S = Hm @ P @ Hm.transpose(1, 2) + torch.diag_embed(torch.where(valid, rd, torch.ones_like(rd)))
            nis = (e0[:, None, :] @ torch.linalg.solve(S, e0[:, :, None]))[:, 0, 0]
            if getattr(self, "_chi2", None) is None or self._chi2.dtype != dt_ \
                    or self._chi2.device != dev:
                from scipy.stats import chi2                          # table, built once
                self._chi2 = torch.as_tensor(
                    [math.inf] + [float(chi2.ppf(self.gate_p, d)) for d in range(1, ny + 1)],
                    dtype=dt_, device=dev)
            thr = self._chi2[valid.sum(1)]                              # by # valid comps
            rej = nis > thr                                             # NaN NIS: not rejected
            if self.gate_max_reject > 0:                                # lock-out guard
                rej = rej & (self._run_rej < self.gate_max_reject)
            self._run_rej = torch.where(rej, self._run_rej + 1, torch.zeros_like(self._run_rej))
            self.n_rejected = self.n_rejected + rej.sum()               # tensor: no host sync
            m0 = m0 * (~rej).to(dt_)[:, None]
        x0, xn, w = self.x, self.x, m0
        iters = max(1, self.irls_iters) if self.robust == "huber" else 1
        for _ in range(iters):
            if self.robust == "huber":
                r = (yv - xn @ H.T) / sig                               # current iterate
                hw = torch.clamp(self.huber_k / r.abs(), max=1.0)       # psi(r) / r
                w = m0 * torch.where(r.abs() > 0, hw, torch.ones_like(hw))
            sw = w.sqrt()
            Hw = sw[:, :, None] * H                                     # (B, ny, n)
            S = Hw @ P @ Hw.transpose(1, 2) + self.R
            K = P @ Hw.transpose(1, 2) @ torch.linalg.inv(S)
            xn = x0 + (K @ (sw * e0)[:, :, None])[:, :, 0]
        A = I - K @ Hw
        self.P = A @ P @ A.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)
        self.x = torch.nan_to_num(xn, nan=1e6, posinf=1e6, neginf=-1e6)


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


# ---------------------------------------------------------------------------
# Robust filters and condition-aware oracle noise (reviewer C3: "the EKF was only
# beaten because it was mis-tuned for the stress condition")
# ---------------------------------------------------------------------------

class HuberEKF(EKF):
    """EKF with the Huber M-estimation measurement update (_Filter._robust_update).
    Keyword options of _Filter (huber_k, irls_iters, gate, gate_p, skip_nan,
    hold_missing) pass through."""
    name = "ekf_huber"

    def __init__(self, *a, **kw):
        kw.setdefault("robust", "huber")
        super().__init__(*a, **kw)


class HuberUKF(UKF):
    """UKF with the Huber M-estimation measurement update (the measurement is a linear
    selection, so the robust update of _Filter applies to the UKF prior unchanged)."""
    name = "ukf_huber"

    def __init__(self, *a, **kw):
        kw.setdefault("robust", "huber")
        super().__init__(*a, **kw)


# E[mean_k b_k^2] of data.measure's drift profile b (a Gaussian random walk normalised to
# max |b| = 1 over the episode); Monte Carlo, 4000 walks: 0.255 (N=200), 0.244 (N=1000),
# 0.241 (N=4000).
DRIFT_MEAN_SQUARE = 0.24


def oracle_noise(sensor_cfg: dict, scales: dict, noise_proc: float = 0.0, dt: float = None,
                 model_err=None, q_inflate: float = 1.0, r_inflate: float = 1.0,
                 outliers: str = "mixture", colored: str = "variance",
                 drift: str = "ignore", r_floor: float = 1e-3):
    r"""Condition-aware oracle noise levels for the model-based filters.

    Returns ``(q_std, r_std)`` per state / per output in the units the filter
    constructors take: ``q_std`` is a std RATE (per-step covariance q^2 dt) and ``r_std``
    a per-sample std, i.e. ``EKF(model, dt, q_std, r_std, p0)``.

    Measurement variance, per output, relative to y_std^2 (data.measure's sensor model):

        sd^2                      white or coloured noise (the AR(1) of data.measure has
                                  unit stationary variance, for kind colored and for the
                                  Student-t driven colored_t). colored="longrun" uses the
                                  long-run variance sd^2 (1 + rho) / (1 - rho) instead --
                                  the equivalent white level when the filter bandwidth is
                                  well below the noise bandwidth.
      + quant^2 / 12              uniform quantisation error
      + rate * scale^2            outlier mixture (zero-mean, +-scale); outliers="ignore"
                                  leaves it out (the right choice for the Huber filter,
                                  which needs the NOMINAL sigma as its scale)
      + 0.24 drift^2              drift="variance": the mean square of the normalised
                                  random-walk bias (DRIFT_MEAN_SQUARE); "ignore" (default)
                                  leaves it out. A bias is not white noise; the
                                  principled fix is a bias-augmented state.

    then times r_inflate (a scalar to be tuned on VALIDATION episodes, default 1). The
    noise floor max(sd, r_floor) matches evaluate.build_observers for sd = 0.

    Process noise: q_eff^2 = (q_inflate * noise_proc * x_std)^2 + model_err^2 * dt, where
    ``model_err`` (per state) is the RMS model-error rate f_true - f_nom (see
    ``model_error_rate``), treated as white with per-step covariance (model_err dt)^2 --
    a deterministic drift is not white, so this is a heuristic; q_inflate is the scalar
    fallback, tuned on validation episodes.

    Dropout (sample-and-hold) changes neither Q nor R; use ``hold_missing=True`` on the
    filter instead (skip held samples)."""
    s = dict(sensor_cfg or {})
    ys = np.asarray(scales["y_std"], float)
    xs = np.asarray(scales["x_std"], float)
    sd = max(float(s.get("noise", 0.0)), r_floor)
    var = sd ** 2
    kind = s.get("kind", "gaussian")
    if kind in ("colored", "colored_t") and colored == "longrun":
        rho = float(s.get("rho", 0.95 if kind == "colored" else 0.9))
        var = var * (1 + rho) / (1 - rho)
    elif colored not in ("variance", "longrun"):
        raise ValueError(colored)
    var += float(s.get("quant", 0.0)) ** 2 / 12.0
    if outliers == "mixture":
        var += float(s.get("outlier_rate", 0.0)) * float(s.get("outlier_scale", 12.0)) ** 2
    elif outliers != "ignore":
        raise ValueError(outliers)
    if drift == "variance":
        var += DRIFT_MEAN_SQUARE * float(s.get("drift", 0.0)) ** 2
    elif drift != "ignore":
        raise ValueError(drift)
    r_std = np.sqrt(var * float(r_inflate)) * ys
    q2 = (float(q_inflate) * float(noise_proc) * xs) ** 2
    if model_err is not None:
        if dt is None:
            raise ValueError("dt is required with model_err")
        q2 = q2 + np.asarray(model_err, float) ** 2 * float(dt)
    return np.sqrt(q2), r_std


@torch.no_grad()
def model_error_rate(nominal, true, box, u_scale: float = 1.0, n: int = 4096,
                     seed: int = 0) -> np.ndarray:
    """Per-state RMS of f_true(x, u) - f_nom(x, u) over a Sobol sample of the box (an
    (nx, 2) array, e.g. scales["domain"] or system.ic_box) and u ~ U[-u_scale, u_scale].
    Data-free: uses only the two models, no evaluation episodes."""
    from scipy.stats import qmc
    box = np.asarray(box, float)
    m = int(math.ceil(math.log2(max(n, 2))))
    Z = qmc.Sobol(d=len(box) + nominal.nu, scramble=True, seed=seed).random_base2(m)[:n]
    x = torch.as_tensor(box[:, 0] + Z[:, :len(box)] * (box[:, 1] - box[:, 0]), dtype=torch.float64)
    u = torch.as_tensor(u_scale * (2 * Z[:, len(box):] - 1), dtype=torch.float64)
    d = true.f(x, u) - nominal.f(x, u)
    d = d[torch.isfinite(d).all(1)]
    return d.pow(2).mean(0).sqrt().numpy()
