"""Model-based reference filters (standalone port of hyperkkl/observers.py: EKF, UKF, chi-square
gate, Huber update). They use ONLY the modeller's model of the knowledge level (context['model'],
nominal parameters) and the nominal noise description (context q_std, r_std, prior).

    EKF(gate=False)      continuous-discrete EKF: RK4 / zero-order-hold prediction, Phi = expm(F dt),
                         Joseph-form update. Q = q_std^2 dt, R = r_std^2, P0 = prior_std^2.
    EKF(gate=True)       + chi-square innovation gate (p = 0.999, NIS on the valid components); after
                         5 consecutive rejections the next update is forced (lock-out guard). This is
                         exactly the gate of the research gating study (experiments/gating), plus the
                         stale-sample rule below. Known failure: it can lock out on a chaotic plant
                         during the initial transient (lorenz/exact). Optional safeguards
                         (gate_warmup_tref, warmup_huber, lockout_reset, lockout_huber) exist, but none
                         of their combinations dominated on a 9-cell comparison (PITFALLS P12), so the
                         reference keeps the published rule.
    EKF(huber=True)      Huber M-estimation update, no gate: never locks out; reported next to it.
    all filters          stale samples (exact repeats: held / frozen sensors) are skipped.
    UKF(...)             Merwe sigma points (alpha 1, beta 2, kappa 0); same options.
    q_inflate, r_inflate tuning knobs (choose them on ``val`` / ``val_ood`` only; PROTOCOL.md).

Online contract: step(u_k, y_k) -> (x_hat_k, P_k): for k >= 1 predict over [t_{k-1}, t_k] with the
held input u_{k-1}, then update with y_k; at k = 0 update the prior with y_0.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.func import jacrev, vmap

from ..estimator import BaseEstimator

torch_dtype = torch.float64


def _rk4(model, x, u, dt):
    k1 = model.f(x, u)
    k2 = model.f(x + dt / 2 * k1, u)
    k3 = model.f(x + dt / 2 * k2, u)
    k4 = model.f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class _KF(BaseEstimator):
    returns_covariance = True
    kind = "ekf"

    def __init__(self, gate: bool = True, huber: bool = False, q_inflate: float = 1.0,
                 r_inflate: float = 1.0, gate_p: float = 0.999, gate_max_reject: int = 5,
                 huber_k: float = 1.345, irls_iters: int = 3, hold_missing: bool | None = None,
                 gate_warmup_tref: float = 0.0, warmup_huber: bool = True, lockout_reset: bool = False,
                 lockout_huber: bool = False, name: str | None = None):
        self.gate, self.huber = bool(gate), bool(huber)
        self.q_inflate, self.r_inflate = float(q_inflate), float(r_inflate)
        self.gate_p, self.gate_max_reject = float(gate_p), int(gate_max_reject)
        self.huber_k, self.irls_iters = float(huber_k), int(irls_iters)
        self._hold_arg = hold_missing
        self.gate_warmup_tref = float(gate_warmup_tref)     # no gating during the first ... t_ref (transient)
        self.warmup_huber = bool(warmup_huber)              # ... but a Huber (bounded-influence) update instead
        self.lockout_reset = bool(lockout_reset)            # re-open P when the lock-out guard forces an update
        self.lockout_huber = bool(lockout_huber)            # ... and make that forced update a Huber update
        self.name = name or (self.kind + ("_huber" if huber else "") + ("_gated" if gate else ""))

    # ------------------------------------------------------------------ setup
    def fit(self, train, val, context):
        self.context = context
        self.m = context["model"]
        if self.m is None:
            raise ValueError(f"{self.name} needs the modeller's model (context['model'])")
        self.dt = float(context["dt"])
        self._gate_warm = int(round(self.gate_warmup_tref * float(context["meta"].get("t_ref", 1.0)) / self.dt))
        # stale-sample rule: a measurement component exactly equal to its previous sample is treated as
        # missing (multi-rate sample-and-hold sensors, frozen sensors). With continuous-valued noisy
        # sensors exact repeats do not occur otherwise; disable it for quantised sensors.
        self.hold_missing = bool(context.get("hold_missing", True)) if self._hold_arg is None else bool(self._hold_arg)
        q = np.asarray(context["q_std"], float) ** 2 * self.dt * self.q_inflate
        r = np.asarray(context["r_std"], float) ** 2 * self.r_inflate
        self.Q = torch.diag(torch.as_tensor(q, dtype=torch_dtype).clamp(min=1e-12))
        self.R = torch.diag(torch.as_tensor(r, dtype=torch_dtype).clamp(min=1e-12))
        nx = len(context["x_std"])
        self.H = torch.zeros(len(context["sensors"]), nx, dtype=torch_dtype)
        for i, s in enumerate(context["sensors"]):
            self.H[i, s] = 1.0
        ny = self.H.shape[0]
        from scipy.stats import chi2
        self._chi2 = torch.as_tensor([math.inf] + [float(chi2.ppf(self.gate_p, d)) for d in range(1, ny + 1)],
                                     dtype=torch_dtype)

    def _reset_b(self, X0, P0std):
        B, nx = X0.shape
        self.x = torch.as_tensor(X0, dtype=torch_dtype).clone()
        p0 = torch.as_tensor(P0std, dtype=torch_dtype) ** 2
        self.P = torch.diag_embed(p0 if p0.ndim == 2 else p0.expand(B, nx)).clone()
        self._run_rej = torch.zeros(B, dtype=torch.long)
        self._p0var = (torch.as_tensor(P0std, dtype=torch_dtype) ** 2).reshape(-1, nx).mean(0)
        self.n_lockout_resets = 0
        self._first = True
        self._uprev = None
        self._ylast = None
        self._k = 0
        self.n_rejected = 0

    # ------------------------------------------------------------------ update
    def _update(self, y):
        H, P, R = self.H, self.P, self.R
        B, n = self.x.shape
        ny = H.shape[0]
        I = torch.eye(n, dtype=torch_dtype)
        valid = torch.ones(B, ny, dtype=torch.bool)
        if self.hold_missing and self._ylast is not None:      # multi-rate sensors: a repeated sample is stale
            valid = valid & (y != self._ylast)
        self._ylast = y.clone()
        e0 = torch.where(valid, y - self.x @ H.T, torch.zeros_like(y))
        m0 = valid.to(torch_dtype)
        rd = torch.diagonal(R)
        forced = torch.zeros(B, dtype=torch.bool)
        if self.gate and self._k >= self._gate_warm:
            Hm = m0[:, :, None] * H
            S = Hm @ P @ Hm.transpose(1, 2) + torch.diag_embed(torch.where(valid, rd, torch.ones_like(rd)))
            nis = (e0[:, None, :] @ torch.linalg.solve_ex(S, e0[:, :, None])[0])[:, 0, 0]
            rej = nis > self._chi2[valid.sum(1)]
            if self.gate_max_reject > 0:
                forced = rej & (self._run_rej >= self.gate_max_reject)
                rej = rej & (self._run_rej < self.gate_max_reject)
                if self.lockout_reset and bool(forced.any()):
                    # lock-out: the estimate disagrees with every recent measurement, so its covariance is
                    # too small; re-open it (add the prior covariance) before the forced update
                    P = torch.where(forced[:, None, None], P + torch.diag(self._p0var), P)
                    self.P = P
                    self.n_lockout_resets += int(forced.sum())
            self._run_rej = torch.where(rej, self._run_rej + 1, torch.zeros_like(self._run_rej))
            self.n_rejected += int(rej.sum())
            m0 = m0 * (~rej).to(torch_dtype)[:, None]
        sig = rd.sqrt()
        x0, xn, w = self.x, self.x, m0
        warm = self.gate and self.warmup_huber and self._k < self._gate_warm
        rows = torch.full((B,), bool(self.huber or warm), dtype=torch.bool)
        if self.lockout_huber:
            rows = rows | forced                          # a forced (lock-out) update is a Huber update
        huber = bool(rows.any())
        iters = max(1, self.irls_iters) if huber else 1
        for _ in range(iters):
            if huber:
                r = (y - xn @ H.T) / sig
                hw = torch.clamp(self.huber_k / r.abs().clamp(min=1e-300), max=1.0)
                w = m0 * torch.where(rows[:, None], hw, torch.ones_like(hw))
            sw = w.sqrt()
            Hw = sw[:, :, None] * H
            S = Hw @ P @ Hw.transpose(1, 2) + R
            K = P @ Hw.transpose(1, 2) @ torch.linalg.inv_ex(S)[0]     # _ex: a diverged episode never raises
            xn = x0 + (K @ (sw * e0)[:, :, None])[:, :, 0]
        A = I - K @ Hw
        self.P = torch.nan_to_num(A @ P @ A.transpose(1, 2) + K @ R @ K.transpose(1, 2), nan=1e12, posinf=1e12, neginf=-1e12)
        self.x = torch.nan_to_num(xn, nan=1e6, posinf=1e6, neginf=-1e6)

    def _predict(self, u):
        raise NotImplementedError

    def _step_b(self, u, y):
        u = torch.as_tensor(u, dtype=torch_dtype).reshape(self.x.shape[0], -1)     # (B, n_u): scalar or vector input
        y = torch.as_tensor(y, dtype=torch_dtype).reshape(self.x.shape[0], -1)
        if not self._first:
            self._predict(self._uprev)
        self._first = False
        self._update(y)
        self._uprev = u
        self._k += 1
        return self.x, self.P

    # ------------------------------------------------------------------ protocol
    @torch.no_grad()
    def reset(self, x0_prior, P0_prior=None):
        p0 = self.context["prior_std"] if P0_prior is None else P0_prior
        self._reset_b(np.asarray(x0_prior, float)[None], np.asarray(p0, float)[None])

    @torch.no_grad()
    def step(self, u, y):
        x, P = self._step_b(np.asarray([u], float), np.asarray(y, float)[None])
        return x[0].numpy().copy(), P[0].numpy().copy()

    @torch.no_grad()
    def run_batch(self, U, Y, X0, P0=None, chunk: int = 256):
        E, N1 = U.shape[:2]
        nx = X0.shape[1]
        Xh = np.empty((E, N1, nx))
        Pd = np.empty((E, N1, nx, nx))
        p0 = np.tile(self.context["prior_std"], (E, 1)) if P0 is None else P0
        for c in range(0, E, chunk):
            sl = slice(c, c + chunk)
            self._reset_b(X0[sl], p0[sl])
            for k in range(N1):
                x, P = self._step_b(U[sl, k], Y[sl, k])
                Xh[sl, k] = x.numpy()
                Pd[sl, k] = P.numpy()
        return Xh, Pd

    def n_params(self):
        return 0

    def state_bytes(self):
        n = len(self.context["x_std"])
        return 8 * (n + n * n)


class EKF(_KF):
    kind = "ekf"

    def _jac(self, x, u):
        if not hasattr(self, "_jf"):
            f1 = lambda xi, ui: self.m.f(xi[None], ui[None])[0]
            self._jf = vmap(jacrev(f1, argnums=0))
        with torch.enable_grad():
            return self._jf(x, u)

    def _predict(self, u):
        F = self._jac(self.x, u)
        Phi = torch.matrix_exp(F * self.dt)
        self.x = _rk4(self.m, self.x, u, self.dt)
        self.P = Phi @ self.P @ Phi.transpose(1, 2) + self.Q

    def flops_per_step(self):
        n, ny = len(self.context["x_std"]), len(self.context["sensors"])
        # rough count: Jacobian (n f-evals ~ 20 n FLOPs each), expm (~ 20 n^3), 2 n^3 propagation, update
        return int(n * 20 * n + 20 * n ** 3 + 4 * n ** 3 + 6 * n * n * ny + 2 * ny ** 3 + 4 * 20 * n)


class UKF(_KF):
    kind = "ukf"

    def _weights(self, n):
        a, b, k = 1.0, 2.0, 0.0
        lam = a * a * (n + k) - n
        wm = torch.full((2 * n + 1,), 1.0 / (2 * (n + lam)), dtype=torch_dtype)
        wc = wm.clone()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1 - a * a + b)
        return lam, wm, wc

    def _predict(self, u):
        B, n = self.x.shape
        lam, wm, wc = self._weights(n)
        P = 0.5 * (self.P + self.P.transpose(1, 2)) + 1e-9 * torch.eye(n, dtype=torch_dtype)
        L = torch.linalg.cholesky_ex((n + lam) * P)[0]
        sp = torch.cat([self.x[:, None], self.x[:, None] + L.transpose(1, 2),
                        self.x[:, None] - L.transpose(1, 2)], 1)
        ur = u[:, None].expand(-1, 2 * n + 1, -1).reshape(-1, u.shape[-1])
        X = _rk4(self.m, sp.reshape(-1, n), ur, self.dt).view(B, 2 * n + 1, n)
        X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        self.x = (wm[None, :, None] * X).sum(1)
        d = X - self.x[:, None]
        self.P = torch.einsum("s,bsi,bsj->bij", wc, d, d) + self.Q

    def flops_per_step(self):
        n, ny = len(self.context["x_std"]), len(self.context["sensors"])
        return int((2 * n + 1) * 4 * 20 * n + n ** 3 / 3 + 2 * (2 * n + 1) * n * n + 6 * n * n * ny + 2 * ny ** 3)


class TunedKF(BaseEstimator):
    """A filter whose (q_inflate, r_inflate) are chosen on the VAL split (PROTOCOL.md section 3):
    each candidate runs on ``n_val`` validation episodes from the nominal prior; the score is the
    mean settled unmeasured NRMSE (diverged episodes count clipped at 100). The grid and the scores
    are kept in ``fit_info``. Model error at partial / mismatch / realistic shows up as process noise,
    so inflating Q is the classical, legitimate repair."""
    returns_covariance = True

    def __init__(self, kind="ekf", gate=True, q_grid=(1, 10, 100, 1000), r_grid=(1, 3), n_val=8, name=None, **kw):
        self.kind, self.gate, self.q_grid, self.r_grid, self.n_val, self.kw = kind, gate, q_grid, r_grid, n_val, kw
        self.name = name or f"{kind}{'_gated' if gate else ''}_tuned"

    def _make(self, q, r):
        cls = EKF if self.kind == "ekf" else UKF
        return cls(gate=self.gate, q_inflate=q, r_inflate=r, **self.kw)

    def fit(self, train, val, context):
        from ..metrics import episode_metrics
        self.context = context
        meta = context["meta"]
        sub = val.subset(np.arange(min(self.n_val, len(val))))
        # real-data tracks give a per-episode prior (es.prior / es.prior_std) and no context["prior_mean"]
        # (PITFALLS P28): select from the same prior the scorer starts the test episodes from
        if getattr(sub, "prior", None) is not None:
            X0 = np.asarray(sub.prior, float)
            P0 = np.tile(np.asarray(sub.prior_std, float), (len(sub), 1))
        else:
            X0, P0 = np.tile(context["prior_mean"], (len(sub), 1)), None
        lab = np.isfinite(sub.x).all(axis=(0, 1))                      # never score unlabelled states (P13)
        scores = {}
        with torch.no_grad():
            for q in self.q_grid:
                for r in self.r_grid:
                    f = self._make(q, r)
                    f.fit(train, val, context)
                    xh, _ = f.run_batch(sub.u.astype(float), sub.y.astype(float), X0, P0)
                    if lab.all():
                        m = episode_metrics(sub.x, xh, meta["x_std"], int(meta["settle_index"]), meta["sensors"])
                    else:
                        keep = [j for j in range(sub.x.shape[-1]) if lab[j] and j not in set(meta["sensors"])]
                        m = episode_metrics(sub.x[..., keep], xh[..., keep], np.asarray(meta["x_std"], float)[keep],
                                            int(meta["settle_index"]), [])
                        m["nrmse_unmeasured"] = m["nrmse"]
                    v = m["nrmse_unmeasured"] if np.isfinite(m["nrmse_unmeasured"]).all() else m["nrmse"]
                    scores[f"q{q}_r{r}"] = float(np.mean(v))
        best = min(scores, key=scores.get)
        q, r = (float(t[1:]) for t in best.split("_"))
        self.inner = self._make(q, r)
        self.inner.fit(train, val, context)
        self.fit_info = {"chosen": {"q_inflate": q, "r_inflate": r}, "val_scores": scores, "n_val": len(sub)}

    def reset(self, x0_prior, P0_prior=None):
        return self.inner.reset(x0_prior, P0_prior)

    def step(self, u, y):
        return self.inner.step(u, y)

    def run_batch(self, U, Y, X0, P0=None):
        return self.inner.run_batch(U, Y, X0, P0)

    def n_params(self):
        return 2

    def state_bytes(self):
        return self.inner.state_bytes()

    def flops_per_step(self):
        return self.inner.flops_per_step()
