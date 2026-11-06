"""Standalone grey-box / UDE filter: the strongest comparator of the research study, reproducible
without the research code.

    f_hat(x, u) = f_model(x, u) + r_theta(x, u)        (hybrid=True: UDE / corrected grey-box)
    f_hat(x, u) = r_theta(x, u)                        (hybrid=False: NODE, no physics)

r_theta is a small MLP (2 hidden SiLU layers, ~5k parameters by default). It is trained **through the
same integrator the filter uses** (RK4, zero-order-hold input, K-step rollouts from labelled states),
which is the fix of PITFALLS P1. It is then deployed inside the gated EKF (or UKF) of filters.py with
the nominal Q, R, P0, optionally with Q/R inflation tuned on ``val``.

Data: TRAIN split only (+ val for checkpoint and learning-rate selection): the equal-data rule. (The
research ude5k1 also used the 400 zero-input phase-1 episodes; on vdp/partial the standalone version
reaches the same accuracy without them.) Sparse labels (NaN in x,
cstr_exo) are handled by starting each window from the linearly interpolated state and masking the
loss to labelled samples (what the research grey-box did for the assays).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from ..estimator import BaseEstimator
from .filters import EKF, UKF


def _filled(es) -> np.ndarray:
    """x with never-labelled entries replaced by the split's placeholders (``x_fill``: e.g. the
    partial model's replay of the RC voltages on real battery data) when the split carries them;
    the loss stays masked to the labelled entries."""
    x = es.x.astype(np.float64)
    xf = getattr(es, "x_fill", None)
    if xf is not None:
        x = np.where(np.isfinite(x), x, xf.astype(np.float64))
    return _interp_nan(x)


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """(E, N1, nx): linear interpolation in time of NaN entries (constant beyond the ends)."""
    if np.isfinite(x).all():
        return x
    x = x.copy()
    t = np.arange(x.shape[1])
    for e in range(x.shape[0]):
        for j in range(x.shape[2]):
            v = x[e, :, j]
            ok = np.isfinite(v)
            if ok.any() and not ok.all():
                x[e, :, j] = np.interp(t, t[ok], v[ok])
    return x


class _Residual(nn.Module):
    def __init__(self, nx, width, x_mean, x_std, u_scale, out_scale, nu=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nx + nu, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, nx))
        f = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
        self.register_buffer("xm", f(x_mean))
        self.register_buffer("xs", f(x_std))
        self.register_buffer("us", f([u_scale]))
        self.register_buffer("os", f(out_scale))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)          # start at r = 0: the UDE starts from the model
        self.double()

    def forward(self, x, u):
        h = torch.cat([(x - self.xm) / self.xs, u / self.us], -1)
        return self.net(h) * self.os


class _NoPhysics:
    def __init__(self, m):
        self.nx, self.nu, self.sensors, self.ny = m.nx, m.nu, m.sensors, m.ny
        self._h = m.h

    def f(self, x, u):
        return torch.zeros_like(x)

    def h(self, x):
        return self._h(x)


class GreyBox:
    """Plant-like model f = f_model + r (functional; differentiable by the EKF's jacrev)."""

    def __init__(self, model, res: _Residual, hybrid=True):
        self.m = model if hybrid else _NoPhysics(model)
        self.r = res.eval()
        self.nx, self.nu, self.sensors, self.ny = model.nx, model.nu, model.sensors, model.ny

    def f(self, x, u):
        return self.m.f(x, u) + self.r(x, u)

    def h(self, x):
        return self.m.h(x)


def _rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class UDEFilter(BaseEstimator):
    returns_covariance = True

    def __init__(self, hybrid=True, kind="ekf", gate=True, width=None, budget=5000, K=1, steps=3000,
                 batch=1024, lrs=(3e-4, 1e-3, 3e-3, 1e-2), n_val_deployed=8, seed=0, threads=1,
                 q_inflate=1.0, r_inflate=1.0, name=None, log_every=0):
        """Defaults = the research ``ude5k1`` recipe: ~5k parameters, K = 1 step through RK4,
        3000 Adam steps, batch 1024, learning rate chosen from ``lrs`` by the DEPLOYED filter's
        settled NRMSE on ``n_val_deployed`` validation episodes (centre prior)."""
        self.hybrid, self.kind, self.gate = bool(hybrid), kind, bool(gate)
        self.cfg = dict(width=width, budget=budget, K=K, steps=steps, batch=batch, lrs=tuple(lrs), seed=seed,
                        n_val_deployed=n_val_deployed)
        self.threads, self.q_inflate, self.r_inflate, self.log_every = threads, q_inflate, r_inflate, log_every
        self.name = name or (("ude" if hybrid else "node") + f"_{kind}" + ("_gated" if gate else ""))

    def _width(self, nx):
        if self.cfg["width"]:
            return int(self.cfg["width"])
        p = lambda W: (nx + 1) * W + W + W * W + W + W * nx + nx
        return min(range(4, 512), key=lambda W: abs(p(W) - self.cfg["budget"]))

    def fit(self, train, val, context):
        c = self.cfg
        nth = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        t0 = time.time()
        self.context = context
        cands = []
        for lr in c["lrs"]:
            res, vroll = self._train_one(train, val, context, lr)
            filt = self._deploy(train, val, context, res)
            score = self._deployed_val(filt, val, context) if val is not None else vroll
            cands.append({"lr": lr, "val_rollout_mse_norm": vroll, "deployed_val": score})
            if self.log_every:
                print(f"  {self.name} lr {lr:g}: rollout {vroll:.3g} deployed val {score:.4f}")
            if score <= min(cc["deployed_val"] for cc in cands):
                self.res, self.filter = res, filt
        torch.set_num_threads(nth)
        best = min(cands, key=lambda cc: cc["deployed_val"])
        self.fit_info = {"chosen_lr": best["lr"], "candidates": cands, "fit_s": time.time() - t0,
                         "n_params": self.n_params(), "width": self._width(len(context["x_std"])), "K": c["K"],
                         "val_rollout_mse_norm": best["val_rollout_mse_norm"]}

    def _deploy(self, train, val, context, res):
        cls = EKF if self.kind == "ekf" else UKF
        f = cls(gate=self.gate, q_inflate=self.q_inflate, r_inflate=self.r_inflate)
        f.fit(train, val, dict(context, model=GreyBox(context["model"], res, self.hybrid)))
        return f

    def _deployed_val(self, filt, val, context):
        from ..metrics import episode_metrics
        sub = val.subset(np.arange(min(self.cfg["n_val_deployed"], len(val))))
        if getattr(sub, "prior", None) is not None:        # real-data tracks: per-episode priors
            X0, P0 = sub.prior.astype(float), np.tile(np.asarray(context["prior_std"], float), (len(sub), 1))
        else:
            X0, P0 = np.tile(context["prior_mean"], (len(sub), 1)), None
        with torch.no_grad():
            xh, _ = filt.run_batch(sub.u.astype(float), sub.y.astype(float), X0, P0)
        meta = context["meta"]
        xl = sub.x
        m = episode_metrics(np.where(np.isfinite(xl), xl, xh), xh, meta["x_std"], int(meta["settle_index"]), meta["sensors"])
        return float(np.mean(m["nrmse"]) + np.mean(m["diverged"]))

    def _train_one(self, train, val, context, lr):
        c = self.cfg
        torch.manual_seed(c["seed"])
        rng = np.random.RandomState(c["seed"])
        model, dt = context["model"], float(context["dt"])
        nx = len(context["x_std"])
        xs = np.asarray(context["x_std"], float)
        X = _filled(train)
        Xlab = train.x.astype(np.float64)
        U = train.u.astype(np.float64)
        E, N1 = U.shape[:2]
        U = U[..., None] if U.ndim == 2 else U     # (E, N+1, n_u)
        dX = (X[:, 1:] - X[:, :-1]) / dt          # output scale of r: typical size of the state derivative
        res = _Residual(nx, self._width(nx), context["x_mean"], xs, float(context["meta"]["input_box"]["u_scale"]),
                        np.nanstd(dX.reshape(-1, nx), 0) + 1e-9, nu=U.shape[-1])
        res.float()                                # train in float32, deploy in float64
        base = model if self.hybrid else _NoPhysics(model)
        f = lambda x, u: base.f(x, u) + res(x, u)
        opt = torch.optim.Adam(res.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["steps"])
        K, B = c["K"], c["batch"]
        Xt, Lt, Ut = (torch.as_tensor(a, dtype=torch.float32) for a in (X, Xlab, U))
        st = torch.as_tensor(xs, dtype=torch.float32)
        best, best_state = math.inf, None
        for it in range(c["steps"]):
            e = torch.as_tensor(rng.randint(0, E, B))
            k = torch.as_tensor(rng.randint(0, N1 - K - 1, B))
            x = Xt[e, k]
            loss, cnt = 0.0, 0
            for i in range(K):
                x = _rk4(f, x, Ut[e, k + i], dt)
                d = (x - Lt[e, k + i + 1]) / st
                m = torch.isfinite(d)
                loss = loss + torch.where(m, d, torch.zeros_like(d)).pow(2).sum()
                cnt = cnt + m.sum()
            loss = loss / cnt.clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(res.parameters(), 1.0)
            opt.step()
            sched.step()
            if val is not None and ((it + 1) % max(1, c["steps"] // 5) == 0 or it + 1 == c["steps"]):
                v = self._val_rollout(val, f, dt, st, max(K, 4))
                if v < best:
                    best, best_state = v, {kk: t.clone() for kk, t in res.state_dict().items()}
        if best_state is not None:
            res.load_state_dict(best_state)
        res.double()
        for p in res.parameters():
            p.requires_grad_(False)
        return res, float(best)

    @torch.no_grad()
    def _val_rollout(self, val, f, dt, st, K, n=2048):
        rng = np.random.RandomState(123)
        X = torch.as_tensor(_filled(val), dtype=torch.float32)
        L = torch.tensor(val.x, dtype=torch.float32)
        U = torch.tensor(val.u, dtype=torch.float32)
        E, N1 = U.shape[:2]
        U = U[..., None] if U.ndim == 2 else U
        e = torch.as_tensor(rng.randint(0, E, n))
        k = torch.as_tensor(rng.randint(0, N1 - K - 1, n))
        x = X[e, k]
        err, cnt = 0.0, 0
        for i in range(K):
            x = _rk4(f, x, U[e, k + i], dt)
            d = (x - L[e, k + i + 1]) / st
            m = torch.isfinite(d)
            err += float(torch.where(m, d, torch.zeros_like(d)).pow(2).sum())
            cnt += int(m.sum())
        return err / max(cnt, 1)

    def reset(self, x0_prior, P0_prior=None):
        return self.filter.reset(x0_prior, P0_prior)

    def step(self, u, y):
        return self.filter.step(u, y)

    def run_batch(self, U, Y, X0, P0=None):
        return self.filter.run_batch(U, Y, X0, P0)

    def n_params(self):
        return int(sum(p.numel() for p in self.res.parameters()))

    def state_bytes(self):
        return 4 * self.n_params() + self.filter.state_bytes()

    def flops_per_step(self):
        n = len(self.context["x_std"])
        W = self.res.net[0].out_features
        mlp = 2 * ((n + 1) * W + W * W + W * n)
        # EKF: the Jacobian costs ~n extra MLP passes (reverse mode) + 4 RK4 stages
        return int(self.filter.flops_per_step() + (4 + n) * mlp)


def _ude_transcendentals(self):
    n = len(self.context["x_std"])
    W = self.res.net[0].out_features
    return int((4 + n) * 2 * W)          # SiLU on 2 hidden layers per MLP pass (plant f transcendentals not counted)


UDEFilter.transcendentals_per_step = _ude_transcendentals
