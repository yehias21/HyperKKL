"""Small self-contained baselines (no research code needed).

    PriorMean   x_hat = training mean of x, always (the 'knows nothing' floor; NRMSE ~ 1 on
                unmeasured states by construction of x_std)
    GRUObserver a GRU over normalised (u_k, y_k) -> x_k, trained by truncated BPTT on random
                windows of the TRAIN split (CPU-friendly; the quick-start baseline). Model-free.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ..estimator import BaseEstimator


class PriorMean(BaseEstimator):
    name = "prior_mean"

    def fit(self, train, val, context):
        self.context = context
        xf = train.x.reshape(-1, train.x.shape[-1])
        lab = np.isfinite(xf).any(0)                  # NaN = unlabelled (sparse labels); never-labelled states -> 0
        self.mu = np.zeros(xf.shape[1])
        self.mu[lab] = np.nanmean(xf[:, lab], 0)

    def reset(self, x0_prior, P0_prior=None):
        pass

    def step(self, u, y):
        return self.mu.copy()

    def run_batch(self, U, Y, X0, P0=None):
        return np.broadcast_to(self.mu, (U.shape[0], U.shape[1], len(self.mu))).copy()

    def n_params(self):
        return len(self.mu)

    def flops_per_step(self):
        return 0


class _GRUNet(nn.Module):
    def __init__(self, ny, nx, hidden, layers, nu=1):
        super().__init__()
        self.rnn = nn.GRU(nu + ny, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, nx)

    def forward(self, inp, h=None):
        o, h = self.rnn(inp, h)
        return self.head(o), h


class GRUObserver(BaseEstimator):
    """Model-free GRU observer. Equal-data rule: uses the train split only (val for early
    stopping); never val_ood or test data."""

    def __init__(self, hidden=32, layers=1, steps=1500, window=256, warmup=32, batch=32, lr=3e-3,
                 seed=0, threads=1, name="gru", log_every=0):
        self.cfg = dict(hidden=hidden, layers=layers, steps=steps, window=window, warmup=warmup,
                        batch=batch, lr=lr, seed=seed)
        self.threads, self.name, self.log_every = threads, name, log_every

    def _inp(self, U, Y):
        U = U[..., None] if U.ndim == Y.ndim - 1 else U          # (.., n_u): scalar or vector input
        return np.concatenate([U / self.us, (Y - self.ym) / self.ys], -1).astype(np.float32)

    def fit(self, train, val, context):
        c = self.cfg
        torch.manual_seed(c["seed"])
        torch.set_num_threads(self.threads)
        rng = np.random.RandomState(c["seed"])
        self.context = context
        self.us = float(context["meta"]["input_box"]["u_scale"])
        self.ym, self.ys = train.y.reshape(-1, train.y.shape[-1]).mean(0), np.asarray(context["y_std"])
        self.xm, self.xs = np.asarray(context["x_mean"]), np.asarray(context["x_std"])
        nx, ny = train.x.shape[-1], train.y.shape[-1]
        self.nu = 1 if train.u.ndim == 2 else train.u.shape[-1]
        self.net = _GRUNet(ny, nx, c["hidden"], c["layers"], self.nu)
        opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["steps"])
        inp = self._inp(train.u, train.y)
        tgt = ((train.x - self.xm) / self.xs).astype(np.float32)
        W, E, N1 = c["window"], len(train), train.u.shape[1]
        best, best_state, t0 = np.inf, None, time.time()
        for it in range(c["steps"]):
            e = rng.randint(0, E, c["batch"])
            k = rng.randint(0, N1 - W, c["batch"])
            idx = k[:, None] + np.arange(W)[None]
            xi = torch.from_numpy(inp[e[:, None], idx])
            yt = torch.from_numpy(tgt[e[:, None], idx])
            out, _ = self.net(xi)
            d = out[:, c["warmup"]:] - yt[:, c["warmup"]:]
            m = torch.isfinite(d)                         # sparse labels (cstr_exo): NaN = no label
            loss = torch.where(m, d, torch.zeros_like(d)).pow(2).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            if val is not None and ((it + 1) % max(1, c["steps"] // 5) == 0 or it + 1 == c["steps"]):
                v = self._val_loss(val)
                if v < best:
                    best, best_state = v, {k_: t.clone() for k_, t in self.net.state_dict().items()}
            if self.log_every and (it + 1) % self.log_every == 0:
                print(f"  {self.name} step {it + 1} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.fit_info = {"val_mse_norm": float(best), "fit_s": time.time() - t0}

    @torch.no_grad()
    def _val_loss(self, val, n=16):
        sub = val.subset(np.arange(min(n, len(val))))
        xh = self.run_batch(sub.u, sub.y, None)
        s = sub.u.shape[1] // 2
        return float(np.nanmean(((xh[:, s:] - sub.x[:, s:]) / self.xs) ** 2))

    @torch.no_grad()
    def reset(self, x0_prior, P0_prior=None):
        self.h = None

    @torch.no_grad()
    def step(self, u, y):
        xi = torch.from_numpy(self._inp(np.asarray(u, float).reshape(1, 1, -1), np.asarray(y, float)[None, None]))
        o, self.h = self.net(xi, self.h)
        return self.xm + self.xs * o[0, 0].double().numpy()

    @torch.no_grad()
    def run_batch(self, U, Y, X0, P0=None):
        o, _ = self.net(torch.from_numpy(self._inp(U, Y)))
        return self.xm + self.xs * o.double().numpy()

    def n_params(self):
        return int(sum(p.numel() for p in self.net.parameters()))

    def state_bytes(self):
        return 4 * (self.n_params() + self.cfg["hidden"] * self.cfg["layers"])

    def flops_per_step(self):
        H, L = self.cfg["hidden"], self.cfg["layers"]
        ni = getattr(self, "nu", 1) + len(self.context["sensors"])
        nx = len(self.context["x_std"])
        f = 0
        for l in range(L):
            i = ni if l == 0 else H
            f += 2 * 3 * (i * H + H * H) + 10 * H
        return int(f + 2 * H * nx)


def _gru_transcendentals(self):
    return int(3 * self.cfg["hidden"] * self.cfg["layers"])     # 2 sigmoid + 1 tanh per hidden unit


GRUObserver.transcendentals_per_step = _gru_transcendentals
