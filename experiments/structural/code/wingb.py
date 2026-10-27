r"""Residual-capacity control (PREREG.md section 4): the grey-box whose residual also sees a
short multi-scale window of the past.

    r_theta(x_k, u_k, w_k),   w_k = [(xhat_{k-l} - x_mean) / x_std, u_{k-l} / u_scale]_{l in LAGS}
    LAGS = 1, 2, 4, ..., 2^m <= round(2 t_ref / dt)       (dyadic: resolves fast and slow memory)

Same MLP class as the grey-box (2 x 64 SiLU), same targets (x_{k+1} - Phi_model(x_k, u_k)) / dt,
same optimiser / schedule / early stopping (baselines_data.DEFAULTS). Trained on TRUE past states
(teacher forcing) with Gaussian noise of 0.05 x_std on the window states; deployed inside the
unchanged EKF / UKF equations with the filter's own past posterior means xhat in the window
(the window is held fixed over the step, like the input; the Jacobian is w.r.t. x_k only).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn
from torch.func import jacrev, vmap

from hyperkkl.observers import EKF, UKF, Observer, _rk4

WIN_NOISE = 0.05


def lags_for(t_ref: float, dt: float) -> list:
    span = max(1, int(round(2.0 * float(t_ref) / float(dt))))
    out, l = [], 1
    while l <= span:
        out.append(l); l *= 2
    return out


class WinMLP(nn.Module):
    def __init__(self, nx, nu, nw, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nx + nu + nw, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, nx))

    def forward(self, h):
        return self.net(h)


class WinGreyBox:
    """f(x, u, w) = f_model(x, u) + r_theta(x, u, w) (w already standardised)."""

    def __init__(self, model, net, xm, xs, us, tm, ts, lags):
        self.model, self.lags = model, list(lags)
        self.nx, self.nu, self.sensors, self.ny = model.nx, model.nu, model.sensors, model.ny
        sd = {k: v.detach().cpu().float() for k, v in net.state_dict().items()}
        lin = sorted({k.rsplit(".", 1)[0] for k in sd}, key=lambda k: int(k.split(".")[1]))
        self.layers = [(sd[f"{k}.weight"], sd[f"{k}.bias"]) for k in lin]
        f = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32)
        self.buf = tuple(f(a) for a in (xm, xs, us, tm, ts))
        self._dev = {}

    def _w(self, device):
        k = str(device)
        if k not in self._dev:
            self._dev[k] = ([(W.to(device), b.to(device)) for W, b in self.layers], tuple(t.to(device) for t in self.buf))
        return self._dev[k]

    def residual(self, x, u, w):
        layers, (xm, xs, us, tm, ts) = self._w(x.device)
        h = torch.cat([(x.float() - xm) / xs, u.float() / us, w.float()], -1)
        for i, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if i < len(layers) - 1:
                h = torch.nn.functional.silu(h)
        return (h * ts + tm).to(x.dtype)

    def f3(self, x, u, w):
        return self.model.f(x, u) + self.residual(x, u, w)

    def window(self, hx, hu):
        """hx (B, L+1, nx) past posterior means (index -1 = current), hu (B, L+1, nu)."""
        _, (xm, xs, us, _, _) = self._w(hx.device)
        idx = [hx.shape[1] - 1 - l for l in self.lags]
        wx = (hx[:, idx].float() - xm) / xs
        wu = hu[:, idx].float() / us
        return torch.cat([wx, wu], -1).reshape(hx.shape[0], -1)


class _Bound:
    def __init__(self, gb, w):
        self.gb, self.w = gb, w

    def f(self, x, u):
        w = self.w
        if w.shape[0] != x.shape[0]:
            w = w.repeat_interleave(x.shape[0] // w.shape[0], 0)
        return self.gb.f3(x, u, w)


class _WinMixin:
    def _win_reset(self, B, x0_hat, u0):
        L = max(self.gb.lags) + 1
        self.hx = x0_hat[:, None].expand(B, L, x0_hat.shape[1]).clone()
        self.hu = u0[:, None].expand(B, L, u0.shape[1]).clone()

    def _push(self, x, u):
        self.hx = torch.cat([self.hx[:, 1:], x[:, None]], 1)
        self.hu = torch.cat([self.hu[:, 1:], u[:, None]], 1)


class WinEKF(_WinMixin, EKF):
    def __init__(self, gb: WinGreyBox, dt, q, r, p0, name="gbw_ekf"):
        super().__init__(gb.model, dt, q, r, p0)
        self.gb, self.name = gb, name

    def reset(self, B, x0_hat, u0):
        self._base_reset(B, x0_hat)
        f1 = lambda xi, ui, wi: self.gb.f3(xi[None], ui[None], wi[None])[0]
        self.jac3 = vmap(jacrev(f1, argnums=0))
        self._win_reset(B, x0_hat, u0)

    def advance(self, y_k, u_k, y_k1, u_k1):
        if self._fresh:
            self._first(y_k)
            self._win_reset(self.x.shape[0], self.x, u_k)
        w = self.gb.window(self.hx, self.hu)
        with torch.enable_grad():
            F = self.jac3(self.x, u_k, w.to(self.x.dtype))
        Phi = torch.matrix_exp(F * self.dt)
        self.x = _rk4(_Bound(self.gb, w), self.x, u_k, u_k, u_k, self.dt)
        self.P = Phi @ self.P @ Phi.transpose(1, 2) + self.Q
        self._update(y_k1)
        self._push(self.x, u_k1)


class WinUKF(_WinMixin, UKF):
    def __init__(self, gb: WinGreyBox, dt, q, r, p0, name="gbw_ukf"):
        super().__init__(gb.model, dt, q, r, p0)
        self.gb, self.name = gb, name

    def reset(self, B, x0_hat, u0):
        super().reset(B, x0_hat, u0)
        self._win_reset(B, x0_hat, u0)

    def advance(self, y_k, u_k, y_k1, u_k1):
        if self._fresh:
            self._first(y_k)
            self._win_reset(self.x.shape[0], self.x, u_k)
        w = self.gb.window(self.hx, self.hu)
        m_save = self.m
        self.m = _Bound(self.gb, w)            # sigma points: w repeated per episode (2n+1 each)
        try:
            UKF.advance(self, y_k, u_k, y_k1, u_k1)
        finally:
            self.m = m_save
        self._push(self.x, u_k1)


def _windows(X, U, ep, k, lags, xm, xs, us, noise=0.0, g=None):
    """Window features for (episode, time) index pairs; X (E, N+1, nx) on device."""
    wx = torch.stack([X[ep, k - l] for l in lags], 1)                  # (n, L, nx)
    if noise > 0:
        wx = wx + noise * xs * torch.randn(wx.shape, generator=g, device=wx.device, dtype=wx.dtype)
    wu = torch.stack([U[ep, k - l] for l in lags], 1)
    return torch.cat([(wx - xm) / xs, wu / us], -1).reshape(len(ep), -1)


def fit_wingb(model, data, scales, dt, u_scale, t_ref, seed, device, opt, log=print) -> dict:
    t0 = time.time()
    lags = lags_for(t_ref, dt)
    L = max(lags)
    xm = torch.as_tensor(np.asarray(scales["x_mean"], float), dtype=torch.float32, device=device)
    xs = torch.as_tensor(np.asarray(scales["x_std"], float), dtype=torch.float32, device=device)
    us = torch.full((1,), float(u_scale), dtype=torch.float32, device=device)

    def index(sets, n_max, s):
        Xs, Us, eps, ks = [], [], [], []
        off = 0
        for es in sets:
            E, N1 = es.x.shape[0], es.x.shape[1]
            if N1 - 1 <= L:
                continue
            Xs.append(es.x); Us.append(es.u)
            e, k = torch.meshgrid(torch.arange(E), torch.arange(L, N1 - 1), indexing="ij")
            eps.append(e.reshape(-1) + off); ks.append(k.reshape(-1)); off += E
        X = torch.cat(Xs).to(device); U = torch.cat(Us).to(device)
        ep, k = torch.cat(eps), torch.cat(ks)
        if len(ep) > n_max:
            gg = torch.Generator().manual_seed(int(s))
            sel = torch.randperm(len(ep), generator=gg)[:n_max]
            ep, k = ep[sel], k[sel]
        ep, k = ep.to(device), k.to(device)
        x0 = X[ep, k].double(); u0 = U[ep, k].double(); x1 = X[ep, k + 1].double()
        T = torch.cat([(x1[i:i + 65536] - _rk4(model, x0[i:i + 65536], u0[i:i + 65536], u0[i:i + 65536],
                                               u0[i:i + 65536], float(dt))) / float(dt)
                       for i in range(0, len(ep), 65536)]).float()
        ok = torch.isfinite(T).all(1)
        return X, U, ep[ok], k[ok], T[ok]

    Xt, Ut, ept, kt, Tt = index([data["train"], data["p1"]], int(opt["gb_n_fit"]), seed + 1)
    Xv, Uv, epv, kv, Tv = index([data["val"]], int(opt["n_val"]), seed + 2)
    nx, nu, nw = Xt.shape[-1], Ut.shape[-1], len(lags) * (Xt.shape[-1] + Ut.shape[-1])
    tm, ts = Tt.mean(0), Tt.std(0).clamp(min=1e-6)
    torch.manual_seed(int(seed))
    net = WinMLP(nx, nu, nw, int(opt["gb_width"])).to(device)
    g = torch.Generator(device=device).manual_seed(int(seed))

    def feats(X, U, ep, k, noise=0.0):
        x, u = X[ep, k], U[ep, k]
        return torch.cat([(x - xm) / xs, u / us, _windows(X, U, ep, k, lags, xm, xs, us, noise, g)], -1)

    Fv = torch.cat([feats(Xv, Uv, epv[i:i + 65536], kv[i:i + 65536]) for i in range(0, len(epv), 65536)])

    def val_loss():
        with torch.no_grad():
            return float(sum((((net(Fv[i:i + 65536]) * ts + tm - Tv[i:i + 65536]) / ts) ** 2).sum()
                             for i in range(0, len(Fv), 65536)) / (len(Fv) * nx))

    opt_ = torch.optim.Adam(net.parameters(), lr=float(opt["gb_lr"]))
    steps, bs = int(opt["gb_steps"]), int(opt["gb_batch"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, steps)
    best, best_state, bad, hist = val_loss(), copy.deepcopy(net.state_dict()), 0, []
    step = 0
    for step in range(1, steps + 1):
        i = torch.randint(0, len(ept), (bs,), generator=g, device=device)
        F = feats(Xt, Ut, ept[i], kt[i], WIN_NOISE)
        loss = (((net(F) * ts + tm - Tt[i]) / ts) ** 2).mean()
        opt_.zero_grad(set_to_none=True); loss.backward(); opt_.step(); sched.step()
        if step % int(opt["gb_val_every"]) == 0 or step == steps:
            v = val_loss()
            hist.append((step, float(loss.detach()), v))
            if v < best - 1e-6 * abs(best):
                best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= int(opt["gb_patience"]):
                    break
    net.load_state_dict(best_state)
    with torch.no_grad():
        rv = torch.cat([net(Fv[i:i + 65536]) * ts + tm for i in range(0, len(Fv), 65536)])
        r2 = (1 - (rv - Tv).pow(2).mean(0) / Tv.var(0).clamp(min=1e-12)).cpu().tolist()
    gb = WinGreyBox(model, net.cpu(), xm.cpu().numpy(), xs.cpu().numpy(), us.cpu().numpy(),
                    tm.cpu().numpy(), ts.cpu().numpy(), lags)
    info = {"n_params": sum(p.numel() for p in net.parameters()), "lags": lags, "n_fit": len(ept),
            "steps": step, "val_loss": best, "val_r2": r2, "hist": hist, "wall_s": time.time() - t0,
            "win_noise": WIN_NOISE}
    log(f"  windowed grey-box residual: {info['n_params']} params, lags {lags}, {step} steps, "
        f"val loss {best:.4g}, val R^2 {[round(v, 3) for v in r2]} ({info['wall_s']:.1f}s)")
    return {"model": gb, "info": info}


class Climatology(Observer):
    """xhat = mean training state (true labels of data.train after washout): a constant."""
    name = "climatology"

    def __init__(self, mean):
        self.mean = torch.as_tensor(np.asarray(mean, float), dtype=torch.float32)

    def reset(self, B, x0_hat, u0):
        self.x = self.mean.to(device=x0_hat.device, dtype=x0_hat.dtype)[None].expand(B, -1).clone()

    def estimate(self):
        return self.x

    def advance(self, y_k, u_k, y_k1, u_k1):
        pass
