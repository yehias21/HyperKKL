"""Data-driven baselines trained on EXACTLY the learned observers' training data
(theory/reviews/editors/reviewer_c.md, items E1 and E2).

Both use the state-labelled trajectories of the REAL plant that the learned KKL observers
use: ``data['train']`` (forced episodes) + ``data['p1']`` (zero-input episodes); any
selection uses ``data['val']`` only. The input is the zero-order-hold sample u_k on
[t_k, t_{k+1}), as everywhere (data.integrate).

E1  ``koopman_kf`` / ``koopman_kf_bilinear``: EDMD with inputs (Korda & Mezic 2018) and a
    Kalman filter on the lifted state.
      dictionary  psi(x) = [x; phi_1(x) .. phi_C(x)],  phi_c = r^2 log r,  r = ||(x - c) / x_std||
                  (thin-plate RBFs; C = 200 if nx <= 3 else 400; centres by seeded Lloyd
                  k-means on subsampled training states). x is in the dictionary, so the
                  state readout and y = h(x) = x[sensors] are selection matrices.
      model       linear     psi+ = K psi + G u + b
                  bilinear   psi+ = K psi + G u + sum_i u_i N_i psi + b
                  ridge least squares on the training transitions (column-standardised
                  regressors; ridge from {1e-6, 1e-4, 1e-2} by the validation one-step error
                  of the standardised lifted state). The intercept b stands in for the
                  constant the thin-plate spline basis needs (the lifted state has no
                  constant coordinate that the filter would have to keep at 1).
      filter      Kalman filter on psi. Q = diag of the regression-residual covariance
                  (+ a small floor), R = the nominal sensor variance (the EKF's R, no
                  oracle), psi_0 = psi(x0_hat) (the prior every observer gets), P_0 = the
                  covariance of psi(x) for x uniform on the IC box (the lifted image of the
                  EKF's P_0 = box width^2 / 12; ``p0: residual`` gives the residual-scale
                  alternative), + Q. xhat = x-part of psi_hat. The linear filter's
                  covariance does not depend on the data, so its gain sequence is computed
                  once (float64) and shared by every episode; the bilinear filter has a
                  per-episode covariance through A_k = K + sum_i u_{k,i} N_i.
E2  ``greybox_ekf`` / ``greybox_ukf`` (/ ``greybox_ekf_oracle``): the modeller's model
    (data.model_plant: f_known under system.knowledge = partial, nominal parameters)
    plus a residual MLP r_theta(x, u) ~ f_real - f_model fitted to
        (x_{k+1} - Phi_model(x_k, u_k)) / dt,   Phi_model = one RK4 / ZOH step at the data dt
    (x is the true state, so no finite differences of noisy signals). 2 x 64 SiLU, Adam,
    early stopping on validation. f_model + r_theta goes into the unchanged EKF / UKF (the
    EKF's jacrev differentiates through the MLP) with the nominal EKF's Q, R, P_0;
    greybox_ekf_oracle is the condition-oracle EKF (evaluate.BlockOracle, the job's
    ekf_oracle tuning) on the grey-box model.

Fits are made once per job (``fit_all``) and held in memory only (no disk cache).
Options: config key ``data_baselines`` (absent from base.yaml on purpose; defaults in
``DEFAULTS``).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from .observers import EKF, UKF, Observer, _rk4

KOOPMAN = {"koopman_kf": "linear", "koopman_kf_bilinear": "bilinear"}
GREYBOX = ("greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")
DATA_BASELINES = tuple(KOOPMAN) + GREYBOX

DEFAULTS = {
    "n_centres": None,           # None: 200 if nx <= 3 else 400
    "kmeans_points": 20000,      # subsampled training states for k-means
    "kmeans_iters": 30,
    "n_fit": 1_000_000,          # training transitions used by the least squares (all if fewer)
    "n_val": 100_000,            # validation transitions
    "ridges": [1e-6, 1e-4, 1e-2],
    "q_floor": 1e-8,             # Q floor, relative to each lifted coordinate's variance
    "p0": "prior",               # prior | residual
    "p0_residual_factor": 1e6,
    "gb_width": 64,
    "gb_steps": 3000,
    "gb_batch": 4096,
    "gb_lr": 1e-3,
    "gb_val_every": 100,
    "gb_patience": 8,            # validation checks without improvement before stopping
    "gb_n_fit": 1_000_000,
}


def options(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("data_baselines") or {})}


def wanted(cfg: dict) -> list:
    return [b for b in cfg.get("baselines", []) if b in DATA_BASELINES]


# ---------------------------------------------------------------------------
# Training transitions
# ---------------------------------------------------------------------------

def _sets(data: dict, split: str) -> list:
    return [data["train"], data["p1"]] if split == "train" else [data["val"]]


@torch.no_grad()
def transitions(sets: list, n_max: int, seed: int, device) -> tuple:
    """(x_k, u_k, x_{k+1}) of every episode and step of ``sets`` (EpisodeSets or (x, u)
    tensor pairs), uniformly subsampled without replacement to at most n_max; float64."""
    xs, us, x1s = [], [], []
    for s in sets:
        x, u = (s.x, s.u) if hasattr(s, "x") else s
        xs.append(x[:, :-1].reshape(-1, x.shape[-1]))
        us.append(u[:, :-1].reshape(-1, u.shape[-1]))
        x1s.append(x[:, 1:].reshape(-1, x.shape[-1]))
    X, U, X1 = torch.cat(xs), torch.cat(us), torch.cat(x1s)
    ok = torch.isfinite(X).all(1) & torch.isfinite(X1).all(1) & torch.isfinite(U).all(1)
    idx = torch.nonzero(ok)[:, 0]
    if len(idx) > n_max:
        g = torch.Generator().manual_seed(int(seed))
        idx = idx[torch.randperm(len(idx), generator=g)[:n_max]]
    f = lambda a: a[idx].to(device=device, dtype=torch.float64)
    return f(X), f(U), f(X1)


# ---------------------------------------------------------------------------
# E1: EDMD with inputs
# ---------------------------------------------------------------------------

def kmeans(X: torch.Tensor, k: int, iters: int = 30, seed: int = 0) -> torch.Tensor:
    """Plain Lloyd k-means from k distinct random points (seeded); an emptied cluster is
    re-seeded at the point farthest from its centre."""
    g = torch.Generator().manual_seed(int(seed))
    C = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(int(iters)):
        d = torch.cdist(X, C)
        a = d.argmin(1)
        cnt = torch.bincount(a, minlength=k).to(X.dtype)
        S = torch.zeros_like(C).index_add_(0, a, X)
        newC = S / cnt.clamp(min=1)[:, None]
        empty = cnt == 0
        if empty.any():
            far = d.gather(1, a[:, None])[:, 0].argsort(descending=True)[:int(empty.sum())]
            newC[empty] = X[far]
        if torch.allclose(newC, C):
            C = newC
            break
        C = newC
    return C


class Dictionary:
    """psi(x) = [x; r^2 log r to every centre], r in x_std-normalised coordinates."""

    def __init__(self, centres, mean, std):
        self.centres = torch.as_tensor(centres, dtype=torch.float64)       # normalised
        self.mean = torch.as_tensor(mean, dtype=torch.float64)
        self.std = torch.as_tensor(std, dtype=torch.float64)
        self.nx = self.mean.numel()
        self._dev = {}

    @property
    def n(self) -> int:
        return self.nx + len(self.centres)

    def _on(self, device):
        k = str(device)
        if k not in self._dev:
            self._dev[k] = tuple(t.to(device) for t in (self.centres, self.mean, self.std))
        return self._dev[k]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, m, s = self._on(x.device)
        x = x.to(torch.float64)
        if len(C) == 0:
            return x
        r2 = torch.cdist((x - m) / s, C).pow(2)
        return torch.cat([x, 0.5 * r2 * torch.log(r2.clamp(min=1e-300))], 1)


def _regressors(P: torch.Tensor, U: torch.Tensor, kind: str) -> torch.Tensor:
    one = torch.ones(len(P), 1, dtype=P.dtype, device=P.device)
    parts = [P, U]
    if kind == "bilinear":
        parts += [U[:, i:i + 1] * P for i in range(U.shape[1])]
    return torch.cat(parts + [one], 1)


def _gram(dic, X, U, X1, kind, chunk=20000):
    """Z'Z, Z'Psi+ (float64, chunked) and the moments of Psi+."""
    G = ZY = None
    s1 = s2 = 0.0
    for i in range(0, len(X), chunk):
        P, P1 = dic(X[i:i + chunk]), dic(X1[i:i + chunk])
        Z = _regressors(P, U[i:i + chunk], kind)
        G = Z.T @ Z if G is None else G + Z.T @ Z
        ZY = Z.T @ P1 if ZY is None else ZY + Z.T @ P1
        s1 = s1 + P1.sum(0)
        s2 = s2 + (P1 * P1).sum(0)
    M = len(X)
    mu = s1 / M
    return G / M, ZY / M, mu, (s2 / M - mu * mu).clamp(min=0)


def _solve(G, ZY, lam):
    d = torch.diagonal(G).clamp(min=1e-300).sqrt()                # column RMS
    Gs = G / (d[:, None] * d[None, :])
    I = torch.eye(len(G), dtype=G.dtype, device=G.device)
    W = torch.linalg.solve(Gs + lam * I, ZY / d[:, None])
    return W / d[:, None]                                          # (nZ, n)


def _split(W, n, nu, kind):
    K, Gu = W[:n].T, W[n:n + nu].T
    N = torch.stack([W[n + nu + i * n:n + nu + (i + 1) * n].T for i in range(nu)]) if kind == "bilinear" \
        else torch.zeros(0, n, n, dtype=W.dtype, device=W.device)
    return K.contiguous(), Gu.contiguous(), N.contiguous(), W[-1].contiguous()


@torch.no_grad()
def _residual_stats(dic, model, X, U, X1, chunk=20000):
    """Per-coordinate mean square of psi+ - model(psi, u) and its standardised MSE."""
    s, M = 0.0, len(X)
    for i in range(0, M, chunk):
        e = dic(X1[i:i + chunk]) - model.step(dic(X[i:i + chunk]), U[i:i + chunk])
        s = s + (e * e).sum(0)
    return s / M


class LiftedModel:
    """psi+ = K psi + G u + sum_i u_i N_i psi + b (float64)."""

    def __init__(self, K, G, N, b, kind):
        self.K, self.G, self.N, self.b, self.kind = K, G, N, b, kind
        self._dev = {}

    def on(self, device) -> "LiftedModel":
        k = str(device)
        if k not in self._dev:
            t = [a.to(device) for a in (self.K, self.G, self.N, self.b)]
            m = LiftedModel(*t, self.kind)
            m._dev = {k: m}
            self._dev[k] = m
        return self._dev[k]

    def A(self, u: torch.Tensor) -> torch.Tensor:
        """(B, n, n) transition matrices K + sum_i u_i N_i."""
        return self.K[None] + torch.einsum("bi,inm->bnm", u, self.N)

    def step(self, psi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        out = psi @ self.K.T + u @ self.G.T + self.b
        if self.kind == "bilinear":
            out = out + torch.einsum("bi,inm,bm->bn", u, self.N, psi)
        return out


@torch.no_grad()
def fit_dictionary(data: dict, scales: dict, nx: int, seed: int, device, opt: dict, log=print) -> Dictionary:
    C = opt["n_centres"]
    C = (200 if nx <= 3 else 400) if C is None else int(C)
    mean = torch.as_tensor(scales.get("x_mean", [0.0] * nx), dtype=torch.float64)
    std = torch.as_tensor(scales["x_std"], dtype=torch.float64)
    if C == 0:
        return Dictionary(torch.zeros(0, nx), mean, std)
    X, _, _ = transitions(_sets(data, "train"), int(opt["kmeans_points"]), seed + 17, device)
    Xn = (X - mean.to(device)) / std.to(device)
    t0 = time.time()
    cen = kmeans(Xn, C, int(opt["kmeans_iters"]), seed)
    log(f"  koopman dictionary: {nx} + {C} thin-plate RBFs, k-means on {len(X)} states ({time.time() - t0:.1f}s)")
    return Dictionary(cen.cpu(), mean, std)


@torch.no_grad()
def fit_edmd(dic: Dictionary, tr: tuple, va: tuple, kind: str, opt: dict, log=print) -> dict:
    """Ridge EDMD(c) on training transitions tr = (X, U, X1); ridge chosen on va."""
    t0 = time.time()
    X, U, X1 = tr
    n, nu = dic.n, U.shape[1]
    G, ZY, mu, var = _gram(dic, X, U, X1, kind)
    sd = var.sqrt().clamp(min=1e-12)
    scores, best = [], None
    for lam in opt["ridges"]:
        m = LiftedModel(*_split(_solve(G, ZY, float(lam)), n, nu, kind), kind)
        e = float((_residual_stats(dic, m, *va) / sd ** 2).mean())
        scores.append({"ridge": float(lam), "val_onestep": e})
        if best is None or e < best[0]:
            best = (e, float(lam), m)
    _, lam, model = best
    res = _residual_stats(dic, model, X, U, X1)
    q = res + float(opt["q_floor"]) * var + 1e-14
    rho = float(torch.linalg.eigvals(model.K).abs().max())
    info = {"kind": kind, "n_lift": n, "n_fit": len(X), "ridge": lam, "ridge_scores": scores,
            "spectral_radius_K": rho, "n_params": int(n * n + n * nu + n + (nu * n * n if kind == "bilinear" else 0)),
            "train_onestep": float((res / sd ** 2).mean()), "wall_s": time.time() - t0}
    log(f"  koopman {kind}: lift {n}, {len(X)} transitions, ridge {lam:g} "
        f"(val one-step {', '.join('%g:%.3g' % (s_['ridge'], s_['val_onestep']) for s_ in scores)}), "
        f"rho(K) {rho:.4f} ({info['wall_s']:.1f}s)")
    return {"model": model, "Q": q, "info": info}


@torch.no_grad()
def lifted_prior(dic: Dictionary, box, n: int = 8192, seed: int = 0, device="cpu") -> torch.Tensor:
    """Covariance of psi(x), x uniform on the IC box (the EKF's prior, lifted)."""
    box = np.asarray(box, float)
    g = torch.Generator().manual_seed(int(seed))
    lo = torch.as_tensor(box[:, 0], dtype=torch.float64)
    hi = torch.as_tensor(box[:, 1], dtype=torch.float64)
    X = (lo + (hi - lo) * torch.rand(n, len(box), generator=g, dtype=torch.float64)).to(device)
    P = dic(X)
    P = P - P.mean(0)
    return (P.T @ P / (n - 1)).cpu()


class KoopmanKF(Observer):
    """Kalman filter on the EDMD lifted state; follows the _Filter contract: y_0 is
    assimilated before the first prediction, then predict with u_k (ZOH), update with
    y_{k+1}. Float64 inside, the estimate is returned in x0_hat's dtype."""
    uses_model = False

    def __init__(self, name, dic: Dictionary, model: LiftedModel, Q, r_std, sensors, P0):
        self.name, self.dic, self.model = name, dic, model
        self.Q = Q.detach().cpu().double() if torch.is_tensor(Q) else torch.as_tensor(np.array(Q, dtype=float))
        self.r2 = torch.as_tensor(np.asarray(r_std, float) ** 2, dtype=torch.float64).clamp(min=1e-12)
        self.sensors = list(sensors)
        self.P0 = torch.as_tensor(P0, dtype=torch.float64)
        self.shared = model.kind == "linear"
        self._gain_cache = {}

    def _dev(self, device):
        m = self.model.on(device)
        return m, torch.diag(self.Q.to(device)), torch.diag(self.r2.to(device)), self.P0.to(device)

    def reset(self, B, x0_hat, u0):
        dev = x0_hat.device
        self._m, self._Q, self._R, P0 = self._dev(dev)
        self._dtype = x0_hat.dtype
        self.psi = self.dic(x0_hat.to(torch.float64))
        self.k, self._fresh = 0, True
        if self.shared:
            key = str(dev)
            if key not in self._gain_cache:
                self._gain_cache[key] = {"gains": [], "P": P0.clone(), "fresh": True}
            self._gc = self._gain_cache[key]
        else:
            self.P = P0.expand(B, -1, -1).clone()

    def estimate(self):
        return torch.nan_to_num(self.psi[:, :self.dic.nx], nan=1e6, posinf=1e6, neginf=-1e6).to(self._dtype)

    def _kgain(self, P):
        """Gain and posterior covariance (Joseph form; H is a selection)."""
        s = self.sensors
        PHt = P[..., :, s]                                   # (.., n, ny)
        S = P[..., s, :][..., :, s] + self._R
        Kg = PHt @ torch.linalg.inv(S)
        HP = P[..., s, :]
        Pn = P - Kg @ HP - HP.transpose(-1, -2) @ Kg.transpose(-1, -2) + Kg @ S @ Kg.transpose(-1, -2)
        return Kg, 0.5 * (Pn + Pn.transpose(-1, -2))

    def _gain(self, j):
        """Shared gain of the j-th measurement update (j = 0: y_0 on the prior)."""
        gc = self._gc
        while len(gc["gains"]) <= j:
            P = gc["P"]
            if not gc["fresh"]:
                P = self._m.K @ P @ self._m.K.T + self._Q
            gc["fresh"] = False
            Kg, gc["P"] = self._kgain(P)
            gc["gains"].append(Kg)
        return gc["gains"][j]

    def _update(self, y, j):
        y = y.to(torch.float64)
        e = y - self.psi[:, self.sensors]
        if self.shared:
            Kg = self._gain(j)
            self.psi = self.psi + e @ Kg.T
        else:
            Kg, self.P = self._kgain(self.P)
            self.psi = self.psi + (Kg @ e[:, :, None])[:, :, 0]
        self.psi = torch.nan_to_num(self.psi, nan=1e6, posinf=1e6, neginf=-1e6)

    def advance(self, y_k, u_k, y_k1, u_k1):
        if self._fresh:
            self._update(y_k, 0)
            self._fresh = False
        u = u_k.to(torch.float64)
        if not self.shared:
            A = self._m.A(u)
            self.P = A @ self.P @ A.transpose(1, 2) + self._Q
        self.psi = self._m.step(self.psi, u)
        self.k += 1
        self._update(y_k1, self.k)


# ---------------------------------------------------------------------------
# E2: grey-box residual model
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """r_theta(x, u): standardised inputs and outputs, 2 hidden layers of SiLU."""

    def __init__(self, nx, nu, width, x_mean, x_std, u_scale, t_mean, t_std):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nx + nu, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, nx))
        f = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32)
        self.register_buffer("xm", f(x_mean))
        self.register_buffer("xs", f(x_std))
        self.register_buffer("us", f(np.full(nu, u_scale)))
        self.register_buffer("tm", f(t_mean))
        self.register_buffer("ts", f(t_std))

    def forward(self, x, u):
        h = torch.cat([(x - self.xm) / self.xs, u / self.us], -1)
        return self.net(h) * self.ts + self.tm


class GreyBoxModel:
    """Plant-like f(x, u) = f_model(x, u) + r_theta(x, u), h = the model's h. The MLP is
    evaluated functionally from float32 weight tensors (per-device copies), so the EKF's
    vmap(jacrev(f)) differentiates through it; f casts to and from the caller's dtype."""

    def __init__(self, model, mlp: ResidualMLP):
        self.model, self.mlp = model, mlp.eval()
        self.nx, self.nu = model.nx, model.nu
        self.sensors, self.ny = model.sensors, model.ny
        self.name = getattr(model, "name", "plant") + "+residual"
        sd = {k: v.detach().cpu().float() for k, v in mlp.state_dict().items()}
        lin = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("net.")}, key=lambda k: int(k.split(".")[1]))
        self._w = {"cpu": ([(sd[f"{k}.weight"], sd[f"{k}.bias"]) for k in lin],
                           tuple(sd[k] for k in ("xm", "xs", "us", "tm", "ts")))}

    def _weights(self, device):
        k = str(torch.device(device))
        if k not in self._w:
            layers, buf = self._w["cpu"]
            self._w[k] = ([(W.to(device), b.to(device)) for W, b in layers], tuple(t.to(device) for t in buf))
        return self._w[k]

    def residual(self, x, u):
        layers, (xm, xs, us, tm, ts) = self._weights(x.device)
        h = torch.cat([(x.to(torch.float32) - xm) / xs, u.to(torch.float32) / us], -1)
        for i, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if i < len(layers) - 1:
                h = torch.nn.functional.silu(h)
        return (h * ts + tm).to(x.dtype)

    def f(self, x, u):
        return self.model.f(x, u) + self.residual(x, u)

    def h(self, x):
        return self.model.h(x)

    def params(self):
        return self.model.params()


@torch.no_grad()
def residual_targets(model, X, U, X1, dt) -> torch.Tensor:
    return (X1 - _rk4(model, X, U, U, U, float(dt))) / float(dt)


def fit_greybox(model, tr: tuple, va: tuple, dt: float, scales: dict, u_scale: float, seed: int,
                device, opt: dict, log=print) -> dict:
    """r_theta ~ (x_{k+1} - Phi_model(x_k, u_k)) / dt, Adam, early stopping on va."""
    t0 = time.time()
    X, U, X1 = tr
    T = residual_targets(model, X, U, X1, dt)
    ok = torch.isfinite(T).all(1)
    X, U, T = X[ok], U[ok], T[ok]
    Xv, Uv, X1v = va
    Tv = residual_targets(model, Xv, Uv, X1v, dt)
    okv = torch.isfinite(Tv).all(1)
    Xv, Uv, Tv = Xv[okv].float(), Uv[okv].float(), Tv[okv].float()
    nx, nu = X.shape[1], U.shape[1]
    tm, ts = T.mean(0), T.std(0).clamp(min=1e-6)
    xm = scales.get("x_mean") or X.mean(0).tolist()
    torch.manual_seed(int(seed))
    net = ResidualMLP(nx, nu, int(opt["gb_width"]), xm, scales["x_std"], u_scale,
                      tm.cpu().numpy(), ts.cpu().numpy()).to(device)
    Xf, Uf, Tf = X.float(), U.float(), T.float()
    tsf = ts.float().to(device)
    opt_ = torch.optim.Adam(net.parameters(), lr=float(opt["gb_lr"]))
    steps, bs = int(opt["gb_steps"]), int(opt["gb_batch"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, steps)
    g = torch.Generator(device=Xf.device).manual_seed(int(seed))

    def val_loss():
        with torch.no_grad():
            return float(sum((((net(Xv[i:i + 65536], Uv[i:i + 65536]) - Tv[i:i + 65536]) / tsf) ** 2).sum()
                             for i in range(0, len(Xv), 65536)) / (len(Xv) * nx))

    best, best_state, bad, hist = val_loss(), copy.deepcopy(net.state_dict()), 0, []
    step = 0
    for step in range(1, steps + 1):
        i = torch.randint(0, len(Xf), (bs,), generator=g, device=Xf.device)
        loss = (((net(Xf[i], Uf[i]) - Tf[i]) / tsf) ** 2).mean()
        opt_.zero_grad(set_to_none=True)
        loss.backward()
        opt_.step()
        sched.step()
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
    net = net.cpu().eval()
    # explained fraction of the target variance on validation (1 = residual fully captured)
    with torch.no_grad():
        rv = net.to(Xv.device)(Xv, Uv)
        var_t = Tv.var(0).clamp(min=1e-12)
        r2 = (1 - (rv - Tv).pow(2).mean(0) / var_t).cpu().tolist()
    net = net.cpu()
    info = {"n_params": sum(p.numel() for p in net.parameters()), "n_fit": len(X), "steps": step,
            "val_loss": best, "val_r2": r2, "target_rms": T.pow(2).mean(0).sqrt().cpu().tolist(),
            "hist": hist, "wall_s": time.time() - t0}
    log(f"  greybox residual: {info['n_params']} params, {len(X)} transitions, {step} steps, "
        f"val loss {best:.4g}, val R^2 per state {[round(v, 3) for v in r2]} ({info['wall_s']:.1f}s)")
    return {"model": GreyBoxModel(model, net), "info": info}


# ---------------------------------------------------------------------------
# One fit per job
# ---------------------------------------------------------------------------

def fit_all(cfg: dict, data: dict, scales: dict, des, device, seed: int, log=print) -> dict:
    """Fit what cfg['baselines'] needs, once: {'koopman': {kind: fit}, 'dictionary',
    'P0', 'greybox', 'info': {baseline: fit info}}."""
    from . import data as Dm
    names = wanted(cfg)
    out = {"info": {}}
    if not names:
        return out
    opt = options(cfg)
    sc = cfg["system"]
    model = Dm.model_plant(sc)
    nx = model.nx
    xs = torch.zeros(4, nx, dtype=torch.float64).uniform_(-1, 1)
    if not torch.equal(model.h(xs), xs[:, list(model.sensors)]):
        raise ValueError(f"{sc['name']}: data-driven baselines assume y = x[sensors]")
    kinds = sorted({KOOPMAN[b] for b in names if b in KOOPMAN})
    gb = any(b in GREYBOX for b in names)
    n_fit = max(int(opt["n_fit"]), int(opt["gb_n_fit"]) if gb else 0)
    t0 = time.time()
    tr = transitions(_sets(data, "train"), n_fit, seed + 1, device)
    va = transitions(_sets(data, "val"), int(opt["n_val"]), seed + 2, device)
    t_data = time.time() - t0
    log(f"  data baselines {names}: {len(tr[0])} training / {len(va[0])} validation transitions "
        f"({t_data:.1f}s)")
    if kinds:
        t0 = time.time()
        k_tr = tuple(a[:int(opt["n_fit"])] for a in tr)
        dic = fit_dictionary(data, scales, nx, seed, device, opt, log)
        t_dic = time.time() - t0
        out["dictionary"] = dic
        out["koopman"] = {k: fit_edmd(dic, k_tr, va, k, opt, log) for k in kinds}
        if opt["p0"] == "prior":
            P0 = lifted_prior(dic, sc["ic_box"], seed=seed, device=device)
        elif opt["p0"] == "residual":
            P0 = None
        else:
            raise ValueError(f"data_baselines.p0 must be prior | residual, not {opt['p0']!r}")
        for k, f in out["koopman"].items():
            Qd = f["Q"].cpu()
            f["P0"] = (P0 if P0 is not None else torch.diag(float(opt["p0_residual_factor"]) * Qd)) + torch.diag(Qd)
            f["Q"] = Qd
            f["info"].update(dictionary_s=t_dic, transitions_s=t_data, p0=opt["p0"])
        for b in names:
            if b in KOOPMAN:
                out["info"][b] = dict(out["koopman"][KOOPMAN[b]]["info"])
    if gb:
        g = fit_greybox(model, tuple(a[:int(opt["gb_n_fit"])] for a in tr), va, des.dt, scales,
                        float(sc.get("u_scale", 1.0)), seed, device, opt, log)
        g["info"]["transitions_s"] = t_data
        out["greybox"] = g
        for b in names:
            if b in GREYBOX:
                out["info"][b] = dict(g["info"])
    return out


def make_observer(name, cfg, des, scales, q, r, p0, fits: dict, oracle_maker=None, block_oracle=None):
    """The observer for one data-driven baseline name (evaluate.build_observers)."""
    if not fits or (name in KOOPMAN and KOOPMAN[name] not in fits.get("koopman", {})) \
            or (name in GREYBOX and "greybox" not in fits):
        raise KeyError(f"baseline {name!r} needs a fit: pass baselines_data.fit_all(...) to "
                       f"build_observers (job.run_job and reeval do)")
    from . import data as Dm
    sensors = list(Dm.model_plant(cfg["system"]).sensors)
    if name in KOOPMAN:
        f = fits["koopman"][KOOPMAN[name]]
        return KoopmanKF(name, fits["dictionary"], f["model"], f["Q"], r, sensors, f["P0"])
    gm = fits["greybox"]["model"]
    if name == "greybox_ekf":
        ob = EKF(gm, des.dt, q, r, p0)
    elif name == "greybox_ukf":
        ob = UKF(gm, des.dt, q, r, p0)
    else:
        make = oracle_maker(cfg, des, scales, gm, p0, "ekf_oracle")

        def mk(sensor, shift):
            f_ = make(sensor, shift)
            f_.name = name
            return f_
        return block_oracle(name, mk, cfg["data"]["sensor"])
    ob.name = name
    return ob
