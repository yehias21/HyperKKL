"""Episode generation for learned KKL observers.

Every episode is a complete forced trajectory on a fixed grid:

    x       (E, N+1, nx)   state (with process noise when enabled)
    u       (E, N+1, 1)    applied input
    y_clean (E, N+1, ny)   h(x)
    y       (E, N+1, ny)   what the observer receives (sensor model applied)
    z       (E, N+1, nz)   KKL latent label: joint RK4 of zdot = A z + B h(x), z0 = 0
    z_obs   (E, N+1, nz)   the latent the DEPLOYED observer carries: the exact discrete
                           filter run on the measured y. Training on z_obs makes the
                           decoder objective identical to deployment (no exposure bias).

Correctness properties (tests/test_data.py):
  * (x, z) are advanced as one augmented state with u at the true RK4 stage times, so
    labels inherit fourth order (the archive's labels were first order).
  * switch instants of piecewise-constant inputs sit on the grid.
  * samples before the washout index ``burn`` are not used for labels; burn comes from
    the exponential tail bound and a declared tolerance, not a fixed percentage.
  * train / val / eval come from independent seed streams; eval seeds depend only on
    (system, tier, split), never on the training seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import qmc

from . import design as D
from . import signals as S
from .cache import DATA_MODULES, module_hash
from .cache import cached as _cached
from .config import ROOT, digest, get
from .systems import build_plant

CACHE = ROOT / "experiments" / "cache"
SPLIT_BASE = {"dev": 1234, "test": 9876}


# ---------------------------------------------------------------------------
# Design (A, B, dt) for one system
# ---------------------------------------------------------------------------

@dataclass
class Design:
    A: np.ndarray
    B: np.ndarray
    dt: float
    N: int
    alpha: float
    step: dict          # exact latent step weights (numpy)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.N + 1) * self.dt

    @property
    def T(self) -> float:
        return self.N * self.dt


def make_design(scfg: dict) -> Design:
    # alpha_scale rescales the observer bandwidth of every plant at once (node C2)
    nz, ny = int(scfg["nz"]), len(scfg["sensors"])
    alpha = float(scfg["alpha"]) * float(scfg.get("alpha_scale", 1.0))
    A, B = D.butterworth_AB(nz, alpha, ny)
    dt = float(scfg.get("dt") or D.choose_dt(A, 0.1))
    N = int(math.ceil(float(scfg["T"]) / dt))
    return Design(A, B, dt, N, alpha, D.latent_step(A, B, dt))


# ---------------------------------------------------------------------------
# Joint (x, z) integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def integrate(plant, des: Design, x0: np.ndarray, sigs: list, sigma_p=None,
              seed: int = 0, device="cpu") -> tuple:
    """RK4 on w = (x, z) with xdot = f(x, u_k), zdot = A z + B h(x). float64.

    The input is applied through a ZERO-ORDER HOLD at the sample rate (digital
    actuation): u(t) = u_k = s(t_k) on [t_k, t_{k+1}). Piecewise-constant families are
    unchanged (their switches sit on the grid); smooth families are their sample-and-hold
    versions. Every observer is therefore given exactly the input the plant received --
    feeding the model-based filters (u_k + u_{k+1}) / 2 and u_{k+1} inside a step was
    wrong at every switch (bioreactor prbs: EKF NRMSE 0.187 instead of 0.006)."""
    dev = torch.device(device)
    A = torch.as_tensor(des.A, dtype=torch.float64, device=dev)
    B = torch.as_tensor(des.B, dtype=torch.float64, device=dev)
    t = des.t
    ug = torch.as_tensor(np.stack([s(t) for s in sigs], 1), dtype=torch.float64, device=dev)  # (N+1, E)
    x = torch.as_tensor(np.asarray(x0), dtype=torch.float64, device=dev)
    z = torch.zeros(x.shape[0], A.shape[0], dtype=torch.float64, device=dev)
    h = des.dt
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    sp = None if sigma_p is None else torch.as_tensor(sigma_p, dtype=torch.float64,
                                                       device=dev) * math.sqrt(h)

    def rhs(uk, xx, zz):
        return plant.f(xx, uk[:, None]), zz @ A.T + plant.h(xx) @ B.T

    xs, zs = [x], [z]
    for k in range(des.N):
        uk = ug[k]
        a1, b1 = rhs(uk, x, z)
        a2, b2 = rhs(uk, x + h / 2 * a1, z + h / 2 * b1)
        a3, b3 = rhs(uk, x + h / 2 * a2, z + h / 2 * b2)
        a4, b4 = rhs(uk, x + h * a3, z + h * b3)
        x = x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        z = z + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        if sp is not None:          # Euler-Maruyama increment after the RK4 step
            x = x + sp * torch.randn(x.shape, generator=gen, dtype=x.dtype, device=dev)
        xs.append(x)
        zs.append(z)
    return torch.stack(xs, 1), torch.stack(zs, 1), ug.T[..., None]


@torch.no_grad()
def run_latent(des: Design, y: torch.Tensor, z0: torch.Tensor = None) -> torch.Tensor:
    """The deployed latent filter on a measured sequence y (E, N+1, ny)."""
    st = {k: torch.as_tensor(v, dtype=y.dtype, device=y.device) for k, v in des.step.items()}
    z = torch.zeros(y.shape[0], des.A.shape[0], dtype=y.dtype, device=y.device) \
        if z0 is None else z0
    out = [z]
    for k in range(y.shape[1] - 1):
        z = z @ st["Ad"].T + y[:, k] @ st["W0"].T + y[:, k + 1] @ st["W1"].T
        out.append(z)
    return torch.stack(out, 1)


# ---------------------------------------------------------------------------
# Sensor and input models
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure(y_clean: torch.Tensor, sensor: dict, y_std, seed: int) -> torch.Tensor:
    """Apply the sensor model. Every magnitude is relative to the output's reference std.

    sensor keys: noise (rel. std), kind (gaussian|colored), rho, outlier_rate,
    outlier_scale, quant (step, rel.), drift (amplitude, rel.), dropout (hold prob.)
    """
    dev = y_clean.device
    g = torch.Generator(device=dev).manual_seed(int(seed) + 7)
    ys = torch.as_tensor(np.asarray(y_std), dtype=y_clean.dtype, device=dev)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=y_clean.dtype, device=dev)
    ru = lambda *s: torch.rand(*s, generator=g, dtype=y_clean.dtype, device=dev)
    E, N1, ny = y_clean.shape
    y = y_clean.clone()
    sd = float(sensor.get("noise", 0.0))
    if sd > 0:
        w = rn(E, N1, ny)
        if sensor.get("kind", "gaussian") == "colored":
            rho = float(sensor.get("rho", 0.95))
            v = torch.empty_like(w)
            v[:, 0] = w[:, 0]
            for k in range(1, N1):
                v[:, k] = rho * v[:, k - 1] + math.sqrt(1 - rho ** 2) * w[:, k]
            w = v
        y = y + sd * ys * w
    if float(sensor.get("drift", 0.0)) > 0:
        b = torch.cumsum(rn(E, N1, ny), 1)
        b = b / b.abs().amax(1, keepdim=True).clamp(min=1e-12)
        y = y + float(sensor["drift"]) * ys * b
    if float(sensor.get("outlier_rate", 0.0)) > 0:
        m = ru(E, N1, ny) < float(sensor["outlier_rate"])
        sgn = torch.where(ru(E, N1, ny) < 0.5, -1.0, 1.0).to(y.dtype)
        y = torch.where(m, y + float(sensor.get("outlier_scale", 12.0)) * ys * sgn, y)
    if float(sensor.get("quant", 0.0)) > 0:
        q = float(sensor["quant"]) * ys
        y = torch.round(y / q) * q
    if float(sensor.get("dropout", 0.0)) > 0:
        hold = ru(E, N1) < float(sensor["dropout"])
        for k in range(1, N1):
            y[:, k] = torch.where(hold[:, k, None], y[:, k - 1], y[:, k])
    return y


@torch.no_grad()
def corrupt_input(u: torch.Tensor, spec: dict, u_ref: float, seed: int) -> torch.Tensor:
    """The input the OBSERVER is given (the plant always receives the true u).
    spec keys: noise (rel. to u_ref), bias (rel.), delay (steps)."""
    out = u.clone()
    d = int(spec.get("delay", 0))
    if d > 0:
        out = torch.cat([out[:, :1].expand(-1, d, -1), out[:, :-d]], 1)
    if float(spec.get("bias", 0.0)):
        out = out + float(spec["bias"]) * u_ref
    if float(spec.get("noise", 0.0)) > 0:
        g = torch.Generator(device=u.device).manual_seed(int(seed) + 11)
        out = out + float(spec["noise"]) * u_ref * torch.randn(
            out.shape, generator=g, dtype=out.dtype, device=out.device)
    return out


# ---------------------------------------------------------------------------
# Episode container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeSet:
    x: torch.Tensor
    u: torch.Tensor
    y_clean: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    z_obs: torch.Tensor
    families: list
    burn: int
    dt: float
    meta: list = field(default_factory=list)
    n_rejected: int = 0

    def __len__(self):
        return self.x.shape[0]

    def to(self, device, dtype=torch.float32) -> "EpisodeSet":
        kw = {k: getattr(self, k).to(device=device, dtype=dtype)
              for k in ("x", "u", "y_clean", "y", "z", "z_obs")}
        return EpisodeSet(**kw, families=self.families, burn=self.burn, dt=self.dt,
                          meta=self.meta, n_rejected=self.n_rejected)

    def select(self, idx) -> "EpisodeSet":
        idx = list(idx)
        kw = {k: getattr(self, k)[idx] for k in ("x", "u", "y_clean", "y", "z", "z_obs")}
        return EpisodeSet(**kw, families=[self.families[i] for i in idx], burn=self.burn,
                          dt=self.dt, meta=[self.meta[i] for i in idx] if self.meta else [],
                          n_rejected=self.n_rejected)

    def with_sensor(self, des: Design, sensor: dict, y_std, seed: int) -> "EpisodeSet":
        """Same episodes, different sensor model (paired across conditions)."""
        y = measure(self.y_clean.double(), sensor, y_std, seed).to(self.y.dtype)
        zo = run_latent(des, y.double()).to(self.z.dtype)
        return EpisodeSet(self.x, self.u, self.y_clean, y, self.z, zo, self.families,
                          self.burn, self.dt, self.meta, self.n_rejected)


def _sobol(box: np.ndarray, n: int, seed: int) -> np.ndarray:
    m = int(math.ceil(math.log2(max(n, 2))))
    X = qmc.Sobol(d=len(box), scramble=True, seed=seed).random_base2(m)[:n]
    return box[:, 0] + X * (box[:, 1] - box[:, 0])


def _signals(families, n, regime, rng, des, scfg, n_strata=4):
    T = des.T
    out = []
    for r in range(n):
        fam = families[r % len(families)]
        s = S.sample_signal(fam, regime, rng, des.dt, T, stratum=r // len(families),
                            n_strata=n_strata, scale=float(scfg.get("u_scale", 1.0)))
        out.append(S.enforce_bounds(s, des.dt, T, scfg.get("u_bounds")))
    return out


def generate(plant, des: Design, scales: dict, scfg: dict, dcfg: dict, families: list,
             n: int, regime: str, seed: int, device, ic_box=None, sensor=None,
             process_noise: bool = True) -> EpisodeSet:
    """Forced episodes with a crossed design: each signal is paired with k initial
    conditions and each initial condition with ~k signals, so the same state occurs
    under different input histories and the state cannot stand in for the input."""
    rng = np.random.RandomState(seed)
    k = int(dcfg.get("k_cross", 3))
    n_sig = n_ic = int(math.ceil(n / k))
    sigs = _signals(families, n_sig, regime, rng, des, scfg)
    box = np.asarray(ic_box if ic_box is not None else scfg["ic_box"], dtype=np.float64)
    ics = _sobol(box, n_ic, seed)
    pick = [(i // k, i % n_ic) for i in range(n)]
    X0 = np.stack([ics[c] for _, c in pick])
    sg = [sigs[s] for s, _ in pick]

    dom = np.asarray(scales["domain"])
    ctr, half = dom.mean(1), (dom[:, 1] - dom[:, 0]) / 2
    rs = float(dcfg.get("reject_scale", 1.5))
    lo, hi = ctr - rs * half, ctr + rs * half
    sig_p = (float(dcfg.get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
             if process_noise and float(dcfg.get("noise_proc", 0.0)) > 0 else None)

    xs, zs, us = [None] * n, [None] * n, [None] * n
    todo, n_rej = list(range(n)), 0
    for attempt in range(6):
        if not todo:
            break
        for c0 in range(0, len(todo), 512):
            ids = todo[c0:c0 + 512]
            x, z, u = integrate(plant, des, X0[ids], [sg[i] for i in ids], sig_p,
                                seed=seed * 7919 + attempt * 104729 + c0, device=device)
            for j, i in enumerate(ids):
                xs[i], zs[i], us[i] = x[j], z[j], u[j]
        bad = []
        for i in todo:
            xi = xs[i]
            lo_t = torch.as_tensor(lo, dtype=xi.dtype, device=xi.device)
            hi_t = torch.as_tensor(hi, dtype=xi.dtype, device=xi.device)
            if not (torch.isfinite(xi).all() and (xi >= lo_t).all() and (xi <= hi_t).all()):
                bad.append(i)
        n_rej += len(bad)
        for i in bad:                                     # fresh IC, same signal
            X0[i] = box[:, 0] + rng.uniform(size=len(box)) * (box[:, 1] - box[:, 0])
        todo = bad
    keep = [i for i in range(n) if i not in set(todo)]

    x = torch.stack([xs[i] for i in keep])
    z = torch.stack([zs[i] for i in keep])
    u = torch.stack([us[i] for i in keep])
    y_clean = torch.stack([plant.h(xi) for xi in x])
    sens = dcfg.get("sensor", {}) if sensor is None else sensor
    y = measure(y_clean, sens, scales["y_std"], seed)
    z_obs = run_latent(des, y)
    burn = int(math.ceil(scales["t_burn"] / des.dt))
    meta = [{"ic": X0[i].tolist(), **sg[i].describe()} for i in keep]
    f32 = lambda a: a.float().cpu()
    return EpisodeSet(f32(x), f32(u), f32(y_clean), f32(y), f32(z), f32(z_obs),
                      [sg[i].family for i in keep], burn, des.dt, meta, n_rej)


# ---------------------------------------------------------------------------
# System scales: operating envelope, reference stds, washout
# ---------------------------------------------------------------------------

def system_scales(cfg: dict, device="cpu") -> dict:
    """Measured once per system (cached): the envelope forced trajectories visit at OOD
    amplitudes (the domain the base maps must cover), reference stds at ID amplitudes
    (the yardstick for noise levels and NRMSE), and the washout / memory horizons."""
    scfg, dcfg = cfg["system"], cfg["data"]
    key = digest(cfg, "system.plant", "system.params", "system.sensors", "system.ic_box",
                 "system.T", "system.nz", "system.alpha", "system.actuator",
                 "system.u_scale", "system.u_bounds", "system.dt", "system.alpha_scale",
                 "data.envelope_safety", "data.eps_label",
                 extra=[module_hash(DATA_MODULES), torch.device(device).type])
    path = CACHE / "scales" / f"{scfg['name']}_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    des = make_design(scfg)
    plant = build_plant(scfg)
    rng = np.random.RandomState(12345)
    box = np.asarray(scfg["ic_box"], dtype=np.float64)

    def probe(regime, fams, n, seed):
        sigs = _signals(fams, n, regime, rng, des, scfg)
        x, z, _ = integrate(plant, des, _sobol(box, n, seed), sigs, device=device)
        ok = torch.isfinite(x).all(2).all(1) & torch.isfinite(z).all(2).all(1)
        return x[ok], z[ok], int((~ok).sum())

    fams = [f for f in S.TRAIN_FAMILIES if f != "zero"]
    x, z, n_div = probe("ood", fams, 96, 1)
    xf = x.reshape(-1, x.shape[-1])
    lo, hi = xf.min(0).values.cpu().numpy(), xf.max(0).values.cpu().numpy()
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    sf = float(dcfg.get("envelope_safety", 1.15))
    dom = np.stack([np.minimum(ctr - sf * half, box[:, 0]),
                    np.maximum(ctr + sf * half, box[:, 1])], 1)
    y_bar = float(plant.h(xf).norm(dim=-1).max())          # (A2): bound on ||h(x)||_2
    z_scale = float(z[:, z.shape[1] // 2:].pow(2).sum(-1).mean().sqrt())
    kappa, alpha = D.spectral_bounds(des.A)
    Bn = float(np.linalg.norm(des.B, 2))
    eps = float(dcfg.get("eps_label", 1e-3))
    t_burn = D.washout_time(kappa, alpha, Bn, y_bar, z_scale, eps)

    xr, _, _ = probe("id", S.families_for("id"), 64, 2)
    b = int(math.ceil(t_burn / des.dt))
    xr = xr[:, b:].reshape(-1, xr.shape[-1])
    out = {
        "domain": dom.tolist(), "y_bar": y_bar, "z_scale": z_scale,
        "t_burn": t_burn, "memory_s": D.memory_length(kappa, alpha, Bn, y_bar, z_scale, eps),
        "x_std": xr.std(0).clamp(min=1e-9).cpu().tolist(),
        "y_std": plant.h(xr).std(0).clamp(min=1e-9).cpu().tolist(),
        "x_mean": xr.mean(0).cpu().tolist(),
        "dt": des.dt, "N": des.N, "n_probe_diverged": n_div,
        "design_check": D.validate(des.A, des.B, des.dt, des.alpha),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")     # atomic: concurrent jobs read this file
    tmp.write_text(json.dumps(out, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))
    os.replace(tmp, path)
    return out


# ---------------------------------------------------------------------------
# Assembled datasets (cached)
# ---------------------------------------------------------------------------

def training_data(cfg: dict, seed: int, device="cpu") -> dict:
    """{p1, train, val, coll}. Train/val depend on the training seed."""
    scfg, dcfg = cfg["system"], cfg["data"]
    scales = system_scales(cfg, device)
    # device type is part of the key: torch generators draw different noise on CPU and CUDA
    key = digest(cfg, "system", "data", extra=[seed, module_hash(DATA_MODULES), torch.device(device).type])
    path = CACHE / "data" / f"{scfg['name']}_train_{key}.pt"

    def build():
        des = make_design(scfg)
        plant = build_plant(scfg, perturb=float(scfg.get("mismatch", 0.0)))
        dom = np.asarray(scales["domain"])
        base = 100_000 * (seed + 1)
        # one (zero) signal: crossing would only replicate each IC k times
        p1 = generate(plant, des, scales, scfg, dict(dcfg, k_cross=1), ["zero"],
                      int(dcfg["n_p1"]), "train", base + 1, device, ic_box=dom)
        tr = generate(plant, des, scales, scfg, dcfg, S.TRAIN_FAMILIES, int(dcfg["n_train"]),
                      "train", base + 2, device)
        va = generate(plant, des, scales, scfg, dcfg, S.TRAIN_FAMILIES, int(dcfg["n_val"]),
                      "train", base + 3, device)
        Xc = _sobol(dom, int(dcfg["n_coll"]), base + 4)
        xt = torch.as_tensor(Xc, dtype=torch.float64)
        nominal = build_plant(scfg)                     # physics uses the NOMINAL model
        coll = {"x": xt.float(), "f0": nominal.f(xt, torch.zeros(len(xt), 1, dtype=xt.dtype)).float(),
                "y": nominal.h(xt).float()}
        return {"p1": p1, "train": tr, "val": va, "coll": coll}

    return _cached(path, build)


def split_seed(system: str, tier: str, split: str = "dev", rep: int = 0) -> int:
    key = f"{system}|{tier}" + (f"|{rep}" if rep else "")
    return SPLIT_BASE[split] + int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100_000


def eval_data(cfg: dict, tier: str, plant_shift: float = 0.0, device="cpu") -> EpisodeSet:
    """Evaluation episodes for one tier. Independent of the training seed.

    ``plant_shift`` re-simulates the same draw on a plant whose parameters drifted
    after training (the model-based baselines still hold the nominal model)."""
    scfg, dcfg, ecfg = cfg["system"], cfg["data"], cfg["eval"]
    scales = system_scales(cfg, device)
    split, rep = ecfg.get("split", "dev"), int(ecfg.get("rep", 0))
    key = digest(cfg, "system", "data", "eval.n_episodes",
                 extra=[tier, split, rep, plant_shift, module_hash(DATA_MODULES),
                        torch.device(device).type])
    path = CACHE / "data" / f"{scfg['name']}_eval_{tier}_{key}.pt"

    def build():
        des = make_design(scfg)
        shift = float(scfg.get("mismatch", 0.0)) + float(plant_shift)
        plant = build_plant(scfg, perturb=shift)
        regime = {"id": "id", "ood": "ood", "waveform_ood": "waveform_ood"}[tier]
        # k_cross = 1: every evaluation episode has its own signal and IC, so episodes are
        # independent units for the bootstrap (k = 3 gave 22 distinct signals in 64)
        return generate(plant, des, scales, scfg, dict(dcfg, k_cross=1), S.families_for(tier),
                        int(ecfg["n_episodes"]), regime,
                        split_seed(scfg["name"], tier, split, rep), device)

    return _cached(path, build)
