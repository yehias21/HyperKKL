r"""Published learned KKL / Luenberger observers for driven plants, as baselines in the HyperKKL harness (PEERS.md).

Package-agnostic like jf.py: import jf_env.setup(plant) first (code/bpkg = eventmix pkg copy).

Implemented methods (each a training function + a deployed observer obeying observers.py's contract:
reset(B, x0_hat, u0) -> estimate() -> advance(y_k, u_k, y_k1, u_k1); the scorer is jf_run.eval_sets, unchanged):

NLOX   Woelk, Morris & Tang (2026, arXiv:2602.24252, J. Process Control), the model-free realisation of the
       Bernard & Andrieu (IEEE TAC 2019) stationary-transformation observer for input-affine plants:
           z' = A z + B y + omega(z) u,      x^ = T_dagger(z),      z(0) = 0,
       omega: R^nz -> R^{nz x nu} and T_dagger: R^nz -> R^nx MLPs (tanh), trained JOINTLY end-to-end on the
       state error of the deployed rollout from z(0) = 0 (their loss (8)); state labels in training only.
       Harness latent filter (A, B) = the plant's Butterworth design shared by every KKL-family method (theirs:
       A = diag(-lambda_i), B = 1, forward Euler; ours: exact step + exponential-midpoint injection).

DKKL   Peralez & Nadri (L4DC 2024, PMLR 242:929-940; code github.com/jolindien-git/DeepKKL, L4DC_2024/), the
       deep model-free discrete-time KKL observer with switching:
           z_{k+1} = A^(i) z_k + B y_k,  x^_k = psi^(i)(z_k),   A^(i) LEARNED (full matrix), B = 1 (fixed),
       i = 2 transient: z_0 = phi(y_0) (learned), psi = T* (learned extension), trained end-to-end;
       i = 1 asymptotic: z_0 = T(x_0) (learned encoder), psi = T^-1, initialised from the transient model,
              trained end-to-end + latent consistency ||T(x_k) - z_k|| (their train.py, use_encoder branch);
       switching: eta^(i)_{k+1} = a eta^(i)_k + eps^(i)T Lambda eps^(i),
              eps^(i) = psi^(i)(A^(i) z + B y) - psi^(i)(A^(i) z + B h(x^(i))), output the observer with the
              smaller eta (a = 0.99, Lambda = diag(1/sigma^2): their P = I in sigma units). Input-blind, as
              published (they leave the model-free non-autonomous case to future work).
DKKL-inj  Peralez & Nadri eq. (17) (= Peralez & Nadri CDC 2021): the asymptotic observer with the model-based
       input injection  z_{k+1} = A z_k + B y_k + Psi(z_k, u_k),
           Psi(z, u) = T(F(T^-1(z), u)) - T(F(T^-1(z), u0)),  u0 = 0,
       F = one RK4 step (dt, zero-order hold) of the MODELLER's model (data.model_plant: partial / mismatched,
       the same model kkl_mb and the filters hold). Trained end-to-end through Psi (they apply (17) to a
       model trained at constant input; we train with it in the loop, which can only help it).

Fairness rules (PREREG.md section 3): same mixed training / validation data as HyperKKL default, same optimiser
(SOAP lr 1e-2, 100-step warmup + cosine, global-norm clip 1), same batch geometry as HyperKKL's rollout variant
(32 episodes x K = 256 samples per step), best-validation checkpoint, trainable parameters matched to the TOTAL
trained parameters of the HyperKKL default of the same plant (phase-1 T0, T0+ and the phase-2 conditioner,
context and history features), `hk_budget`.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperkkl.beat import HistoryFeatures, make_opt
from hyperkkl.conditioning import match_width
from hyperkkl.monitor import TrainMonitor
from hyperkkl.nets import MLP, BaseMap, build_context, n_params
from hyperkkl.observers import KKLObserver, LatentStep, Observer
from hyperkkl.train import _Graphed, _sched, _stats


# ===================================================================================================== budget
def hk_budget(cfg, data, des, base, opts) -> dict:
    """Total trained parameters of the HyperKKL default (A3) of this plant/seed: T0 + T0+ (phase 1; T0+ is
    fine-tuned in phase 2) + LoRA conditioner (match_width at phase2.budget) + SFM context + z-window features."""
    p = cfg["phase2"]
    tr = data["train"]
    nz = des.A.shape[0]
    z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
    ctx = build_context(p["context"]["kind"], p["context"], 1, des.dt, np.linalg.eigvals(des.A),
                        float(cfg["system"].get("u_scale", 1.0)))
    hist = HistoryFeatures(nz, int(opts["window"]), int(opts["stride"]), None, z_std=z_std)
    _, npar = match_width("lora", base["T0inv"], ctx.dim + hist.dim, int(p["budget"]), int(p["rank"]), z_std)
    out = {"T0": n_params(base["T0"], False), "T0inv": n_params(base["T0inv"], False), "cond": npar,
           "ctx": n_params(ctx, False), "hist": n_params(hist, False)}
    out["total"] = sum(out.values())
    return out


def solve_width(count, budget, lo=2, hi=1 << 12):
    """Width w (>= lo) with count(w) closest to budget (count increasing in w)."""
    a, b = lo, hi
    while a < b:
        m = (a + b) // 2
        if count(m) < budget:
            a = m + 1
        else:
            b = m
    return min({max(a - 1, lo), a}, key=lambda w: abs(count(w) - budget))


def _opt(name, params, lr):
    name = name.lower()
    if name == "rmsprop":                       # Woelk et al.'s optimiser (published-protocol variant)
        return torch.optim.RMSprop(params, lr=lr)
    return make_opt(name, [{"params": list(params), "lr": lr}])


class _Streams:
    """Stateful truncated BPTT over whole episodes (the harness NLOX scheme): Bs streams, each walks one episode
    from k = 0 in windows of K steps (state detached between optimiser steps) and restarts on a fresh episode
    when the next window would run past the end. Index buffers are static (CUDA-graph safe)."""

    def __init__(self, n_ep, N, Bs, K, nz, device, gen):
        self.n_ep, self.N, self.K, self.g = n_ep, N, K, gen
        self.ep = torch.randint(0, n_ep, (Bs,), device=device, generator=gen)
        self.kk = torch.zeros(Bs, dtype=torch.long, device=device)
        self.z = torch.zeros(Bs, nz, device=device)
        self.start = torch.ones(Bs, dtype=torch.bool, device=device)

    def advance_if_done(self):
        done = self.kk + self.K > self.N
        if done.any():
            self.ep[done] = torch.randint(0, self.n_ep, (int(done.sum()),), device=self.ep.device, generator=self.g)
            self.kk[done] = 0
            self.z[done] = 0.0
        self.start.copy_(self.kk == 0)

    def step_done(self, z_last):
        self.z.copy_(z_last.detach())
        self.kk.add_(self.K)


def _train_loop(name, loss_fn, params, opt, steps, streams, out, validate, every, log, modules, graphs=True):
    """Shared optimiser loop: graphed loss+backward, clip, step, cosine schedule, best-validation checkpoint."""
    sch = _sched(opt, steps, 100)
    graphed = _Graphed(loss_fn, params, enabled=graphs)
    try:
        streams.advance_if_done()
        graphed()                                      # capture (or eager warm-up) once, before the loop
    except Exception as ex:                            # model code that cannot be captured: eager, same maths
        log(f"    [{name}] CUDA-graph capture failed ({type(ex).__name__}: {ex}); eager")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        graphed = _Graphed(loss_fn, params, enabled=False)
    mon = TrainMonitor(params, opt)
    best, best_state, hist = float("inf"), None, []
    for s in range(steps):
        streams.advance_if_done()
        loss = graphed()
        mon.clip()
        opt.step(); sch.step()
        streams.step_done(out["z"])
        if (s + 1) % every == 0 or s == steps - 1:
            v = validate()
            hist.append([s + 1, float(loss.detach()), v])
            log(f"    [{name}] step {s + 1:5d}  train {hist[-1][1]:.3e}  val {v:.3e}  " + mon.fmt(mon.flush(s + 1)))
            if best_state is None or (np.isfinite(v) and v < best):
                best, best_state = v, [copy.deepcopy(m.state_dict()) for m in modules]
    for m, sd in zip(modules, best_state):
        m.load_state_dict(sd)
    return best, hist, mon.windows


# ===================================================================================================== NLOX
class Omega(nn.Module):
    """omega(z) in R^{nz x nu}; injection omega(z) u (z in z_std units inside, u in u_ref units)."""

    def __init__(self, nz, nu, width, depth, z_std, u_ref=1.0):
        super().__init__()
        self.net = MLP(nz, nz * nu, width, depth, "tanh")
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32))
        self.nz, self.nu, self.u_ref = nz, nu, float(u_ref)

    def forward(self, z, u):
        om = self.net(z / self.z_std).view(-1, self.nz, self.nu)
        return torch.bmm(om, (u / self.u_ref)[:, :, None])[:, :, 0] * self.z_std


class NoCtxDecoder(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z, c):
        return self.dec(z)


def nlox_count(nz, nx, nu, w, om_depth=2, dec_depth=3):
    om = Omega(nz, nu, w, om_depth, torch.ones(nz))
    dec = BaseMap(nz, nx, w, dec_depth, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(nx), torch.ones(nx)))
    return n_params(om) + n_params(dec)


def train_nlox(cfg, data, sc, des, dev, seed, o, log=print) -> dict:
    """o: optimizer, lr, steps, K, Bs, val_every, and either budget (one width w for omega and T_dagger solved so
    that the total hits it) or om_width/dec_width (published sizes)."""
    torch.manual_seed(seed)
    t0 = time.time()
    tr, va = data["train"].to(dev), data["val"].to(dev)
    nz, nx, nu = des.A.shape[0], tr.x.shape[-1], tr.u.shape[-1]
    sig = torch.as_tensor(sc["x_std"], dtype=torch.float32, device=dev)
    zf = tr.z_obs[:, tr.burn:].reshape(-1, nz)
    xf = tr.x[:, tr.burn:].reshape(-1, nx)
    if o.get("budget"):
        w = solve_width(lambda w_: nlox_count(nz, nx, nu, w_), int(o["budget"]))
        om_w, dec_w = w, w
    else:
        om_w, dec_w = int(o["om_width"]), int(o["dec_width"])
    om = Omega(nz, nu, om_w, 2, zf.std(0).clamp(min=1e-6), float(cfg["system"].get("u_scale", 1.0))).to(dev)
    dec = BaseMap(nz, nx, dec_w, 3, "tanh", _stats(zf), _stats(xf)).to(dev)
    params = list(om.parameters()) + list(dec.parameters())
    npar = sum(q.numel() for q in params)
    opt = _opt(o["optimizer"], params, float(o["lr"]))
    lat = LatentStep(des, dev)
    K, burn, N = int(o["K"]), tr.burn, tr.x.shape[1] - 1
    g = torch.Generator(device=dev).manual_seed(seed)
    st = _Streams(len(tr), N, int(o["Bs"]), K, nz, dev, g)
    log(f"    [nlox] widths omega {om_w} T_dagger {dec_w} (3 x tanh)  params {npar}  opt {o['optimizer']} "
        f"lr {o['lr']} steps {o['steps']} K {K} Bs {o['Bs']}")
    out = {}
    ar = torch.arange(1, K + 1, device=dev)[None]

    def loss_fn():
        z, zs, msk = st.z, [], []
        for j in range(K):
            k = st.kk + j
            uk = tr.u[st.ep, k]
            z = lat(z, tr.y[st.ep, k], tr.y[st.ep, k + 1], lambda zz: om(zz, uk))
            zs.append(z)
            msk.append(k + 1 >= burn)
        out["z"] = z
        Z, Mk = torch.stack(zs, 1), torch.stack(msk, 1)
        X = tr.x[st.ep[:, None], st.kk[:, None] + ar]
        err = ((dec(Z.reshape(-1, nz)) - X.reshape(-1, nx)) / sig).pow(2).mean(-1)
        wgt = Mk.reshape(-1).float()
        return (err * wgt).sum() / wgt.sum().clamp(min=1.0)

    @torch.no_grad()
    def validate():
        tot = 0.0
        for i in range(0, len(va), 64):
            e = torch.arange(i, min(i + 64, len(va)), device=dev)
            z, zs = torch.zeros(len(e), nz, device=dev), []
            for k in range(N):
                uk = va.u[e, k]
                z = lat(z, va.y[e, k], va.y[e, k + 1], lambda zz: om(zz, uk))
                if k + 1 >= burn:
                    zs.append(z)
            Z = torch.stack(zs, 1)
            X = va.x[e, burn:burn + Z.shape[1]]
            tot += float(((dec(Z.reshape(-1, nz)) - X.reshape(-1, nx)) / sig).pow(2).mean()) * len(e)
        v = tot / len(va)
        return v if math.isfinite(v) else float("inf")

    best, hist, mon = _train_loop("nlox", loss_fn, params, opt, int(o["steps"]), st, out, validate,
                                  int(o["val_every"]), log, [om, dec], graphs=cfg.get("cuda_graphs", True))
    for q in params:
        q.requires_grad_(False)
    return {"omega": om, "dec": dec, "n_params": npar, "widths": [om_w, dec_w], "val": best, "hist": hist,
            "opt": mon, "wall_s": time.time() - t0}


def nlox_observer(r, des, dev, name="nlox"):
    from hyperkkl.evaluate import RawInput
    return KKLObserver(name, LatentStep(des, dev), NoCtxDecoder(r["dec"]), ctx=RawInput(), injection=r["omega"])


# ===================================================================================================== DKKL
class DKKLModel(nn.Module):
    """One discrete-time KKL observer of Peralez & Nadri (2024): learnable A (full), B = 1 fixed, decoder psi,
    encoder T (x -> z) and initialiser phi (y_0 -> z_0, half width) as in their KKL_AutoEncoder (tanh MLPs with
    net_arch = [w, w]). Units: y enters as y / y_std (B = 1 on standardised outputs), z is standardised by fixed
    statistics of an initial rollout, x is decoded in x_std units around the training mean."""

    def __init__(self, nx, ny, nz, w, Ad, y_std, x_stats, z_stats):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.A = nn.Linear(nz, nz, bias=False)
        with torch.no_grad():
            self.A.weight.copy_(torch.as_tensor(Ad, dtype=torch.float32))
        self.register_buffer("B", torch.ones(nz, ny))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        self.register_buffer("x_mu", x_stats[0].float()); self.register_buffer("x_sd", x_stats[1].float())
        self.register_buffer("z_mu", z_stats[0].float()); self.register_buffer("z_sd", z_stats[1].float())
        self.dec = MLP(nz, nx, w, 2, "tanh")
        self.enc = MLP(nx, nz, w, 2, "tanh")
        self.phi = MLP(ny, nz, max(1, w // 2), 2, "tanh")
        self.psi_inj = None                               # set by DKKL-inj: callable (z, u) -> Psi

    def step(self, z, y, u=None):
        zn = self.A(z) + (y / self.y_std) @ self.B.T
        if self.psi_inj is not None:
            zn = zn + self.psi_inj(z, u)
        return zn

    def decode(self, z):
        return self.dec((z - self.z_mu) / self.z_sd) * self.x_sd + self.x_mu

    def encode(self, x):
        return self.enc((x - self.x_mu) / self.x_sd) * self.z_sd + self.z_mu

    def init_y(self, y0):
        return self.phi(y0 / self.y_std) * self.z_sd + self.z_mu


def dkkl_count(nx, ny, nz, w, use_phi=True):
    m = DKKLModel(nx, ny, nz, w, np.eye(nz), torch.ones(ny), (torch.zeros(nx), torch.ones(nx)),
                  (torch.zeros(nz), torch.ones(nz)))
    n = n_params(m)
    return n if use_phi else n - n_params(m.phi)


class ModelInjection(nn.Module):
    """Psi(z, u) = T(F(T^-1(z), u)) - T(F(T^-1(z), u0)), u0 = 0, F = RK4 step of the modeller's model (ZOH u).
    T, T^-1 = the observer's own encoder / decoder (gradients flow through both)."""

    def __init__(self, model, dt, owner_ref):
        super().__init__()
        self.model, self.dt = model, float(dt)
        self._owner = owner_ref                           # list([DKKLModel]) (no submodule cycle)

    def F(self, x, u):
        f, h = self.model.f, self.dt
        k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, z, u):
        m = self._owner[0]
        x = m.decode(z)
        return m.encode(self.F(x, u)) - m.encode(self.F(x, torch.zeros_like(u)))


def _huber(a, b, delta):
    return F.huber_loss(a, b, reduction="none", delta=delta).mean(-1)


def _dkkl_stage(stage, m, tr, va, sig, dev, seed, o, log, sensors):
    """stage: 'transient' (z_0 = phi(y_0), loss Huber(psi(z_k), x_k)) | 'asymptotic' (z_0 = T(x_0), + latent
    consistency Huber(T(x_k), z_k) in z_sd units) | 'inj' (asymptotic with Psi(z, u) in the step)."""
    nz, nx = m.nz, m.nx
    delta = float(o.get("huber", 0.1))
    K, N = int(o["K"]), tr.x.shape[1] - 1
    g = torch.Generator(device=dev).manual_seed(seed + {"transient": 0, "asymptotic": 1, "inj": 2}[stage])
    st = _Streams(len(tr), N, int(o["Bs"]), K, nz, dev, g)
    params = [q for q in m.parameters() if q.requires_grad]
    if stage == "transient":
        params = [q for n_, q in m.named_parameters() if not n_.startswith("enc.")]
    else:
        params = [q for n_, q in m.named_parameters() if not n_.startswith("phi.")]
    opt = _opt(o["optimizer"], params, float(o["lr"]))
    steps = int(o["steps_" + stage])
    out = {}

    def z_init(e, k0_is_start):
        zi = m.init_y(tr.y[e, 0]) if stage == "transient" else m.encode(tr.x[e, 0])
        return torch.where(k0_is_start[:, None], zi, st.z)

    def loss_fn():
        z = z_init(st.ep, st.start)
        tot, cons = 0.0, 0.0
        for j in range(K):
            k = st.kk + j
            xk = tr.x[st.ep, k]
            tot = tot + _huber(m.decode(z) / sig, xk / sig, delta).mean()
            if stage != "transient":
                cons = cons + _huber(m.encode(xk) / m.z_sd, z / m.z_sd, delta).mean()
            z = m.step(z, tr.y[st.ep, k], tr.u[st.ep, k])
        out["z"] = z
        return (tot + cons) / K

    @torch.no_grad()
    def validate():
        """Deployed observer of this stage on every validation episode, sigma-MSE over the settled window."""
        tot = 0.0
        settle = va.burn
        for i in range(0, len(va), 64):
            e = torch.arange(i, min(i + 64, len(va)), device=dev)
            z = m.init_y(va.y[e, 0]) if stage == "transient" else m.encode(_prior_like(va, e))
            acc, n = 0.0, 0
            for k in range(N + 1):
                if k >= settle:
                    acc += float(((m.decode(z) - va.x[e, k]) / sig).pow(2).mean()); n += 1
                if k < N:
                    z = m.step(z, va.y[e, k], va.u[e, k])
            tot += acc / max(n, 1) * len(e)
        v = tot / len(va)
        return v if math.isfinite(v) else float("inf")

    best, hist, mon = _train_loop(f"dkkl-{stage}", loss_fn, params, opt, steps, st, out, validate,
                                  int(o["val_every"]), log, [m], graphs=o.get("cuda_graphs", True))
    return {"val": best, "hist": hist, "opt": mon}


_PRIOR = {}


def _prior_like(es, e):
    """The harness prior used at deployment (the centre of the IC box), for validating T(x^_0) initialisation."""
    return _PRIOR["x0"].to(es.x.device).expand(len(e), -1)


def _z_stats(Ad, tr, y_std, dev, n=64):
    """Fixed latent standardisation: statistics of z_{k+1} = Ad z + 1 y/y_std from z_0 = 0 (settled part) on n
    training episodes. A units choice only (absorbed by the first decoder layer / last encoder layer)."""
    A = torch.as_tensor(Ad, dtype=torch.float32, device=dev)
    y = tr.y[:n] / y_std
    z = torch.zeros(y.shape[0], A.shape[0], device=dev)
    zs = []
    for k in range(y.shape[1]):
        z = z @ A.T + y[:, k].sum(-1, keepdim=True).expand(-1, A.shape[0])
        if k >= tr.burn:
            zs.append(z)
    Z = torch.cat(zs)
    return Z.mean(0), Z.std(0).clamp(min=1e-6)


def train_dkkl(cfg, data, sc, des, dev, seed, o, log=print, inj=False) -> dict:
    """o: optimizer, lr, steps (total), K, Bs, val_every, budget, huber. inj=False: transient (steps/2) then
    asymptotic initialised from it (steps/2), switching at run time. inj=True: the eq.-(17) asymptotic observer
    with the model-based Psi, all steps."""
    from hyperkkl import data as Dm
    from hyperkkl.evaluate import initial_guess
    torch.manual_seed(seed)
    t0 = time.time()
    tr, va = data["train"].to(dev), data["val"].to(dev)
    nz, nx, ny = des.A.shape[0], tr.x.shape[-1], tr.y.shape[-1]
    sig = torch.as_tensor(sc["x_std"], dtype=torch.float32, device=dev)
    y_std = torch.as_tensor(sc["y_std"], dtype=torch.float32, device=dev)
    _PRIOR["x0"] = initial_guess(cfg, 1, "center", 0, "cpu")
    Ad = np.asarray(des.step["Ad"], float)
    xs = _stats(tr.x[:, tr.burn:].reshape(-1, nx))
    zst = _z_stats(Ad, tr, y_std, dev)
    sensors = list(Dm.model_plant(cfg["system"]).sensors)
    budget = int(o["budget"])
    if inj:
        w = solve_width(lambda w_: dkkl_count(nx, ny, nz, w_, use_phi=False), budget)
    else:
        w = solve_width(lambda w_: 2 * dkkl_count(nx, ny, nz, w_), budget)
    m = DKKLModel(nx, ny, nz, w, Ad, y_std, xs, zst).to(dev)
    res = {"width": w, "stages": {}}
    if inj:
        model = Dm.model_plant(cfg["system"])
        m.psi_inj = ModelInjection(model, des.dt, [m]).to(dev)
        o2 = dict(o, steps_inj=int(o["steps"]))
        res["stages"]["inj"] = _dkkl_stage("inj", m, tr, va, sig, dev, seed, o2, log, sensors)
        npar = dkkl_count(nx, ny, nz, w, use_phi=False)
        models = {"asym": m, "trans": None}
    else:
        o2 = dict(o, steps_transient=int(o["steps"]) // 2, steps_asymptotic=int(o["steps"]) - int(o["steps"]) // 2)
        res["stages"]["transient"] = _dkkl_stage("transient", m, tr, va, sig, dev, seed, o2, log, sensors)
        ma = copy.deepcopy(m)                               # asymptotic initialised with the transient model
        res["stages"]["asymptotic"] = _dkkl_stage("asymptotic", ma, tr, va, sig, dev, seed, o2, log, sensors)
        npar = 2 * dkkl_count(nx, ny, nz, w)
        models = {"asym": ma, "trans": m}
    for mm in models.values():
        if mm is not None:
            for q in mm.parameters():
                q.requires_grad_(False)
    log(f"    [dkkl{'-inj' if inj else ''}] width {w} params {npar} (budget {budget})")
    res.update({"models": models, "n_params": npar, "sensors": sensors, "sigma": sig,
                "val": min(s_["val"] for s_ in res["stages"].values()), "wall_s": time.time() - t0})
    return res


class DKKLObserver(Observer):
    """Deployed Peralez-Nadri observer. mode 'hybrid' (switching between transient and asymptotic), 'trans',
    'asym' (also DKKL-inj). xhat_0 = the harness prior (no measurement yet); the transient latent starts at
    phi(y_0) at the first advance, the asymptotic one at T(x0_hat). x^_k = psi(z_k), z_k built from y_<k
    (their discrete-time structure: one-sample output delay)."""
    learned = True

    def __init__(self, name, trans, asym, sensors, sigma, mode="hybrid", a=0.99):
        self.name, self.mt, self.ma, self.sens, self.mode, self.a = name, trans, asym, list(sensors), mode, float(a)
        self.lam = 1.0 / sigma.pow(2)

    def reset(self, B, x0_hat, u0):
        self.x0, self.first = x0_hat, True
        self.zt = None
        self.za = self.ma.encode(x0_hat) if self.ma is not None else None
        self.eta = torch.zeros(2, B, device=x0_hat.device)
        self.use_t = torch.zeros(B, dtype=torch.bool, device=x0_hat.device)

    def _eps(self, m, z, y, u):
        xh = m.decode(z)
        yh = xh[:, self.sens]
        return m.decode(m.step(z, y, u)) - m.decode(m.step(z, yh, u))

    def estimate(self):
        if self.first:
            return self.x0
        if self.mode == "asym":
            return self.ma.decode(self.za)
        if self.mode == "trans":
            return self.mt.decode(self.zt)
        xt, xa = self.mt.decode(self.zt), self.ma.decode(self.za)
        return torch.where(self.use_t[:, None], xt, xa)

    def advance(self, y_k, u_k, y_k1, u_k1):
        if self.first and self.mt is not None:
            self.zt = self.mt.init_y(y_k)
        self.first = False
        if self.mode == "hybrid":
            for i, (m, z) in enumerate(((self.mt, self.zt), (self.ma, self.za))):
                e = self._eps(m, z, y_k, u_k)
                self.eta[i] = self.a * self.eta[i] + (e.pow(2) * self.lam).sum(-1)
            self.use_t = self.eta[0] <= self.eta[1]          # their tie rule: transient when v0 <= v1
        if self.mt is not None and self.mode != "asym":
            self.zt = self.mt.step(self.zt, y_k, u_k)
        if self.ma is not None and self.mode != "trans":
            self.za = self.ma.step(self.za, y_k, u_k)


def dkkl_observers(r, prefix):
    """The deployed variants of a trained DKKL result: hybrid (the published observer) + its two components."""
    mt, ma = r["models"]["trans"], r["models"]["asym"]
    if mt is None:
        return {prefix: DKKLObserver(prefix, None, ma, r["sensors"], r["sigma"], mode="asym")}
    return {prefix: DKKLObserver(prefix, mt, ma, r["sensors"], r["sigma"], mode="hybrid"),
            prefix + "_trans": DKKLObserver(prefix + "_trans", mt, None, r["sensors"], r["sigma"], mode="trans"),
            prefix + "_asym": DKKLObserver(prefix + "_asym", None, ma, r["sensors"], r["sigma"], mode="asym")}
