"""Evaluation: every observer x every tier x every stress condition, on paired episodes.

Tiers (signals.py, input protocol v2): id, shape_ood, sequence_ood -- every tier keeps
the input inside the same amplitude box; the OOD tiers change its waveform shape or its
event sequence. Conditions (configs/base.yaml ``conditions``)
change only what the observers receive or the plant they face:

    sensor      sensor model override (noise level, outliers, dropout, colour, burst, ...)
    input       corruption of the u given to observers (noise, bias, delay)
    init        prior x0_hat: "center" of the IC box, or "poor" (a far corner)
    plant_shift the plant drifted after training; baselines keep the nominal model
    horizon     episodes horizon x T long (with n_episodes, tiers: condition ``long``)
    tiers       restrict the condition to these tiers

Sensor/input/init conditions reuse the SAME simulated episodes as `clean`, so condition
effects are paired too. Blocks of equal episode length are batched into one pass per
observer; a ``long`` block is a second pass of horizon x N steps.

Every model-based observer holds the MODELLER'S model (data.model_plant: nominal
parameters, system.knowledge full | partial). Oracle filters (ORACLE) are rebuilt per
condition with observers.oracle_noise for that condition's sensor and a Q that includes
the model-error rate between the filter's model and the real plant of the block.

Data-driven baselines (baselines_data: koopman_kf, koopman_kf_bilinear, greybox_ekf,
greybox_ukf, greybox_ekf_oracle) are built from fits made once per job on the learned
observers' own training data (``data_fits`` = baselines_data.fit_all).

Per block and method, besides the per-episode metrics, ``scorecard`` holds the deployment
scorecard (metrics.block_scorecard). ``memory_probe`` is the C2 context-memory probe.
"""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch

from . import baselines_data as BD
from . import data as Dm
from .conditioning import ConditionedDecoder
from .config import deep_merge
from .metrics import block_scorecard, episode_metrics, fit_decay, forget_time
from .nets import ContextEncoder, GRUEncoder, SFMEncoder, WindowEncoder
from .observers import (EKF, UKF, HuberEKF, HuberUKF, KKLObserver, LatentStep, ModelBasedKKL,
                        Observer, ParticleFilter, model_error_rate, oracle_noise, run,
                        time_per_step)
from .systems import build_plant


TRAJ_TIERS = ("shape_ood", "sequence_ood")     # tiers whose first 4 clean trajectories are saved
ORACLE = ("ekf_oracle", "ukf_oracle", "ekf_oracle_huber")


class RawInput(ContextEncoder):
    """c_k = u_k (used by NLOX, whose injection is omega(z) u)."""
    dim = 1

    def seq(self, u):
        return u

    def init_state(self, B, device):
        return None

    def step(self, u_k, state):
        return u_k, None


class BlockOracle(Observer):
    """A condition-oracle filter, rebuilt for every evaluation block by ``for_block(sensor,
    shift)`` (evaluate()). Used directly (cost timing, a plain run) it is the filter of
    the nominal sensor and plant."""
    uses_model = True

    def __init__(self, name, make, sensor, shift=0.0):
        self.name, self._make, self._built = name, make, {}
        self.default = self.for_block(sensor, shift)

    def for_block(self, sensor: dict, shift: float = 0.0) -> Observer:
        key = json.dumps([sensor or {}, float(shift)], sort_keys=True, default=str)
        if key not in self._built:
            self._built[key] = self._make(dict(sensor or {}), float(shift))
        return self._built[key]

    @property
    def m(self):
        return self.default.m

    def reset(self, B, x0_hat, u0):
        return self.default.reset(B, x0_hat, u0)

    def estimate(self):
        return self.default.estimate()

    def advance(self, y_k, u_k, y_k1, u_k1):
        return self.default.advance(y_k, u_k, y_k1, u_k1)


def oracle_maker(cfg, des, scales, model, p0, name):
    """make(sensor, shift) -> the oracle filter ``name`` for one condition block. The
    condition's sensor sets R (observers.oracle_noise); eval.oracle_tuning {q_inflate,
    r_inflate}, chosen per job on validation episodes, scales the model-error part of Q and R."""
    sc, dcfg = cfg["system"], cfg["data"]
    base_shift = float(sc.get("mismatch", 0.0))
    knowledge = sc.get("knowledge", "full")
    u_scale = float(sc.get("u_scale", 1.0))
    me_cache = {}

    def me(shift):
        tot = base_shift + shift
        if not tot and knowledge == "full":
            return None                       # the filter holds the real plant
        if tot not in me_cache:
            me_cache[tot] = model_error_rate(model, build_plant(sc, perturb=tot),
                                             scales["domain"], u_scale)
        return me_cache[tot]

    # q_inflate scales the model-error VARIANCE (the process-noise term is kept), r_inflate the
    # measurement variance; both chosen per job on validation episodes (job.tune_oracle)
    tun = cfg["eval"].get("oracle_tuning") or {}
    qi, ri = float(tun.get("q_inflate", 1.0)), float(tun.get("r_inflate", 1.0))

    def make(sensor, shift):
        huber = name == "ekf_oracle_huber"
        m_e = me(shift)
        m_e = None if m_e is None else np.sqrt(qi) * np.asarray(m_e)
        q_o, r_o = oracle_noise(sensor, scales, float(dcfg.get("noise_proc", 0.0)), des.dt, m_e,
                                r_inflate=ri, outliers="ignore" if huber else "mixture")
        hm = (float(sensor.get("dropout", 0.0)) > 0 or bool(sensor.get("burst"))) \
            and not float(sensor.get("quant", 0.0)) > 0
        if name == "ekf_oracle":
            f = EKF(model, des.dt, q_o, r_o, p0, hold_missing=hm)
        elif name == "ukf_oracle":
            f = UKF(model, des.dt, q_o, r_o, p0, hold_missing=hm)
        else:
            f = HuberEKF(model, des.dt, q_o, r_o, p0, gate=True, hold_missing=hm)
        f.name = name
        return f

    return make


def build_observers(cfg, des, scales, base, trained: dict, nlox_res, device, data_fits: dict = None) -> dict:
    lat = LatentStep(des, device)
    T0, Ti = (base["T0"], base["T0inv"]) if base else (None, None)   # None: filters only
    obs = {}
    for name, r in trained.items():
        if name.split("@")[0] == "latent":
            obs[name] = KKLObserver(name, lat, ConditionedDecoder(Ti, "none", 0, 0),
                                    ctx=r["ctx"], injection=r["cond"], z0_map=T0)
        else:
            obs[name] = KKLObserver(name, lat, r["cond"], ctx=r["ctx"], z0_map=T0)
    if nlox_res is not None:                                  # model-free: no plant model
        om = nlox_res["omega"]
        obs["nlox"] = KKLObserver("nlox", lat, nlox_res["dec"], ctx=RawInput(),
                                  injection=lambda z, c: om(z, c), z0_map=None)
    sc, dcfg = cfg["system"], cfg["data"]
    model = Dm.model_plant(sc)               # the MODELLER's model: nominal, full | partial
    q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
    r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(scales["y_std"])
    box = np.asarray(sc["ic_box"], float)
    p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
    for b in cfg["baselines"]:
        if b == "ekf":
            obs[b] = EKF(model, des.dt, q, r, p0)
        elif b == "ukf":
            obs[b] = UKF(model, des.dt, q, r, p0)
        elif b == "ekf_huber":
            obs[b] = HuberEKF(model, des.dt, q, r, p0)
        elif b == "ukf_huber":
            obs[b] = HuberUKF(model, des.dt, q, r, p0)
        elif b in ORACLE:
            obs[b] = BlockOracle(b, oracle_maker(cfg, des, scales, model, p0, b), dcfg["sensor"])
        elif b == "pf":
            pf = cfg["eval"].get("pf_schedule") or {"inflate": 3.0, "anneal_tau": 0.0}
            obs[b] = ParticleFilter(model, des.dt, q, r, p0,
                                    n_particles=int(cfg["eval"].get("pf_particles", 512)),
                                    inflate=float(pf["inflate"]), anneal_tau=float(pf["anneal_tau"]))
        elif b == "kkl_mb":
            obs[b] = ModelBasedKKL(lat, T0, Ti, model)
        elif b in BD.DATA_BASELINES:                          # fitted on the training data
            obs[b] = BD.make_observer(b, cfg, des, scales, q, r, p0, data_fits, oracle_maker, BlockOracle)
    return obs


def initial_guess(cfg, n, kind, seed, device) -> torch.Tensor:
    box = np.asarray(cfg["system"]["ic_box"], float)
    ctr, half = box.mean(1), (box[:, 1] - box[:, 0]) / 2
    if kind == "center":
        x0 = np.tile(ctr, (n, 1))
    elif kind == "poor":
        rng = np.random.RandomState(seed)
        x0 = ctr + 0.9 * half * rng.choice([-1.0, 1.0], size=(n, len(ctr)))
    else:
        raise ValueError(kind)
    return torch.as_tensor(x0, dtype=torch.float32, device=device)


def _blocks(cfg, des, scales, device) -> list:
    """Every (tier, condition) block: its episodes, the inputs and priors the observers
    get, the sensor it was measured with and the scorecard windows."""
    ecfg, scfg = cfg["eval"], cfg["system"]
    t_ref = float(scfg.get("t_ref", 1.0))
    base_sets, blocks = {}, []
    for tier in ecfg["tiers"]:
        for cname in ecfg["conditions"]:
            cond = cfg["conditions"][cname] or {}
            if cond.get("tiers") is not None and tier not in cond["tiers"]:
                continue
            shift = float(cond.get("plant_shift", 0.0))
            horizon = float(cond.get("horizon", 1.0))
            n_ep = cond.get("n_episodes")
            key = (tier, shift, horizon, n_ep)
            if key not in base_sets:
                kw = {} if horizon == 1.0 and n_ep is None else {"horizon": horizon, "n_episodes": n_ep}
                base_sets[key] = Dm.eval_data(cfg, tier, plant_shift=shift, device=device, **kw)
            es = base_sets[key]
            seed = Dm.split_seed(scfg["name"], tier, ecfg.get("split", "dev"),
                                 int(ecfg.get("rep", 0)))          # same stream as eval_data
            sensor = cfg["data"]["sensor"]
            if cond.get("sensor"):
                sensor = deep_merge(cfg["data"]["sensor"], cond["sensor"])
                es = es.with_sensor(des, sensor, scales["y_std"], seed + 1, t_ref=t_ref)
            u_obs = es.u
            if cond.get("input"):
                u_obs = Dm.corrupt_input(es.u, cond["input"], float(scfg.get("u_scale", 1.0)), seed + 2)
            x0 = initial_guess(cfg, len(es), cond.get("init", "center"), seed + 3, "cpu")
            N1 = es.x.shape[1]
            burst = Dm.burst_window(sensor["burst"], N1, des.dt, t_ref) if sensor.get("burst") else None
            drift = int(float(ecfg.get("settle_frac", 0.5)) * des.N) if horizon != 1.0 else None
            blocks.append({"tier": tier, "cname": cname, "es": es, "u": u_obs, "x0": x0,
                           "sensor": sensor, "shift": shift, "N1": N1, "burst": burst, "drift": drift})
    return blocks


def _run_chunks(ob, Y, U, X0, c) -> torch.Tensor:
    return torch.cat([run(ob, Y[i:i + c], U[i:i + c], X0[i:i + c]) for i in range(0, len(Y), c)])


def evaluate(cfg, des, scales, observers: dict, device, log=print, timing: dict = None) -> dict:
    """Blocks of equal episode length are concatenated into ONE batch and every observer
    runs over it once: the per-step cost is launch-bound, so this is ~blocks x faster
    than looping, and it changes nothing per episode. Oracle filters run once per distinct
    (sensor, plant shift) of the batch, chunks clipped at the end of that sub-batch.
    ``timing`` (a dict, optional) receives the wall time per observer and pass."""
    ecfg, scfg = cfg["eval"], cfg["system"]
    sigma = scales["x_std"]
    t_ref = float(scfg.get("t_ref", 1.0))
    sensors = list(build_plant(scfg).sensors)
    blocks = _blocks(cfg, des, scales, device)
    chunk = int(ecfg.get("chunk", 1024))
    out = {t: {} for t in ecfg["tiers"]}
    for b in blocks:
        out[b["tier"]][b["cname"]] = {"families": b["es"].families, "methods": {}, "scorecard": {}}
    groups = {}
    for b in blocks:
        groups.setdefault(b["N1"], []).append(b)
    timing = {} if timing is None else timing
    for N1, gb in sorted(groups.items()):
        X = torch.cat([b["es"].x for b in gb]).to(device)
        Y = torch.cat([b["es"].y for b in gb]).to(device)
        U = torch.cat([b["u"] for b in gb]).to(device)
        X0 = torch.cat([b["x0"] for b in gb]).to(device)
        settle = int(float(ecfg.get("settle_frac", 0.5)) * (N1 - 1))
        offs = np.cumsum([0] + [len(b["es"]) for b in gb])
        tag = "" if N1 - 1 == des.N else f" [x{(N1 - 1) / des.N:g} T]"
        for name, ob in observers.items():
            t0 = time.time()
            c = chunk if name != "pf" else max(1, chunk // 8)
            if isinstance(ob, BlockOracle):
                xh = torch.empty_like(X)
                sub = {}
                for i, b in enumerate(gb):
                    k = json.dumps([b["sensor"], b["shift"]], sort_keys=True, default=str)
                    sub.setdefault(k, []).append(i)
                for k, ids in sub.items():
                    f = ob.for_block(gb[ids[0]]["sensor"], gb[ids[0]]["shift"])
                    idx = torch.cat([torch.arange(int(offs[i]), int(offs[i + 1])) for i in ids]).to(device)
                    xh[idx] = _run_chunks(f, Y[idx], U[idx], X0[idx], c)
            else:
                xh = _run_chunks(ob, Y, U, X0, c)
            t_run = time.time() - t0
            m = episode_metrics(X, xh, sigma, settle, sensors)
            for i, b in enumerate(gb):
                o, n = int(offs[i]), int(offs[i + 1] - offs[i])
                blk = out[b["tier"]][b["cname"]]
                blk["methods"][name] = {k: v[o:o + n].tolist() for k, v in m.items()}
                blk["scorecard"][name] = block_scorecard(
                    X[o:o + n], xh[o:o + n], Y[o:o + n], sigma, scales["y_std"], sensors, settle,
                    des.dt, t_ref, burst=b["burst"], drift_start=b["drift"])
                if ecfg.get("save_traj") and b["tier"] in TRAJ_TIERS and b["cname"] == "clean":
                    blk.setdefault("_traj", {})[name] = xh[o:o + 4].cpu().numpy()
                    blk["_traj"]["true"] = X[o:o + 4].cpu().numpy()
            for i, b in enumerate(gb):                         # paired convergence (init blocks)
                spec = cfg["conditions"][b["cname"]] or {}
                if set(spec) != {"init"} or spec["init"] == "center":
                    continue
                ref = [j for j, r in enumerate(gb) if r["tier"] == b["tier"] and r["cname"] == "clean"]
                if ref:
                    o, n, r0 = int(offs[i]), int(offs[i + 1] - offs[i]), int(offs[ref[0]])
                    out[b["tier"]][b["cname"]]["scorecard"][name].update(
                        forget_time(xh[o:o + n], xh[r0:r0 + n], sigma, des.dt, t_ref))
            t_all = time.time() - t0
            timing.setdefault(name, []).append({"N1": int(N1), "episodes": int(len(Y)), "run_s": t_run,
                                                "metrics_s": t_all - t_run})
            summ = "  ".join(f"{b['tier'][:3]}/{b['cname']}={np.mean(out[b['tier']][b['cname']]['methods'][name]['nrmse']):.4f}"
                             for b in gb[:3])
            log(f"    eval {name:16s}{tag} {summ}  ({t_all:.0f}s, {len(Y)} episodes)")
    return out


# ---------------------------------------------------------------------------
# C2 memory probe
# ---------------------------------------------------------------------------

PROBE_CONTEXTS = {SFMEncoder: "sfm", GRUEncoder: "gru", WindowEncoder: "window"}


@torch.no_grad()
def probe_curves(ctx, decode, u, z, t_idx, Ls, sigma, perm=None) -> dict:
    """Sensitivity of the decoded state to inputs older than L, through the context path.

    For each L and each probe time t: the input of the last L samples (t-L+1 .. t) is kept,
    everything older is replaced by the input of another episode (``other``: episode
    perm[e], same indices) or by zero (``zero``); the context is recomputed on the
    truncated input, the latent z_t is NOT (this isolates the context path), and
    d(L) = mean over episodes and t of RMS_j |xhat_full - xhat_trunc|_j / sigma_j.
    u (E, N+1, nu), z (E, N+1, nz); decode(z, c) -> xhat."""
    E, _, nu = u.shape
    dev = u.device
    ti = torch.as_tensor(np.asarray(t_idx), dtype=torch.long, device=dev)
    tmax = int(ti.max())
    u = u[:, :tmax + 1]
    s = torch.as_tensor(np.asarray(sigma), dtype=u.dtype, device=dev)
    nT = len(ti)
    zt = z[:, ti].reshape(E * nT, -1)
    xf = decode(zt, ctx.seq(u)[:, ti].reshape(E * nT, -1))
    perm = torch.as_tensor((np.arange(E) + 1) % E if perm is None else perm, device=dev)
    alt = {"other": u[perm], "zero": torch.zeros_like(u)}
    k = torch.arange(tmax + 1, device=dev)
    rows, tsel = torch.arange(E * nT, device=dev), ti.repeat(E)
    out = {v: [] for v in alt}
    for L in Ls:
        keep = (k[None, :] > (ti[:, None] - int(L)))[None, :, :, None]        # (1, nT, tmax+1, 1)
        for v, ua in alt.items():
            ut = torch.where(keep, u[:, None], ua[:, None]).reshape(E * nT, tmax + 1, nu)
            ct = ctx.seq(ut)[rows, tsel]
            d = ((decode(zt, ct) - xf) / s).pow(2).mean(-1).sqrt().mean()
            out[v].append(float(d))
    return out


def probe_lengths(max_L: int = 512, n: int = 24) -> np.ndarray:
    return np.unique(np.round(np.logspace(0, np.log10(max_L), n)).astype(int))


def memory_probe(cfg, des, scales, trained: dict, device, log=print, n_episodes: int = 16,
                 n_times: int = 8, max_L: int = 512, floor: float = 1e-9) -> dict:
    """C2 probe for every conditioned decoder-side method with an sfm / gru / window
    context (eval.memory_probe). Float64 copies of the trained modules, id-tier episodes,
    n_times probe times spread over the settled window; z = the deployed latent filter's
    state z_obs (from z0 = 0; the probe holds z fixed, so its initial value is immaterial).
    Fitted decay rates (metrics.fit_decay, tail above the float floor) next to the design
    alpha and, for sfm, the context's slowest learned pole (Lemma 2)."""
    ecfg = cfg["eval"]
    tier = "id" if "id" in ecfg["tiers"] else ecfg["tiers"][0]
    es = Dm.eval_data(cfg, tier, device=device)
    n = min(int(n_episodes), len(es))
    u = es.u[:n].double().to(device)
    z = es.z_obs[:n].double().to(device)
    N1 = u.shape[1]
    settle = int(float(ecfg.get("settle_frac", 0.5)) * (N1 - 1))
    t_idx = np.unique(np.linspace(settle, N1 - 1, n_times).round().astype(int))
    Ls = probe_lengths(min(max_L, int(t_idx.min())))
    out = {}
    for name, r in trained.items():
        kind = name.split("@")[0]
        ctx = r.get("ctx")
        if ctx is None or kind in ("none", "autonomous"):
            continue
        if kind == "latent":
            out[name] = {"skipped": "latent injection: the context acts through the latent dynamics, "
                                    "and the probe holds z fixed"}
            continue
        ck = PROBE_CONTEXTS.get(type(ctx))
        if ck is None:
            out[name] = {"skipped": f"context {type(ctx).__name__} has no history"}
            continue
        t0 = time.time()
        c64, d64 = copy.deepcopy(ctx).double(), copy.deepcopy(r["cond"]).double()
        curves = probe_curves(c64, d64, u, z, t_idx, Ls, scales["x_std"])
        fits = {v: fit_decay(Ls, d, des.dt, floor) for v, d in curves.items()}
        res = {"context": ck, "tier": tier, "n_episodes": n, "t_idx": t_idx.tolist(),
               "L": Ls.tolist(), "L_s": (Ls * des.dt).tolist(), "floor": floor,
               "d_other": curves["other"], "d_zero": curves["zero"],
               "fit_other": fits["other"], "fit_zero": fits["zero"],
               "rate_other": fits["other"]["rate"], "rate_zero": fits["zero"]["rate"],
               "alpha_design": float(des.alpha)}
        res["rate_over_alpha"] = res["rate_other"] / float(des.alpha)
        if ck == "sfm":
            res["alpha_sfm"] = float(ctx.alpha())
        out[name] = res
        log(f"    memory probe {name:14s} [{ck}] rate other {res['rate_other']:.3g}/s zero "
            f"{res['rate_zero']:.3g}/s  alpha {des.alpha:.3g}"
            + (f"  sfm slowest {res['alpha_sfm']:.3g}" if ck == "sfm" else "")
            + f"  ({time.time() - t0:.1f}s)")
    return out


def costs(observers: dict, nx, ny, device) -> dict:
    """Online cost per step: microseconds at batch 1 and batch 64 (CUDA-synchronised)."""
    out = {}
    for name, ob in observers.items():
        try:
            out[name] = {"us_b1": time_per_step(ob, ny, 1, nx, device, B=1),
                         "us_b64": time_per_step(ob, ny, 1, nx, device, B=64)}
        except Exception as e:                                  # cost is diagnostic only
            out[name] = {"error": repr(e)}
    return out
