"""Stage 1 of the uncertainty study (experiments/uncertainty/PREREG.md): run ONE finished run's
cached lora observer and the reference filters on the fit / calibration / test sets and store, on
the settled window at a stride, everything the bands need (truth, estimates, online features,
filter covariances, NEES). Read-only: models are loaded from experiments/cache/models and a miss
raises; DATA misses (fresh calibration episodes, drifted-plant episodes) are simulated in memory
and never written. The package is imported from the run's own code snapshot (SNAP argument).

usage: collect.py NODE SYSTEM SEED DEVICE OUT.npz SNAPDIR
"""
import copy, json, math, os, sys, time
from pathlib import Path

node, system, seed, device, out, snap = (sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4],
                                         Path(sys.argv[5]), Path(sys.argv[6]))
sys.path.insert(0, str(snap))
import numpy as np
import torch
from torch.func import jacrev, vmap

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
from hyperkkl import data as Dm, cache as C            # noqa: E402
from hyperkkl import systems_extra  # noqa: F401,E402
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.config import digest, deep_merge         # noqa: E402
from hyperkkl.job import method_cfg, _to               # noqa: E402
from hyperkkl import evaluate as Ev                     # noqa: E402
from hyperkkl.observers import EKF, UKF, KKLObserver, LatentStep   # noqa: E402

assert Path(Dm.__file__).resolve().is_relative_to(snap.resolve()), Dm.__file__
TREE, MC = ROOT / "experiments" / "tree", ROOT / "experiments" / "cache" / "models"
MISSES = []
SCRATCH_DATA = out.parent.parent / "datacache"


def ro_data(path, build=None, **kw):
    path = Path(path)
    if path.exists():
        return torch.load(path, weights_only=False, map_location="cpu")
    if build is None:
        raise FileNotFoundError(path)
    MISSES.append(path.name)                 # simulated; never written to experiments/cache
    sc_ = SCRATCH_DATA / path.name           # shared across seeds in the scratchpad only (same key = same data)
    if sc_.exists():
        return torch.load(sc_, weights_only=False, map_location="cpu")
    obj = build()
    SCRATCH_DATA.mkdir(parents=True, exist_ok=True)
    tmp = sc_.with_suffix(f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, sc_)
    return obj


def ro_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model cache miss (read-only): {path.name}")
    return torch.load(path, weights_only=False, map_location="cpu")


Dm.CACHE = ROOT / "experiments" / "cache"
Dm._cached = ro_data
C.cached = ro_data
_ST = {}
Dm.system_scales = lambda cfg, device="cpu": _ST["scales"]
Ev.Dm.system_scales = Dm.system_scales

torch.set_num_threads(4)
torch.manual_seed(0)
t0 = time.time()
dev = torch.device(device)
payload = json.loads((TREE / node / "runs" / system / f"seed{seed}" / "results.json").read_text())
assert payload["code"] in str(snap), (payload["code"], snap)
cfg = copy.deepcopy(payload["config"])
scales = payload["scales"]
_ST["scales"] = scales
des = Dm.make_design(cfg["system"])
data = Dm.training_data(cfg, seed, device)
mh = module_hash(MODEL_MODULES)
k1 = digest(cfg, "system", "data", "phase1", extra=[seed, mh])
base = _to(ro_model(MC / f"{system}_p1_{k1}.pt"), device)
kind, mc = method_cfg(cfg, "lora")
k2 = digest(mc, "system", "data", "phase1", "phase2", extra=[seed, kind, mh])
lora = _to(ro_model(MC / f"{system}_p2_{kind}_{k2}.pt"), device)
print(f"loaded {node} {system} s{seed} code {payload['code']} ({time.time() - t0:.0f}s)", flush=True)

sc = cfg["system"]
sensors = list(sc["sensors"])
nx, ny = len(scales["x_std"]), len(sensors)
sig = torch.tensor(scales["x_std"], device=dev, dtype=torch.float32)
ystd = torch.tensor(scales["y_std"], device=dev, dtype=torch.float32)
alpha_kkl = float(des.alpha)
lam_ema = math.exp(-alpha_kkl * des.dt)
lam_gram = math.exp(-2 * alpha_kkl * des.dt)

# ---------------- evaluation blocks (same episodes / sensors / plants as the runs) ------------
CONDS = ["clean", "outliers", "noise10", "plant_drift10"]
assert all(c in cfg["conditions"] for c in CONDS), list(cfg["conditions"])
cfg["eval"]["tiers"] = ["id", "shape_ood", "sequence_ood"]
cfg["eval"]["conditions"] = CONDS
DROPPED = []
try:
    blocks = Ev._blocks(cfg, des, scales, device)
except RuntimeError as ex:                  # e.g. every drifted-plant episode diverges (rejected)
    print(f"  blocks failed ({ex!r}); dropping plant_drift10", flush=True)
    DROPPED.append(["plant_drift10", repr(ex)])
    CONDS = [c for c in CONDS if c != "plant_drift10"]
    cfg["eval"]["conditions"] = CONDS
    blocks = Ev._blocks(cfg, des, scales, device)

# fresh calibration episodes (id distribution, rep 1, 128 episodes) and the fit set (phase-2 val)
cal_cfg = copy.deepcopy(cfg)
cal_cfg["eval"]["rep"] = 1
cal_es = Dm.eval_data(cal_cfg, "id", device=device, n_episodes=128)
fit_es = data["val"]
seed_cal = Dm.split_seed(sc["name"], "id", cfg["eval"].get("split", "dev"), 1)
x0_cal = Ev.initial_guess(cfg, len(cal_es), "center", seed_cal + 3, "cpu")
x0_fit = Ev.initial_guess(cfg, len(fit_es), "center", 12345, "cpu")
sets = [{"name": "fit", "es": fit_es, "u": fit_es.u, "x0": x0_fit, "sensor": cfg["data"]["sensor"], "shift": 0.0},
        {"name": "cal", "es": cal_es, "u": cal_es.u, "x0": x0_cal, "sensor": cfg["data"]["sensor"], "shift": 0.0}]
for b in blocks:
    sets.append({"name": f"{b['tier']}|{b['cname']}", **{k: b[k] for k in ("es", "u", "x0", "sensor", "shift")}})
N1 = sets[0]["es"].x.shape[1]
assert all(s["es"].x.shape[1] == N1 for s in sets)
settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (N1 - 1))
stride = max(1, (N1 - settle) // 1000)
keep = np.arange(settle, N1, stride)
keep_t = torch.as_tensor(keep, device=dev)
print(f"sets ready: {[(s['name'], len(s['es'])) for s in sets]}  N1 {N1} settle {settle} stride {stride} "
      f"data misses {MISSES} ({time.time() - t0:.0f}s)", flush=True)

model = Dm.model_plant(sc)                        # the modeller's model (full | partial)
q = float(cfg["data"].get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
r = max(float(cfg["data"]["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(scales["y_std"])
box = np.asarray(sc["ic_box"], float)
p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
Hsel = torch.zeros(ny, nx, device=dev)
for i, s in enumerate(sensors):
    Hsel[i, s] = 1.0
Rinv = torch.diag(torch.as_tensor(1.0 / r ** 2, dtype=torch.float32, device=dev))
HtRH = Hsel.T @ Rinv @ Hsel
P0 = torch.diag(torch.as_tensor(p0 ** 2, dtype=torch.float32, device=dev))
Pcap = torch.as_tensor(100.0 * p0 ** 2, dtype=torch.float32, device=dev)
Rm = torch.diag(torch.as_tensor(r ** 2, dtype=torch.float32, device=dev))
Qp = torch.diag(torch.as_tensor(np.maximum(q, 1e-6 * np.asarray(scales["x_std"])) ** 2 * des.dt, dtype=torch.float32, device=dev))
f1 = lambda xi, ui: model.f(xi[None], ui[None])[0]
jac_f = vmap(jacrev(f1, argnums=0))

# training contexts for the context-distance feature
ctx, dec = lora["ctx"], lora["cond"]
with torch.no_grad():
    tr = data["train"]
    ctr = torch.cat([ctx.seq(tr.u[i:i + 32].to(dev).float()) for i in range(0, len(tr), 32)])
    ctr = ctr[:, tr.burn:].reshape(-1, ctr.shape[-1])
    g = torch.Generator().manual_seed(seed)
    ctr = ctr[torch.randperm(len(ctr), generator=g)[:20000].to(dev)]
    c_mu, c_sd = ctr.mean(0), ctr.std(0).clamp(min=1e-6)
    ctr_n = (ctr - c_mu) / c_sd
    zo = tr.z_obs[:, tr.burn:].reshape(-1, tr.z_obs.shape[-1]).to(dev).float()
    z_std = zo.std(0).clamp(min=1e-6)


@torch.no_grad()
def ctx_dist(c):                                    # (M, dc) -> (M,)
    out = []
    for i in range(0, len(c), 4096):
        d = torch.cdist((c[i:i + 4096] - c_mu) / c_sd, ctr_n)
        out.append(d.topk(10, largest=False).values.mean(1))
    return torch.cat(out)


def dec_jac_rows(z, c):                              # per-state row norm of d(xhat/sig)/d(z/z_std)
    out = []
    for i in range(0, len(z), 8192):
        zz, cc = z[i:i + 8192], c[i:i + 8192]
        fz = lambda zi, ci: dec(zi[None] * z_std, ci[None])[0] / sig
        J = vmap(jacrev(fz, argnums=0))(zz / z_std, cc)          # (M, nx, nz)
        out.append(J.norm(dim=-1).detach())
    return torch.cat(out)


def cat(Ss, key):
    if key == "x":
        return torch.cat([S["es"].x.to(dev).float() for S in Ss])
    if key == "y":
        return torch.cat([S["es"].y.to(dev).float() for S in Ss])
    return torch.cat([S[key].to(dev).float() for S in Ss])


@torch.no_grad()
def run_kkl(Y, U, X0):
    B = len(Y)
    ob = KKLObserver("lora", LatentStep(des, dev), dec, ctx=ctx, z0_map=base["T0"])
    ob.reset(B, X0, U[:, 0])
    rema = torch.zeros(B, ny, device=dev)
    P = P0.expand(B, nx, nx).clone()
    eye = torch.eye(nx, device=dev)
    rec = {k: [] for k in ("xh", "res", "rema", "gram", "z", "c")}
    for k in range(N1):
        if k > 0:
            ob.advance(Y[:, k - 1], U[:, k - 1], Y[:, k], U[:, k])
        xh = ob.estimate()
        res = (Y[:, k] - xh[:, sensors]) / ystd
        rema = lam_ema * rema + (1 - lam_ema) * res * res if k > 0 else res * res
        # local observability: covariance (Riccati) recursion of the modeller's model linearised along
        # xhat, with forgetting lam (P / lam = information I scaled by lam) and the run's process noise
        if k > 0:
            with torch.enable_grad():
                F = jac_f(xh_prev, U[:, k - 1])
            F = torch.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
            Phi = torch.linalg.matrix_exp(F * des.dt)
            P = Phi @ P @ Phi.transpose(1, 2) / lam_gram + Qp
        S_ = Hsel @ P @ Hsel.T + Rm
        K = P @ Hsel.T @ torch.linalg.inv(S_)
        A_ = eye - K @ Hsel
        P = A_ @ P @ A_.transpose(1, 2) + K @ Rm @ K.transpose(1, 2)
        P = 0.5 * (P + P.transpose(1, 2))
        d = torch.diagonal(P, dim1=1, dim2=2).clamp(min=1e-30)
        sc_ = torch.clamp(torch.sqrt(Pcap / d), max=1.0)      # cap each variance at 100 x prior
        P = sc_[:, :, None] * P * sc_[:, None, :]
        bad = ~torch.isfinite(P).all(-1).all(-1)
        if bad.any():
            P = torch.where(bad[:, None, None], P0.expand_as(P), P)
        xh_prev = xh
        if k >= settle and (k - settle) % stride == 0:
            cov = P
            rec["xh"].append(xh); rec["res"].append(res); rec["rema"].append(rema.sqrt())
            rec["gram"].append(torch.diagonal(cov, dim1=1, dim2=2).clamp(min=0).sqrt() / sig)
            rec["z"].append(ob.z.clone()); rec["c"].append(ob.c.clone())
    out = {k: torch.stack(v, 1) for k, v in rec.items()}
    T = out["z"].shape[1]
    zf, cf = out.pop("z").reshape(B * T, -1), out.pop("c").reshape(B * T, -1)
    out["ctxd"] = ctx_dist(cf).reshape(B, T)
    with torch.enable_grad():
        out["jac"] = dec_jac_rows(zf, cf).reshape(B, T, nx)
    return out


@torch.no_grad()
def run_filter(f, Y, U, X0, X):
    f.reset(len(Y), X0, U[:, 0])
    eye = torch.eye(nx, device=dev)
    rec = {"xh": [], "pd": [], "nees": []}
    for k in range(N1):
        if k > 0:
            f.advance(Y[:, k - 1], U[:, k - 1], Y[:, k], U[:, k])
        if k >= settle and (k - settle) % stride == 0:
            xh = f.estimate()
            P = 0.5 * (f.P + f.P.transpose(1, 2))
            e = (X[:, k] - xh)
            nees = (e[:, None, :] @ torch.linalg.solve(P + 1e-12 * eye, e[:, :, None]))[:, 0, 0]
            rec["xh"].append(torch.nan_to_num(xh, nan=1e6, posinf=1e6, neginf=-1e6))
            rec["pd"].append(torch.nan_to_num(torch.diagonal(P, dim1=1, dim2=2).clamp(min=0).sqrt(), nan=1e6))
            rec["nees"].append(torch.nan_to_num(nees, nan=1e12, posinf=1e12, neginf=1e12))
    return {k: torch.stack(v, 1) for k, v in rec.items()}


def split_store(Ss, outd, prefix):
    o = 0
    for S in Ss:
        n = len(S["es"])
        for k, v in outd.items():
            res[f"{S['name']}/{prefix}_{k}"] = v[o:o + n]
        o += n


oracle_make = Ev.oracle_maker(cfg, des, scales, model, p0, "ekf_oracle")
res = {}
for S in sets:
    res[f"{S['name']}/x"] = S["es"].x.to(dev).float()[:, keep_t]
t1 = time.time()
split_store(sets, run_kkl(cat(sets, "y"), cat(sets, "u"), cat(sets, "x0")), "lora")
print(f"  lora done ({time.time() - t1:.0f}s)", flush=True)
fs = sets[1:]
Yf, Uf, X0f, Xf = cat(fs, "y"), cat(fs, "u"), cat(fs, "x0"), cat(fs, "x")
for fname, f in (("ekf", EKF(model, des.dt, q, r, p0)), ("ukf", UKF(model, des.dt, q, r, p0))):
    t1 = time.time()
    try:
        split_store(fs, run_filter(f, Yf, Uf, X0f, Xf), fname)
    except Exception as ex:                       # a filter blow-up is recorded, not fatal
        print(f"  filter {fname} failed: {ex!r}", flush=True)
    print(f"  {fname} done ({time.time() - t1:.0f}s)", flush=True)
# ekf_oracle: one batched pass, each episode with the Q, R its condition's oracle filter would use
fo = []
for S in fs:
    g_ = oracle_make(dict(S["sensor"] or {}), float(S["shift"]))
    fo.append((g_, len(S["es"])))
t1 = time.time()
f = fo[0][0]
_reset = f.reset


def _batched_reset(B, x0_hat, u0):
    _reset(B, x0_hat, u0)
    f.Q = torch.cat([torch.diag(torch.as_tensor(g_.q2, dtype=x0_hat.dtype, device=x0_hat.device).clamp(min=1e-12))
                     .expand(n_, nx, nx) for g_, n_ in fo])
    f.R = torch.cat([torch.diag(torch.as_tensor(g_.r2, dtype=x0_hat.dtype, device=x0_hat.device).clamp(min=1e-12))
                     .expand(n_, ny, ny) for g_, n_ in fo])
    f.p02 = fo[0][0].p02


assert all(not getattr(g_, "hold_missing", False) and getattr(g_, "_plain", True) for g_, _ in fo)
assert all(np.allclose(g_.p02, f.p02) for g_, _ in fo)
f.reset = _batched_reset
try:
    split_store(fs, run_filter(f, Yf, Uf, X0f, Xf), "ekf_oracle")
except Exception as ex:
    print(f"  filter ekf_oracle failed: {ex!r}", flush=True)
print(f"  ekf_oracle done ({time.time() - t1:.0f}s)", flush=True)

meta = {"node": node, "system": system, "seed": seed, "code": payload["code"], "sensors": sensors,
        "x_std": scales["x_std"], "y_std": scales["y_std"], "nx": nx, "ny": ny, "N1": N1, "settle": settle,
        "stride": stride, "dt": des.dt, "alpha_kkl": alpha_kkl, "sets": [S["name"] for S in sets],
        "data_misses": MISSES, "dropped_conditions": DROPPED, "knowledge": sc.get("knowledge", "full"), "mismatch": sc.get("mismatch", 0.0),
        "sensor": cfg["data"]["sensor"], "wall_s": time.time() - t0,
        "stored_nrmse_lora": {t: float(np.mean(payload["results"][t]["clean"]["methods"]["lora"]["nrmse"]))
                              for t in ("id", "shape_ood", "sequence_ood")},
        "stored_nrmse_ekf": {t: float(np.mean(payload["results"][t]["clean"]["methods"]["ekf"]["nrmse"]))
                             for t in ("id", "shape_ood", "sequence_ood") if "ekf" in payload["results"][t]["clean"]["methods"]}}
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, meta=json.dumps(meta), **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in res.items()})
print("done", out, f"{time.time() - t0:.0f}s", flush=True)
