"""Subprocess runner that TRAINS HyperKKL (the paper's method) on any benchmark dataset with the research
training code, then predicts. It imports the beat_greybox package copy (the only copy of the A3 recipe:
SOAP optimiser + trainable base decoder + strided latent-window hypernetwork context, beat.py), so it runs
in a separate interpreter:

    python -m hyperkkl_bench.baselines._hkkl_train_runner train   SPEC.json
    python -m hyperkkl_bench.baselines._hkkl_train_runner predict SPEC.json IN.npz OUT.npz

SPEC.json (written by baselines.hyperkkl_train.HyperKKLTrain.fit):
    plant_module, plant_class, params, sensors, labelled   the benchmark's partial model (f_known) and label mask
    nz, alpha, dt, N, u_scale, ic_box                      latent design (Butterworth A, B) and the prior box
    arm ("A3"), seed, steps_scale, n_sim                   recipe; steps_scale < 1 for smoke runs
    train_npz, work                                        data (u, y, x with unlabelled entries replaced by the
                                                           dataset's placeholders, labels mask) and output folder

What it does (the research real-data pipeline, hyperkkl/realdata_run.py prepare + train, generalised):
  1. phase-1 data on the PARTIAL MODEL only: 400 zero-input episodes from Sobol states over the operating
     domain, the training windows' measured inputs replayed on the model from perturbed initial states, and
     32768 collocation states for the PDE residual;
  2. phase 1 (T0, T0+) with the A3 optimiser; phase 2 (lora, rank 4, 140k conditioner) on the REAL training
     windows with the loss masked to the labelled states (realdata.masked_scales), best-validation checkpoint;
  3. predict: the deployed observer z_{k+1} = latent step(z_k, y_k), z_0 = T0(x0_prior), x_hat = decoder.
Nothing in the research repository is written; outputs go to SPEC['work'].

Two options exist for plants the research pipeline never met (both off by default, so the scalar tracks are unchanged):
  nu > 1           vector input: a local copy of the RK4 of hyperkkl.data.integrate, and the context encoder built with n_u
                   inputs (the research code passes the literal 1; BG.build_context is wrapped here, the code is not edited).
  replay = "real_windows"   a closed-loop-flown unstable vehicle tumbles in an open-loop model replay, so the phase-1 trajectories
                   are the REAL training windows (states, and the latent state driven by the measured y) instead of replays
                   of the partial model. The zero-input episodes and the PDE residual stay on the partial model.
"""
import importlib
import json
import math
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[3] / "archive" / "beat_greybox_scratch" / "pkg"


def _env(spec):
    if not (PKG / "hyperkkl" / "beat.py").exists():
        raise FileNotFoundError(f"HyperKKL training code (beat_greybox package copy) not found at {PKG}")
    sys.path.insert(0, str(PKG))
    import hyperkkl
    assert Path(hyperkkl.__file__).resolve().parent == (PKG / "hyperkkl").resolve(), hyperkkl.__file__
    from hyperkkl import systems as RS
    cls = getattr(importlib.import_module(spec["plant_module"]), spec["plant_class"])
    name = "bench_" + spec["plant_class"]
    RS.REGISTRY[name] = cls
    return cls, name


def _design(spec):
    import numpy as np
    from hyperkkl import design as D
    from hyperkkl.data import Design
    ny = len(spec["sensors"])
    A, B = D.butterworth_AB(int(spec["nz"]), float(spec["alpha"]), ny)
    dt = float(spec["dt"])
    des = Design(A, B, dt, int(spec["N"]), float(spec["alpha"]), D.latent_step(A, B, dt))
    chk = D.validate(A, B, dt, float(spec["alpha"]))
    return des, chk


def _sobol(box, n, seed):
    import numpy as np
    from scipy.stats import qmc
    m = int(math.ceil(math.log2(max(n, 2))))
    X = qmc.Sobol(d=len(box), scramble=True, seed=seed).random_base2(m)[:n]
    return box[:, 0] + X * (box[:, 1] - box[:, 0])


def _integrate_vec(plant, des, x0, U, device):
    """hyperkkl.data.integrate for vector inputs: RK4 of (x, z) under a zero-order hold, U (E, N+1, n_u)."""
    import numpy as np
    import torch
    dev = torch.device(device)
    A = torch.as_tensor(des.A, dtype=torch.float64, device=dev)
    B = torch.as_tensor(des.B, dtype=torch.float64, device=dev)
    ug = torch.as_tensor(np.asarray(U), dtype=torch.float64, device=dev)
    x = torch.as_tensor(np.asarray(x0), dtype=torch.float64, device=dev)
    z = torch.zeros(x.shape[0], A.shape[0], dtype=torch.float64, device=dev)
    h = des.dt
    rhs = lambda uk, xx, zz: (plant.f(xx, uk), zz @ A.T + plant.h(xx) @ B.T)
    xs, zs = [x], [z]
    for k in range(des.N):
        uk = ug[:, k]
        a1, b1 = rhs(uk, x, z)
        a2, b2 = rhs(uk, x + h / 2 * a1, z + h / 2 * b1)
        a3, b3 = rhs(uk, x + h / 2 * a2, z + h / 2 * b2)
        a4, b4 = rhs(uk, x + h * a3, z + h * b3)
        x = x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        z = z + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        xs.append(x)
        zs.append(z)
    return torch.stack(xs, 1), torch.stack(zs, 1), ug


def train(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    cls, name = _env(spec)
    import numpy as np
    import torch
    from hyperkkl import beat as BG
    from hyperkkl import design as D
    from hyperkkl import realdata as R
    from hyperkkl.config import deep_merge, load
    from hyperkkl.data import EpisodeSet, integrate, run_latent
    from hyperkkl.train import phase1
    torch.set_num_threads(4)
    work = Path(spec["work"]); work.mkdir(parents=True, exist_ok=True)
    logf = open(work / "train.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    device = spec.get("device", "cpu")
    seed = int(spec["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    des, chk = _design(spec)
    log(f"design nz {spec['nz']} alpha {spec['alpha']} dt {des.dt:.5g} N {des.N} gate {chk['ok']} "
        f"(max|lam|dt {chk['max_lam_dt']:.4f})")
    if not chk["ok"]:
        raise RuntimeError(f"latent design fails the design gate: {chk}")
    plant = cls(sensors=tuple(spec["sensors"]), **spec["params"])
    nx = plant.nx
    nu = int(spec.get("nu", 1))
    replay_real = spec.get("replay", "model") == "real_windows"
    if nu > 1:
        orig_bc = BG.build_context
        BG.build_context = lambda kind, ccfg, _nu, dt, poles, u_ref: orig_bc(kind, ccfg, nu, dt, poles, u_ref)
    d = np.load(spec["train_npz"])
    box = np.asarray(spec["ic_box"], float)

    # 1. phase-1 data on the partial model -------------------------------------------------------
    rng = np.random.RandomState(seed)
    Xtr, Utr = d["train_x"].astype(np.float64), d["train_u"].astype(np.float64)
    E, N1 = Utr.shape[:2]
    N = N1 - 1
    assert N == des.N, (N, des.N)
    xs_data = np.nanstd(Xtr.reshape(-1, nx), 0)
    pick = rng.randint(0, E, size=int(spec.get("n_sim", 600)))
    X0 = Xtr[pick, 0] + rng.uniform(-0.5, 0.5, (len(pick), nx)) * xs_data
    X0 = np.clip(X0, box[:, 0], box[:, 1])
    if replay_real:                            # real windows, not a model replay
        keep = rng.permutation(E)[:int(spec.get("n_sim", 600))]
        xs = torch.as_tensor(Xtr[keep], dtype=torch.float64)
        us = torch.as_tensor(Utr[keep], dtype=torch.float64)
        zs = run_latent(des, torch.as_tensor(d["train_y"][keep], dtype=torch.float64).to(device)).cpu()
    elif nu > 1:
        xs, zs, us = _integrate_vec(plant, des, X0, Utr[pick], device)
    else:
        sigs = [lambda t, u=Utr[j].copy(): u[np.clip(np.round(t / des.dt).astype(int), 0, N)] for j in pick]
        xs, zs, us = integrate(plant, des, X0, sigs, device=device)
    xs, zs, us = xs.cpu(), zs.cpu(), us.cpu()
    xf = xs.reshape(-1, nx).numpy()
    ok = np.isfinite(xf).all(1)
    lo, hi = xf[ok].min(0), xf[ok].max(0)
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    dom = np.stack([np.minimum(ctr - 1.15 * half, box[:, 0]), np.maximum(ctr + 1.15 * half, box[:, 1])], 1)
    kappa, alpha = D.spectral_bounds(des.A)
    y_bar = float(plant.h(xs.reshape(-1, nx)).norm(dim=-1).max())
    z_scale = float(zs[:, zs.shape[1] // 2:].pow(2).sum(-1).mean().sqrt())
    Bn = float(np.linalg.norm(des.B, 2))
    t_burn = D.washout_time(kappa, alpha, Bn, y_bar, z_scale, 1e-3)
    burn = min(int(math.ceil(t_burn / des.dt)), N // 2)
    xr = xs[:, burn:].reshape(-1, nx)
    lab = list(spec["labelled"])
    x_std = xr.std(0).clamp(min=1e-9).numpy()
    if spec.get("x_std"):                      # the dataset's scorer scales for the labelled states
        x_std = np.asarray(spec["x_std"], float)
    scales = {"domain": dom.tolist(), "y_bar": y_bar, "z_scale": z_scale, "t_burn": t_burn,
              "memory_s": D.memory_length(kappa, alpha, Bn, y_bar, z_scale, 1e-3),
              "x_std": x_std.tolist(), "y_std": plant.h(xr).std(0).tolist(), "x_mean": xr.mean(0).tolist(),
              "dt": des.dt, "N": des.N, "design_check": chk}

    def es_sim(x, z, u):
        y = plant.h(x.reshape(-1, nx)).reshape(x.shape[0], x.shape[1], -1)
        zo = run_latent(des, y)
        f = lambda a: a.float()
        return EpisodeSet(f(x), f(u), f(y), f(y), f(z), f(zo), ["replay"] * len(x), burn, des.dt, [], 0)

    train_sim = es_sim(xs, zs, us)
    XP = _sobol(dom, 400, 101)
    if nu > 1:
        xp, zp, up = _integrate_vec(plant, des, XP, np.zeros((len(XP), N1, nu)), device)
        fin = torch.isfinite(xp).all(-1).all(-1) & (xp.abs().amax((1, 2)) < 1e3)       # tumbled episodes are dropped
        log(f"zero-input episodes kept: {int(fin.sum())} / {len(fin)}")
        xp, zp, up = xp[fin], zp[fin], up[fin]
    else:
        xp, zp, up = integrate(plant, des, XP, [lambda t: np.zeros_like(t)] * len(XP), device=device)
    p1 = es_sim(xp.cpu(), zp.cpu(), up.cpu())
    Xc = torch.as_tensor(_sobol(dom, 32768, 104), dtype=torch.float64)
    coll = {"x": Xc.float(), "f0": plant.f(Xc, torch.zeros(len(Xc), nu, dtype=Xc.dtype)).float(), "y": plant.h(Xc).float()}
    log(f"phase-1 data: {len(p1)} zero-input, {len(train_sim)} replayed, {len(Xc)} collocation; burn {burn} "
        f"({time.time() - t0:.0f}s)")

    def real(prefix):
        x = torch.as_tensor(d[f"{prefix}_x"], dtype=torch.float32)
        u = torch.as_tensor(d[f"{prefix}_u"], dtype=torch.float32)
        u = u[..., None] if u.ndim == 2 else u
        y = torch.as_tensor(d[f"{prefix}_y"], dtype=torch.float32)
        zo = run_latent(des, y.double()).float()
        return EpisodeSet(x, u, y.clone(), y, zo.clone(), zo, ["real"] * len(x), burn, des.dt, [], 0)

    data2 = {"train": real("train"), "val": real("val")}

    # 2. phase 1 + phase 2 (A3) --------------------------------------------------------------------
    scfg = {"name": name, "plant": name, "role": "real", "params": spec["params"], "sensors": list(spec["sensors"]),
            "ic_box": box.tolist(), "T": des.N * des.dt, "nz": int(spec["nz"]), "alpha": float(spec["alpha"]),
            "dt": des.dt, "u_scale": float(spec["u_scale"]), "t_ref": 1, "u_bounds": [],
            "actuator": {"kind": "affine"}, "mismatch": 0.0, "knowledge": "full"}
    cfg = load("battery")
    cfg["system"] = scfg
    cfg["baselines"] = []
    cfg["methods"] = ["lora"]
    s = float(spec.get("steps_scale", 1.0))
    arm = spec.get("arm", "A3")
    assert arm == "A3", "only the A3 recipe is wired (S + P + H)"
    p1o = {"optimizer": "soap", "lr": 1e-2,
           "steps_enc": max(50, int(cfg["phase1"]["steps_enc"] * s)), "steps_dec": max(50, int(cfg["phase1"]["steps_dec"] * s))}
    p2o = {"optimizer": "soap", "lr": 1e-2, "base_lr_factor": 0.1, "window": 20,
           "stride": max(1, int(round(6.0 / (des.alpha * des.dt) / 20.0))), "steps": max(50, int(cfg["phase2"]["steps"] * s))}
    cfg1 = deep_merge(cfg, {"phase1": p1o, "phase2": {"val_every": max(10, min(int(cfg["phase2"].get("val_every", 100)), p2o["steps"] // 5))}})
    base = phase1(cfg1, {"p1": p1, "train": train_sim, "coll": coll}, scales, des, device, seed, log)
    log(f"phase 1 done ({time.time() - t0:.0f}s): " + json.dumps({k: round(float(v), 5) for k, v in base.get("diag", {}).items()}))
    ms = R.masked_scales(scales, lab)
    r = BG.phase2_bg("lora", cfg1, data2, ms, des, base, device, seed, p2o, log)
    log(f"phase 2 done: val {r.get('val')} ({time.time() - t0:.0f}s)")
    cpu = lambda o: o.cpu() if isinstance(o, torch.nn.Module) else o
    torch.save({"base": {k: cpu(v) for k, v in base.items()}, "r": {k: cpu(v) for k, v in r.items()}, "spec": spec,
                "p1": p1o, "p2": p2o, "scales": scales, "wall_s": time.time() - t0}, work / "model.pt")
    npar = int(r.get("trainable_total") or r.get("n_params") or 0)
    (work / "train_info.json").write_text(json.dumps({"wall_s": time.time() - t0, "val": float(r.get("val", float("nan"))),
                                                      "n_params_conditioner": npar, "p1": p1o, "p2": p2o,
                                                      "design_check_ok": bool(chk["ok"]), "burn": burn}, indent=1))
    log("saved")


def predict(spec_path, fin, fout):
    spec = json.loads(Path(spec_path).read_text())
    _env(spec)
    import numpy as np
    import torch
    from hyperkkl import beat as BG
    from hyperkkl.observers import run
    torch.set_num_threads(4)
    device = spec.get("device", "cpu")
    des, _ = _design(spec)
    m = torch.load(Path(spec["work"]) / "model.pt", weights_only=False, map_location="cpu")
    to = lambda o: o.to(device) if isinstance(o, torch.nn.Module) else o
    base = {k: to(v) for k, v in m["base"].items()}
    r = {k: to(v) for k, v in m["r"].items()}
    ob = BG.make_observer("A3", des, base, r, device)
    d = np.load(fin)
    Y = torch.as_tensor(d["Y"], dtype=torch.float32)
    U = torch.as_tensor(d["U"], dtype=torch.float32)
    U = U[..., None] if U.ndim == 2 else U
    X0 = torch.as_tensor(d["X0"], dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Y), 256):
            out.append(run(ob, Y[i:i + 256].to(device), U[i:i + 256].to(device), X0[i:i + 256].to(device)).double().cpu())
    npar = sum(p.numel() for v in m["r"].values() if isinstance(v, torch.nn.Module) for p in v.parameters())
    npar += sum(p.numel() for v in m["base"].values() if isinstance(v, torch.nn.Module) for p in v.parameters())
    np.savez(fout, X_hat=torch.cat(out).numpy(), n_params=npar)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    else:
        predict(sys.argv[2], sys.argv[3], sys.argv[4])
