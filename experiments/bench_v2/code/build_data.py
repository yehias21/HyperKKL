"""Build the frozen v2 episodes of one cell (DESIGN.md sections 1, 4).

    CUDA_VISIBLE_DEVICES=2 python build_data.py driven battery realistic
    CUDA_VISIBLE_DEVICES=2 python build_data.py autonomous vdp white

Output data/<family>/<plant>/<variant>/<split>.pt (research EpisodeSet; meta has the IC, signal description and, for
profile episodes, operator + profile seed so the command can be regenerated), scales.json, MANIFEST.json (sha256 of
x/u/y per split, counts, rejections). Hidden splits: labels and the seed secret go to hidden_labels/ (private).
Refuses to overwrite an existing split (frozen).

family B (driven) splits:  train (600 operating-profile episodes, training operators, k_cross 3), train_event (600,
  eventmix D1), val / val_event (64 each), val_ood (64, v1 val-OOD generators), phase1 (400 zero input over the
  envelope), coll (collocation), test_op_id / test_op_ood (64), hidden_op_id / hidden_op_ood (32), and
  test_legacy_{id,shape_ood,sequence_ood} (64: the v1 export where it exists, else regenerated with the v1 tier
  generators and the v1 split seeds rule).
family A (autonomous) splits:  train (600, u = 0, ICs in the core box), train_event (600, u = 0 + state kicks),
  val / val_event (64), val_ood (64, IC shell 0.6-0.8), phase1, coll, test_id (64, core box), test_ic_ood (64, IC
  shell 0.8-1.0), test_param_shift (64, plant parameters exp(+-0.15) further off), hidden_id / hidden_ic_ood (32).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import env
import numpy as np
import torch

import v2cfg

N_TRAIN, N_VAL, N_TEST, N_HID, N_P1 = 600, 64, 64, 32, 400
CORE, SHELL_VAL, SHELL_TEST = 0.6, (0.6, 0.8), (0.8, 1.0)
P_KICK, KICK = 0.45, (0.2, 1.0)
SEG = (3.0, 12.0)


def sha(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()


def setup_cache():
    from hyperkkl import data as Dm
    Dm.CACHE = env.HERE / "cache"
    (Dm.CACHE / "scales").mkdir(parents=True, exist_ok=True)
    live = env.LIVE / "experiments" / "cache" / "scales"
    for p in live.glob("*.json"):
        q = Dm.CACHE / "scales" / p.name
        if not q.exists():
            q.write_bytes(p.read_bytes())
    return Dm


class _Fixed:
    """Drop-in for data._signals returning a precomputed list (cycled)."""
    def __init__(self, sigs):
        self.sigs = sigs

    def __call__(self, families, n, regime, rng, des, scfg, n_strata=4):
        return [self.sigs[i % len(self.sigs)] for i in range(n)]


def gen(Dm, plant, des, sc, scfg, dcfg, sigs, n, seed, dev, k=1, ic_box=None, signals_fn=None, families=("x",),
        regime="train"):
    orig = Dm._signals
    Dm._signals = signals_fn if signals_fn is not None else _Fixed(sigs)
    try:
        return Dm.generate(plant, des, sc, scfg, dict(dcfg, k_cross=k), list(families), n, regime, seed, dev,
                           ic_box=ic_box)
    finally:
        Dm._signals = orig


def profile_sigs(plant, split, n, seed0, des):
    import profiles_v2 as PV
    out = []
    for i in range(n):
        pr = PV.generate(plant, "train" if split == "train" else split, seed0 + i, T=des.T, dt=des.dt)
        s = pr.signal()
        s.params.update(operator=pr.operator, profile_seed=seed0 + i, profile_split=split)
        out.append(s)
    return out


def shell_box(box, lo, hi, n, seed):
    """n points with normalised inf-norm |x - c| / half in [lo, hi] (uniform rejection sampling)."""
    box = np.asarray(box, float)
    c, h = box.mean(1), (box[:, 1] - box[:, 0]) / 2
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        v = rng.uniform(-1, 1, size=(4096, len(c)))
        r = np.abs(v).max(1)
        v = v[(r >= lo) & (r <= hi)]
        out.extend(list(c + v * h))
    return np.asarray(out[:n])


def core_box(box, f=CORE):
    box = np.asarray(box, float)
    c, h = box.mean(1), (box[:, 1] - box[:, 0]) / 2
    return np.stack([c - f * h, c + f * h], 1)


@torch.no_grad()
def kick_episodes(Dm, plant, des, sc, scfg, dcfg, n, seed, dev, box):
    """u = 0 episodes with impulsive state kicks (the autonomous event-structured recipe)."""
    from hyperkkl import data as D
    rng = np.random.RandomState(seed)
    t_ref = float(scfg.get("t_ref", 1.0))
    N1 = des.N + 1
    xs = np.asarray(sc["x_std"], float)
    dom = np.asarray(sc["domain"], float)
    kick = np.zeros((n, N1, len(xs)))
    nk = 0
    for e in range(n):
        t = 0.0
        while True:
            t += math.exp(rng.uniform(math.log(SEG[0]), math.log(SEG[1]))) * t_ref
            k = int(round(t / des.dt))
            if k >= N1 - 1:
                break
            if rng.rand() < P_KICK:
                d = rng.randn(len(xs)); d /= np.linalg.norm(d)
                kick[e, k] = d * rng.uniform(*KICK) * xs
                nk += 1
    x0 = D._sobol(box, n, seed)
    sig_p = float(dcfg.get("noise_proc", 0.0)) * xs * math.sqrt(des.dt)
    x = torch.as_tensor(x0, dtype=torch.float64, device=dev)
    u = torch.zeros(n, 1, dtype=torch.float64, device=dev)
    K = torch.as_tensor(kick, dtype=torch.float64, device=dev)
    lo = torch.as_tensor(dom[:, 0], dtype=torch.float64, device=dev)
    hi = torch.as_tensor(dom[:, 1], dtype=torch.float64, device=dev)
    sp = torch.as_tensor(sig_p, dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    h = des.dt
    out = [x]
    for k in range(des.N):
        a1 = plant.f(x, u); a2 = plant.f(x + h / 2 * a1, u); a3 = plant.f(x + h / 2 * a2, u); a4 = plant.f(x + h * a3, u)
        x = x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4) + sp * torch.randn(x.shape, generator=g, dtype=x.dtype, device=dev)
        x = torch.minimum(torch.maximum(x + K[:, k + 1], lo), hi)
        out.append(x)
    X = torch.stack(out, 1)
    ok = torch.isfinite(X).all(2).all(1)
    X = X[ok]
    y_clean = torch.stack([plant.h(xi) for xi in X])
    y = Dm.measure(y_clean, dcfg.get("sensor", {}), sc["y_std"], seed, dt=des.dt, t_ref=t_ref)
    z = Dm.run_latent(des, y_clean)
    zo = Dm.run_latent(des, y)
    burn = int(math.ceil(sc["t_burn"] / des.dt))
    U = torch.zeros(X.shape[0], N1, 1, dtype=torch.float64)
    f32 = lambda a: a.float().cpu()
    meta = [{"ic": x0[i].tolist(), "family": "kicks", "n_kicks": int((kick[i] != 0).any(-1).sum())}
            for i in range(n) if bool(ok[i])]
    return Dm.EpisodeSet(f32(X), f32(U), f32(y_clean), f32(y), f32(z), f32(zo), ["kicks"] * len(meta), burn,
                         des.dt, meta, int((~ok).sum()))


def legacy_from_export(Dm, plant, variant, tier, des, cfg, sc):
    """The v1 frozen test tier from the export parquet (u, y, x), as an EpisodeSet (z from the design)."""
    import pyarrow.parquet as pq
    var = {"white": "white", "realistic": "realistic", "mismatch30": "mismatch30"}[variant]
    f = env.EXPORT / "data" / plant / var / f"{tier}.parquet"
    if not f.exists():
        return None
    tb = pq.read_table(f)
    meta = json.loads((env.EXPORT / "metadata" / f"{plant}.json").read_text())
    nx, ny = int(meta["n_x"]), int(meta["n_y"])
    st = lambda c, d: np.asarray(tb.column(c).to_pylist(), np.float32).reshape(tb.num_rows, -1, d)
    x, y = torch.as_tensor(st("x", nx)), torch.as_tensor(st("y", ny))
    u = torch.as_tensor(np.asarray(tb.column("u").to_pylist(), np.float32))[..., None]
    assert x.shape[1] == des.N + 1, (x.shape, des.N)
    from hyperkkl.systems import build_plant
    pl = build_plant(cfg["system"])
    yc = torch.stack([pl.h(xi.double()) for xi in x]).float()
    zo = Dm.run_latent(des, y.double()).float()
    z = Dm.run_latent(des, yc.double()).float()
    fam = tb.column("family").to_pylist()
    sig = tb.column("signal").to_pylist()
    x0 = st("x0", nx)[:, 0] if "x0" in tb.column_names else x[:, 0].numpy()
    m = [{"ic": x0[i].tolist(), "family": fam[i], "signal": sig[i], "source": str(f)} for i in range(len(fam))]
    burn = int(math.ceil(sc["t_burn"] / des.dt))
    return Dm.EpisodeSet(x, u, yc, y, z, zo, fam, burn, des.dt, m, 0)


def main(family, plant, variant):
    t0 = time.time()
    Dm = setup_cache()
    from hyperkkl import signals as S
    from hyperkkl import systems_extra, systems_bounded, systems_chaos  # noqa: F401
    from hyperkkl.systems import build_plant
    import seqdiv_signals as SD
    import valood_bc as VB
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = v2cfg.config(family, plant, variant)
    scfg, dcfg = cfg["system"], cfg["data"]
    out = env.DATA / family / plant / variant
    out.mkdir(parents=True, exist_ok=True)
    priv = env.HERE / "hidden_labels" / family / plant / variant
    priv.mkdir(parents=True, exist_ok=True)
    secret_f = env.HERE / "hidden_labels" / "SECRET"
    if not secret_f.exists():
        secret_f.write_text(hashlib.sha256(os.urandom(32)).hexdigest())
    secret = secret_f.read_text().strip()
    des = Dm.make_design(scfg)
    sc = dict(Dm.system_scales(cfg, dev))
    mf = env.EXPORT / "metadata" / f"{v2cfg.SYSTEM.get(plant, plant)}.json"
    if mf.exists() and plant not in v2cfg.SYSTEM:
        m = json.loads(mf.read_text())
        for k in ("x_std", "y_std", "x_mean"):
            sc[k + "_scales"] = sc[k]
            sc[k] = m[k]
    real = build_plant(scfg, perturb=float(scfg.get("mismatch", 0.0)))
    box = np.asarray(scfg["ic_box"], float)
    dom = np.asarray(sc["domain"], float)
    base = 31_000_000 + int(hashlib.sha256(f"{family}|{plant}|{variant}".encode()).hexdigest(), 16) % 1_000_000
    hseed = lambda s: 61_000_000 + int(hashlib.sha256(f"{secret}|{family}|{plant}|{variant}|{s}".encode()).hexdigest(), 16) % 1_000_000
    man = {"family": family, "plant": plant, "variant": variant, "config_source": cfg["v2"]["base_config"],
           "design": {"dt": des.dt, "N": des.N, "nz": int(des.A.shape[0]), "alpha": des.alpha}, "splits": {}}
    log = open(out / "build.log", "a")

    def save(name, es, private=False):
        f = (priv if private else out) / f"{name}.pt"
        if f.exists():
            print(f"  {name}: exists (frozen), kept", flush=True)
            es = torch.load(f, weights_only=False)
        else:
            torch.save(es, f)
        man["splits"][name] = {"n": len(es), "rejected": int(es.n_rejected), "private": private,
                               "sha256_x": sha(es.x), "sha256_u": sha(es.u), "sha256_y": sha(es.y),
                               "families": sorted(set(es.families))[:12]}
        msg = f"{time.strftime('%H:%M:%S')} {family}/{plant}/{variant} {name}: n {len(es)} rej {es.n_rejected}"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        return es

    zero = S.enforce_bounds(S.sample_signal("zero", "train", np.random.RandomState(0), des.dt, des.T,
                                            scale=float(scfg.get("u_scale", 1.0)), t_ref=float(scfg.get("t_ref", 1.0))),
                            des.dt, des.T, scfg.get("u_bounds"))
    p1 = gen(Dm, real, des, sc, scfg, dcfg, [zero], N_P1, base + 1, dev, ic_box=dom)
    save("phase1", p1)
    if not (out / "coll.pt").exists():
        nominal = Dm.model_plant(scfg)
        Xc = Dm._sobol(dom, int(dcfg["n_coll"]), base + 4)
        xt = torch.as_tensor(Xc, dtype=torch.float64)
        coll = {"x": xt.float(), "f0": nominal.f(xt, torch.zeros(len(xt), 1, dtype=xt.dtype)).float(),
                "y": nominal.h(xt).float()}
        torch.save(coll, out / "coll.pt")
    if family == "driven":
        tr = gen(Dm, real, des, sc, scfg, dcfg, profile_sigs(plant, "train", N_TRAIN // 3 + 1, base + 100_000, des),
                 N_TRAIN, base + 2, dev, k=3)
        save("train", tr)
        save("val", gen(Dm, real, des, sc, scfg, dcfg, profile_sigs(plant, "train", N_VAL, base + 200_000, des),
                        N_VAL, base + 3, dev))
        save("train_event", gen(Dm, real, des, sc, scfg, dcfg, None, N_TRAIN, base + 5, dev, k=3,
                                signals_fn=SD.d1_signals, families=S.TRAIN_FAMILIES))
        save("val_event", gen(Dm, real, des, sc, scfg, dcfg, None, N_VAL, base + 6, dev,
                              signals_fn=SD.d1_signals, families=S.TRAIN_FAMILIES))
        save("val_ood", gen(Dm, real, des, sc, scfg, dcfg, None, N_VAL, base + 7, dev,
                            signals_fn=VB.valood_signals, families=VB.FAMILIES))
        save("test_op_id", gen(Dm, real, des, sc, scfg, dcfg, profile_sigs(plant, "id", N_TEST, base + 300_000, des),
                               N_TEST, base + 8, dev))
        save("test_op_ood", gen(Dm, real, des, sc, scfg, dcfg, profile_sigs(plant, "ood", N_TEST, base + 400_000, des),
                                N_TEST, base + 9, dev))
        for t, sp in (("op_id", "id"), ("op_ood", "ood")):
            hs = hseed(t)
            es = gen(Dm, real, des, sc, scfg, dcfg, profile_sigs(plant, sp, N_HID, hs, des), N_HID, hs + 1, dev)
            save(f"hidden_{t}", es, private=True)
        for tier in S.TIERS:
            es = None if plant in v2cfg.SYSTEM else legacy_from_export(Dm, plant, variant, tier, des, cfg, sc)
            if es is None:
                es = Dm.generate(real, des, sc, scfg, dict(dcfg, k_cross=1), S.families_for(tier), N_TEST, tier,
                                 Dm.split_seed(scfg["name"], tier, "dev"), dev)
                for m in es.meta:
                    m["source"] = "regenerated (v1 tier generator, v1 split-seed rule)"
            save(f"test_legacy_{tier}", es)
    else:
        cb = core_box(box)
        tr = gen(Dm, real, des, sc, scfg, dcfg, [zero], N_TRAIN, base + 2, dev, ic_box=cb)
        # family A reference stds: settled part of the unforced standard training set
        b = tr.burn
        xr = tr.x[:, b:].reshape(-1, tr.x.shape[-1]).double()
        sc["x_std_forced"], sc["y_std_forced"], sc["x_mean_forced"] = sc["x_std"], sc["y_std"], sc["x_mean"]
        sc["x_std"] = xr.std(0).clamp(min=1e-9).tolist()
        sc["x_mean"] = xr.mean(0).tolist()
        sc["y_std"] = real.h(xr).std(0).clamp(min=1e-9).tolist()
        tr = gen(Dm, real, des, sc, scfg, dcfg, [zero], N_TRAIN, base + 2, dev, ic_box=cb)   # noise at the A y_std
        save("train", tr)
        save("val", gen(Dm, real, des, sc, scfg, dcfg, [zero], N_VAL, base + 3, dev, ic_box=cb))
        save("train_event", kick_episodes(Dm, real, des, sc, scfg, dcfg, N_TRAIN, base + 5, dev, cb))
        save("val_event", kick_episodes(Dm, real, des, sc, scfg, dcfg, N_VAL, base + 6, dev, cb))

        # ICs: explicit points (not a Sobol box) for the shells
        def ic_points(n, seed, lo, hi, plant_=real):
            x0 = shell_box(box, lo, hi, n, seed)
            orig = Dm._sobol
            Dm._sobol = lambda bx, nn, sd: x0[:nn]
            try:
                return gen(Dm, plant_, des, sc, scfg, dcfg, [zero], n, seed, dev, ic_box=box)
            finally:
                Dm._sobol = orig
        save("val_ood", ic_points(N_VAL, base + 7, *SHELL_VAL))
        save("test_id", gen(Dm, real, des, sc, scfg, dcfg, [zero], N_TEST, base + 8, dev, ic_box=cb))
        save("test_ic_ood", ic_points(N_TEST, base + 9, *SHELL_TEST))
        shifted = build_plant(scfg, perturb=float(scfg.get("mismatch", 0.0)) + 0.15)
        try:
            save("test_param_shift", gen(Dm, shifted, des, sc, scfg, dcfg, [zero], N_TEST, base + 10, dev, ic_box=cb))
        except RuntimeError as ex:                 # every episode of the shifted plant leaves the envelope
            man["splits"]["test_param_shift"] = {"n": 0, "unavailable": f"shifted plant (+0.15) leaves 1.5x the "
                                                 f"envelope from every IC after 6 redraws ({ex})"}
            print("  test_param_shift: UNAVAILABLE (shifted plant escapes)", flush=True)
        save("hidden_id", gen(Dm, real, des, sc, scfg, dcfg, [zero], N_HID, hseed("id"), dev, ic_box=cb), private=True)
        save("hidden_ic_ood", ic_points(N_HID, hseed("ic_ood"), *SHELL_TEST), private=True)
    (out / "scales.json").write_text(json.dumps(sc, indent=1, default=str))
    man["wall_s"] = time.time() - t0
    man["built"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"done {family}/{plant}/{variant} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main(*sys.argv[1:4])
