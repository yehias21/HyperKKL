r"""Data-driven baselines on the real battery data (experiments/real/DESIGN.md, Addendum A).

    CUDA_VISIBLE_DEVICES=5 python -I experiments/real/run.py baselines --seed 0     # fit + tune
    CUDA_VISIBLE_DEVICES=5 python -I experiments/real/run.py evaluate_baselines

Uses hyperkkl/baselines_data.py unchanged (Dictionary / fit_edmd / KoopmanKF, fit_greybox /
GreyBoxModel inside observers.EKF / UKF) and train.nlox with the masked loss. Training data =
the learned observers' real training data: each training recording once (the union of the
training windows), x = (SOC_ref, V1_model, V2_model, dT, Vm); validation = Cycle 4.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import baselines_data as BD
from . import realdata as R
from .realdata_run import (CACHE, DT_S, FILTER_GRID, INIT_ERR, R_DT, SEEDS, cold_plant, design, expand,
                           filter_grid, full_cfg, grid_filter, load_prep, make_filter, run_soc,
                           soc_metrics, _pad_full)

DATA_BASELINES = ("koopman_kf", "koopman_kf_bilinear", "greybox_ekf", "greybox_ukf", "nlox")
KOOP_GRID = {"q_scale": [0.1, 1.0, 10.0, 100.0], "r_vm": [0.003, 0.01, 0.03, 0.1]}


def recording_pairs(prep, names):
    """(x, u) tensors, one episode per recording (full length), float64."""
    out = []
    for c in names:
        r, rp = prep["recs"][f"25/{c}"], prep["replays"][f"25/{c}"]
        x = np.stack([r.soc, rp[:, 1], rp[:, 2], r.dT, r.vm], 1)
        out.append((torch.as_tensor(x[None]), torch.as_tensor(r.u[None, :, None])))
    return out


def koopman_observer(fit, dic, g, sensors=(3, 4)):
    return BD.KoopmanKF("koopman", dic, fit["model"], fit["Q"] * g["q_scale"], np.array([R_DT, g["r_vm"]]),
                        sensors, fit["P0"])


def fit(seed: int, device: str, log=print):
    from .train import nlox
    torch.manual_seed(seed); np.random.seed(seed)
    prep = load_prep()
    scales = prep["scales"]
    params = prep["scfg"]["params"]
    plant = R.Pan18650PF(sensors=(3, 4), **params)
    des = design()
    opt = BD.options({})
    t0 = time.time()
    tr = BD.transitions(recording_pairs(prep, R.SPLIT["train"]), int(opt["n_fit"]), seed + 1, device)
    va = BD.transitions(recording_pairs(prep, R.SPLIT["val"]), int(opt["n_val"]), seed + 2, device)
    log(f"transitions: train {len(tr[0])}, val {len(va[0])}")
    # Koopman: dictionary centres by k-means on training states (as fit_dictionary, data = the
    # training recordings)
    Xk, _, _ = BD.transitions(recording_pairs(prep, R.SPLIT["train"]), int(opt["kmeans_points"]), seed + 17, device)
    mean = torch.as_tensor(scales["x_mean"], dtype=torch.float64)
    std = torch.as_tensor(scales["x_std"], dtype=torch.float64)
    cen = BD.kmeans((Xk - mean.to(device)) / std.to(device), 400, int(opt["kmeans_iters"]), seed)
    dic = BD.Dictionary(cen.cpu(), mean, std)
    P0 = BD.lifted_prior(dic, prep["scfg"]["ic_box"], seed=seed, device=device)
    koop = {}
    for kind in ("linear", "bilinear"):
        f = BD.fit_edmd(dic, tr, va, kind, opt, log)
        f["Q"] = f["Q"].cpu()
        f["P0"] = P0 + torch.diag(f["Q"])
        koop[kind] = f
    gb = BD.fit_greybox(plant, tr, va, des.dt, scales, float(prep["scfg"]["u_scale"]), seed, device, opt, log)
    log(f"fits done ({time.time() - t0:.0f}s)")
    # noise tuning on the validation windows (wrong-prior protocol)
    Y, U, S, X0, _, _ = expand(prep["sets"]["val"])
    tuning = {}
    for kind, name in (("linear", "koopman_kf"), ("bilinear", "koopman_kf_bilinear")):
        sc = []
        for qs in KOOP_GRID["q_scale"]:
            for rv in KOOP_GRID["r_vm"]:
                g = {"q_scale": qs, "r_vm": rv}
                sh = run_soc(koopman_observer(koop[kind], dic, g), Y, U, X0, device, chunk=10 ** 9,
                             dtype=torch.float64)
                m = soc_metrics(sh, S.numpy())
                sc.append({**g, "val_rmse_settled": float(np.mean(m["rmse_settled"]))})
        best = min(sc, key=lambda s: s["val_rmse_settled"])
        tuning[name] = {"choice": {k: best[k] for k in ("q_scale", "r_vm")}, "scores": sc}
        log(f"  {name}: choice {tuning[name]['choice']} val {best['val_rmse_settled']:.3f} %")
    grid = filter_grid()
    rep = lambda a: a.repeat(len(grid), *([1] * (a.dim() - 1)))
    for kind, name in (("ekf", "greybox_ekf"), ("ukf", "greybox_ukf")):
        f = grid_filter(kind, gb["model"], scales, grid, len(Y))
        sh = run_soc(f, rep(Y), rep(U), rep(X0), device, chunk=10 ** 9, dtype=torch.float64)
        sc = []
        for i, g in enumerate(grid):
            m = soc_metrics(sh[i * len(Y):(i + 1) * len(Y)], S.numpy())
            sc.append({**g, "val_rmse_settled": float(np.mean(m["rmse_settled"]))})
        best = min(sc, key=lambda s: s["val_rmse_settled"])
        tuning[name] = {"choice": {k: best[k] for k in ("q_soc", "c", "r_vm")}, "scores": sc}
        log(f"  {name}: choice {tuning[name]['choice']} val {best['val_rmse_settled']:.3f} %")
    # NLOX with the masked loss (phase-2 data: real training / validation windows)
    cfg = full_cfg(prep)
    nl = nlox(cfg, {"train": prep["sets"]["train"], "val": prep["sets"]["val"]}, R.masked_scales(scales),
              des, device, seed, log)
    from .job import _to
    nl = _to(nl, "cpu")
    obs = nlox_observer(prep, nl, device)
    sh = run_soc(obs, Y, U, X0, device)
    tuning["nlox"] = {"val_rmse_settled": float(np.mean(soc_metrics(sh, S.numpy())["rmse_settled"]))}
    log(f"  nlox: val {tuning['nlox']['val_rmse_settled']:.3f} %  ({time.time() - t0:.0f}s)")
    info = {"koopman_kf": koop["linear"]["info"], "koopman_kf_bilinear": koop["bilinear"]["info"],
            "greybox": gb["info"], "nlox": {k: nl.get(k) for k in ("n_params", "wall_s", "val")}}
    torch.save({"dic": dic, "koopman": koop, "greybox_mlp": gb["model"].mlp, "nlox": nl,
                "tuning": tuning, "info": info, "seed": seed}, CACHE / f"baselines_seed{seed}.pt")
    log(f"saved ({time.time() - t0:.0f}s)")


def nlox_observer(prep, nl, device):
    from .evaluate import build_observers
    from .job import _to
    cfg = full_cfg(prep)
    return build_observers(cfg, design(), prep["scales"], None, {}, _to(nl, device), device)["nlox"]


def observers(prep, b, plant, device):
    """name -> (observer, dtype) for one seed's fits and one plant (25 or 0 degC model)."""
    t = b["tuning"]
    gm = BD.GreyBoxModel(plant, b["greybox_mlp"])
    return {"koopman_kf": (koopman_observer(b["koopman"]["linear"], b["dic"], t["koopman_kf"]["choice"]), torch.float64),
            "koopman_kf_bilinear": (koopman_observer(b["koopman"]["bilinear"], b["dic"],
                                                     t["koopman_kf_bilinear"]["choice"]), torch.float64),
            "greybox_ekf": (make_filter("ekf", gm, prep["scales"], t["greybox_ekf"]["choice"]), torch.float64),
            "greybox_ukf": (make_filter("ukf", gm, prep["scales"], t["greybox_ukf"]["choice"]), torch.float64),
            "nlox": (nlox_observer(prep, b["nlox"], device), torch.float32)}


def evaluate(device, log=print, seeds=SEEDS, methods=None, temps=("25", "0"), tag=None):
    prep = load_prep()
    params = prep["scfg"]["params"]
    plants = {"25": R.Pan18650PF(sensors=(3, 4), **params), "0": cold_plant(params, 0.0 + prep["offset_K"])[0]}
    sets = {k: v for k, v in prep["sets"].items() if k != "train"}
    temp_of = lambda name: "0" if name.startswith("test0/") else "25"
    batch = {"25": [], "0": []}
    for name, es in sets.items():
        Y, U, S, X0, _, _ = expand(es)
        batch[temp_of(name)].append((name, Y, U, X0))
    fkeys = {"25": [f"25/{c}" for c in R.SPLIT["test"]], "0": [k for k in prep["recs"] if k.startswith("0/")]}
    fb = {tmp: _pad_full([prep["recs"][k] for k in keys]) for tmp, keys in fkeys.items()}
    pred, full = {}, {k: {} for ks in fkeys.values() for k in ks}
    for seed in seeds:
        b = torch.load(CACHE / f"baselines_seed{seed}.pt", weights_only=False, map_location="cpu")
        b["dic"]._dev = {}                  # per-device copies pickled with the object: rebuild
        for f_ in b["koopman"].values():
            f_["model"]._dev = {}
        for tmp in temps:
            obs = {k: v for k, v in observers(prep, b, plants[tmp], device).items()
                   if methods is None or k in methods}
            Y = torch.cat([x[1] for x in batch[tmp]]); U = torch.cat([x[2] for x in batch[tmp]])
            X0 = torch.cat([x[3] for x in batch[tmp]])
            for m, (ob, dtype) in obs.items():
                t0 = time.time()
                chunk = 64 if m == "koopman_kf_bilinear" else 10 ** 9
                sh = run_soc(ob, Y, U, X0, device, chunk=chunk, dtype=dtype)
                o = 0
                for name, y, _, _ in batch[tmp]:
                    pred[(name, m, seed)] = sh[o:o + len(y)]; o += len(y)
                sf = run_soc(ob, *fb[tmp], device, chunk=chunk, dtype=dtype)
                for i, k in enumerate(fkeys[tmp]):
                    full[k][f"{m}/{seed}"] = sf[i, :len(prep["recs"][k])]
                log(f"  {m}/{seed:<2d} {tmp:>2s} degC: " + "  ".join(
                    f"{n.split('/')[-1]} {np.mean(soc_metrics(pred[(n, m, seed)], soc_truth(prep, n))['rmse_settled']):.2f}"
                    for n, *_ in batch[tmp]) + f"  ({time.time() - t0:.0f}s)")
    if tag is None:
        tag = "" if tuple(seeds) == tuple(SEEDS) else "_seed" + "".join(map(str, seeds))
    torch.save({"pred": pred, "full": full}, CACHE / f"predictions_baselines{tag}.pt")
    log("evaluate_baselines done")


_TRUTH = {}


def soc_truth(prep, name):
    if name not in _TRUTH:
        _TRUTH[name] = expand(prep["sets"][name])[2].numpy().astype(np.float64)
    return _TRUTH[name]
