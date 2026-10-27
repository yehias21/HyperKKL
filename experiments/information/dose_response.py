"""Dose-response: evaluation-input excitation vs information budget vs achieved error.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 .venv/bin/python experiments/information/dose_response.py \
        --system quad_tank --seeds 0 1 2 --levels 0.25 0.5 1 2 4 --n_ep 32

Read-only: trained models and training data come from the shared cache under the keys of the runs'
own code snapshot (experiments/code/b2cb5b3dfe80.tar.gz, extracted in the scratchpad; the live
package's model hash differs, exactly as experiments/diagnostics did). Nothing is written outside
experiments/information/dose/.

Levels: the evaluation inputs are drawn from the id families with the plant's signal time unit
t_ref multiplied by s (s < 1: faster switching / higher frequencies, less energy in the slow band;
s > 1: slower, more slow-band energy). Same amplitude box, same IC box, same sensor and process noise,
new seeds (never the cached evaluation episodes). Per level: the information budget (Bayesian PCRB
and local recursion; windows 1, 3, 10 / alpha and the full past) and the settled NRMSE of HyperKKL
(lora, the node's trained models), the node's EKF/UKF (the modeller's partial model), an EKF with
the full model (information-tracking reference) and the grey-box EKF/UKF refitted on the run's own
training data (baselines_data.fit_all, as reeval does).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
SNAP = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/snap_b2cb")
sys.path.insert(0, str(SNAP))
import torch  # noqa: E402
import numpy as np  # noqa: E402
import hyperkkl  # noqa: E402
assert Path(hyperkkl.__file__).resolve().is_relative_to(SNAP), hyperkkl.__file__
from hyperkkl import baselines_data as BD  # noqa: E402
from hyperkkl import cache as C  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl import signals as S  # noqa: E402
from hyperkkl import systems_extra  # noqa: E402,F401
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.config import digest  # noqa: E402
from hyperkkl.evaluate import build_observers, initial_guess  # noqa: E402
from hyperkkl.job import _to, method_cfg  # noqa: E402
from hyperkkl.metrics import episode_metrics  # noqa: E402
from hyperkkl.observers import EKF, run  # noqa: E402
from hyperkkl.systems import build_plant  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infolib as L  # noqa: E402  (uses the same snapshot package: hyperkkl is already imported)


def ro_cached(path, build=None, **kw):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cache miss (read-only mode): {path.name}")
    return torch.load(path, weights_only=False, map_location="cpu")


Dm.CACHE = ROOT / "experiments" / "cache"
Dm._cached = ro_cached
C.cached = ro_cached
_STORED = {}
Dm.system_scales = lambda cfg, device="cpu": _STORED["scales"]
MC = Dm.CACHE / "models"


def load_run(node, system, seed, device, methods):
    payload = L.run_payload(system, node, seed)
    cfg = copy.deepcopy(payload["config"])
    _STORED["scales"] = payload["scales"]
    scales = payload["scales"]
    des = Dm.make_design(cfg["system"])
    data = Dm.training_data(cfg, seed, payload["device"].split(":")[0])     # key carries the run's device type
    mh = module_hash(MODEL_MODULES)
    k1 = digest(cfg, "system", "data", "phase1", extra=[seed, mh])
    base = _to(ro_cached(MC / f"{system}_p1_{k1}.pt"), device)
    trained = {}
    for m in methods:
        kind, mc = method_cfg(cfg, m)
        k2 = digest(mc, "system", "data", "phase1", "phase2", extra=[seed, kind, mh])
        trained[m] = _to(ro_cached(MC / f"{system}_p2_{kind}_{k2}.pt"), device)
    return payload, cfg, scales, des, data, base, trained


def make_episodes(cfg, scales, des, level: float, n_ep: int, li: int):
    scfg, dcfg = cfg["system"], cfg["data"]
    scfg_mod = dict(scfg, t_ref=float(scfg.get("t_ref", 1.0)) * level)
    plant = build_plant(scfg, perturb=float(scfg.get("mismatch", 0.0)))
    seed = Dm.split_seed(scfg["name"], "id", cfg["eval"].get("split", "dev"), 0) + 7_000 + li
    return Dm.generate(plant, des, scales, scfg_mod, dict(dcfg, k_cross=1), S.families_for("id"),
                       n_ep, "id", seed, "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--node", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--levels", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--n_ep", type=int, default=32)
    ap.add_argument("--filters_once", action="store_true", help="run the seed-independent filters for the first seed only")
    ap.add_argument("--no_greybox", action="store_true")
    ap.add_argument("--out", default=str(HERE / "dose"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    node = a.node or L.RUN_PARTIAL[a.system]
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *s: print(time.strftime("[%H:%M:%S]"), *s, flush=True)

    # episodes and budgets once (shared by the seeds)
    pay0 = L.run_payload(a.system, node, a.seeds[0])
    cfg0, scales = pay0["config"], pay0["scales"]
    scfg = cfg0["system"]
    des = Dm.make_design(scfg)
    plant_full = build_plant(scfg)                           # full knowledge: the information budget's model
    sensors = list(plant_full.sensors)
    alpha = float(scfg["alpha"])
    es_ref = L.load_cached_eval(cfg0, "id")
    settle = int(0.5 * (es_ref.x.shape[1] - 1))
    Sigma = L.climatology(es_ref.x[:, settle:].double())
    f_slow = L.slow_mode_rate(plant_full, scales["x_mean"])
    levels = {}
    for li, s in enumerate(a.levels):
        t0 = time.time()
        es = make_episodes(cfg0, scales, des, s, a.n_ep, li)
        b = L.budget_along(plant_full, es.x, es.u, cfg0, scales, sensors, [1 / alpha, 3 / alpha, 10 / alpha],
                           n_ep=a.n_ep, n_t=12, Sigma=Sigma)
        b["input"] = L.input_descriptors(es.u, scales["dt"], f_slow / (2 * np.pi))
        b["families"] = sorted(set(es.families)); b["n_rejected"] = es.n_rejected
        levels[f"{s:g}"] = {"es": es, "budget": b}
        log(f"level s={s:g}: {len(es)} episodes, bits full {b['bayes']['full']['bits']:.2f}, 3/a {b['bayes']['windows'][f'{3 / alpha:g}']['bits']:.2f}, "
            f"unm lb full {b['bayes']['full']['unmeasured_lb']:.4f} 3/a {b['bayes']['windows'][f'{3 / alpha:g}']['unmeasured_lb']:.4f}, "
            f"slow-band {b['input']['slow_band_fraction']:.3f} ({time.time() - t0:.0f}s)")

    for si, seed in enumerate(a.seeds):
        out_path = out_dir / f"{a.system}_{node}_seed{seed}.json"
        payload, cfg, scales, des, data, base, trained = load_run(node, a.system, seed, device, ["lora"])
        cfg = copy.deepcopy(cfg)
        cfg["baselines"] = ["ekf", "ukf"] + ([] if a.no_greybox else ["greybox_ekf", "greybox_ukf"])
        t0 = time.time()
        fits = BD.fit_all(cfg, data, scales, des, device, seed, log=log) if not a.no_greybox else None
        log(f"seed {seed}: grey-box fits {time.time() - t0:.0f}s")
        obs = build_observers(cfg, des, scales, base, trained, None, device, fits)
        q = float(cfg["data"].get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
        r = max(float(cfg["data"]["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(scales["y_std"])
        box = np.asarray(scfg["ic_box"], float)
        p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
        obs["ekf_full"] = EKF(build_plant(scfg, knowledge="full"), des.dt, q, r, p0)
        if a.filters_once and si > 0:
            for k in ("ekf", "ukf", "ekf_full"):
                obs.pop(k, None)
        res = {"system": a.system, "node": node, "seed": seed, "device": device, "snapshot": payload["code"],
               "levels": {}, "alpha": alpha, "sensors": sensors, "f_slow": f_slow}
        sigma = scales["x_std"]
        for tag, lv in levels.items():
            es = lv["es"]
            X, Y, U = es.x.to(device), es.y.to(device), es.u.to(device)
            X0 = initial_guess(cfg, len(es), "center", seed, "cpu").to(device)
            settle = int(0.5 * (X.shape[1] - 1))
            r_lv = {"budget": lv["budget"], "methods": {}}
            for name, ob in obs.items():
                t0 = time.time()
                xh = torch.cat([run(ob, Y[i:i + 1024], U[i:i + 1024], X0[i:i + 1024]) for i in range(0, len(Y), 1024)])
                m = episode_metrics(X, xh, sigma, settle, sensors)
                nx = X.shape[-1]
                r_lv["methods"][name] = {
                    "nrmse": float(np.mean(np.asarray(m["nrmse"]))), "nrmse_episodes": np.asarray(m["nrmse"]).tolist(),
                    "state": [float(np.mean(np.asarray(m[f"nrmse_x{j + 1}"]))) for j in range(nx)],
                    "rms": float(np.sqrt(np.mean(np.square(np.asarray(m["nrmse"]))))),
                    "rms_state": [float(np.sqrt(np.nanmean(np.square(np.asarray(m[f"nrmse_x{j + 1}"]))))) for j in range(nx)],
                    "unmeasured": float(np.nanmean(np.asarray(m["nrmse_unmeasured"]))), "measured": float(np.nanmean(np.asarray(m["nrmse_measured"]))),
                    "wall_s": time.time() - t0}
                log(f"  seed {seed} s={tag} {name:12s} nrmse {r_lv['methods'][name]['nrmse']:.4f} "
                    f"unm {r_lv['methods'][name]['unmeasured']:.4f} ({time.time() - t0:.0f}s)")
            res["levels"][tag] = r_lv
            out_path.write_text(json.dumps(res, indent=1))
        log("written", out_path)


if __name__ == "__main__":
    main()
