"""Extrapolation test (PREREG addendum 3): training-state envelope vs settled eval states, and
HyperKKL error inside vs outside. Eval only: cached models are loaded, a cache miss raises.

    python coverage.py --which old --system cartpole --seed 0        # TEST-C3 lora (+ greybox_ukf for cartpole), snapshot code
    python coverage.py --which new --system cartpole --seed 0        # A2, A4 from p2_rescore models
    python coverage.py --which camp --system quad_tank --seed 0      # A2 from the campaign's C3-partial models (read-only)
-> diag/coverage/<which>_<system>_s<seed>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P = HERE.parent
LIVE = P.parent.parent
LIVE_CACHE = LIVE / "experiments" / "cache"
CAMP = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/beat_greybox")
TIERS = ("id", "shape_ood", "sequence_ood")


def train_states(data):
    xs = [data["p1"].x.reshape(-1, data["p1"].x.shape[-1]).cpu(), data["train"].x.reshape(-1, data["train"].x.shape[-1]).cpu(),
          data["coll"]["x"].cpu()]
    return torch.cat(xs).double().numpy()


def analyse(cfg, des, scales, data, observers, device, Ev, Dm, sensors):
    cfg0 = cfg
    xtr = train_states(data)
    lo, hi = xtr.min(0), xtr.max(0)
    sig = np.asarray(scales["x_std"], float)
    nx = len(sig)
    unm = [j for j in range(nx) if j not in sensors]
    cfg = dict(cfg); cfg["eval"] = dict(cfg["eval"], conditions=["clean"])
    out = {"envelope": {"lo": lo.tolist(), "hi": hi.tolist()}, "x_std": sig.tolist(), "sensors": list(sensors),
           "train_q": np.quantile(xtr, [0.01, 0.5, 0.99], axis=0).tolist(), "tiers": {}}
    # ---- joint coverage (PREREG addendum 4): KD-tree NN distance in x_std units
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    ep = lambda ds, st=5: ds.x[:, ::st].reshape(-1, nx).double().cpu().numpy()
    parts = {"p1": ep(data["p1"]), "train": ep(data["train"]), "coll": data["coll"]["x"].double().cpu().numpy()}
    ref = np.concatenate(list(parts.values())) / sig
    tree = cKDTree(ref)
    vx = data["val"].x; vs = int(float(cfg["eval"].get("settle_frac", 0.5)) * (vx.shape[1] - 1))
    dnull, _ = tree.query(vx[:, vs:].reshape(-1, nx).double().cpu().numpy() / sig)
    d99 = float(np.quantile(dnull, 0.99))
    out["joint"] = {"n_ref": int(len(ref)), "null_q50_q99": [float(np.median(dnull)), d99],
                    "std": {k: v.std(0).tolist() for k, v in parts.items()}}
    for b in Ev._blocks(cfg, des, scales, device):
        X = b["es"].x.to(device); N1 = X.shape[1]; settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (N1 - 1))
        xs = X[:, settle:].double().cpu().numpy().reshape(-1, nx)
        outs = (xs < lo) | (xs > hi)
        joint = outs.any(1)
        q01, q99 = np.quantile(xtr, 0.01, axis=0), np.quantile(xtr, 0.99, axis=0)   # secondary (no verdict)
        joint_q = ((xs < q01) | (xs > q99)).any(1)
        x1 = X[..., 0].double().cpu().numpy()
        dev_, _ = tree.query(xs / sig)
        far = dev_ > d99
        qb = np.quantile(dev_, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(dev_, qb)
        c0 = xs.mean(0)
        rad = lambda a: np.quantile(np.linalg.norm((a - c0) / sig, axis=1), [0.01, 0.1, 0.5, 0.9, 0.99]).tolist()
        t = {"radius_q_about_eval_mean": {"eval": rad(xs), **{k: rad(v) for k, v in parts.items()}},
             "joint": {"frac_beyond_d99": float(far.mean()), "d_q50_q99": [float(np.median(dev_)), float(np.quantile(dev_, 0.99))],
                       "eval_std": xs.std(0).tolist(), "bin_edges": qb.tolist()},
             "eval_q": np.quantile(xs, [0.01, 0.5, 0.99], axis=0).tolist(), "frac_out_state": outs.mean(0).tolist(),
             "frac_out_joint": float(joint.mean()), "frac_out_q01_q99_joint": float(joint_q.mean()),
             "frac_out_q01_q99_state": ((xs < q01) | (xs > q99)).mean(0).tolist(),
             "x1_abs_q50_q99_at": {k: np.quantile(np.abs(x1[:, i]), [0.5, 0.99]).tolist()
                                   for k, i in (("t0", 0), ("tmid", (N1 - 1) // 2), ("tend", N1 - 1))},
             "methods": {}}
        for name, ob in observers.items():
            with torch.no_grad():
                xh = Ev._run_chunks(ob, b["es"].y.to(device), b["u"].to(device), b["x0"].to(device), 1024)
            e = ((xh[:, settle:] - X[:, settle:]).double().cpu().numpy().reshape(-1, nx)) / sig
            fin = np.isfinite(e).all(1)
            m = {}
            for lab, mask in (("inside", ~joint & fin), ("outside", joint & fin), ("all", fin),
                              ("in_q", ~joint_q & fin), ("out_q", joint_q & fin)):
                if mask.sum() == 0:
                    m[lab] = None; continue
                rs = np.sqrt(np.mean(e[mask] ** 2, 0))
                m[lab] = {"n": int(mask.sum()), "rms_state": rs.tolist(), "unm": float(np.mean(rs[unm]))}
            for lab, mask in (("near", ~far & fin), ("far", far & fin)):
                if mask.sum() == 0:
                    m[lab] = None; continue
                rs = np.sqrt(np.mean(e[mask] ** 2, 0))
                m[lab] = {"n": int(mask.sum()), "rms_state": rs.tolist(), "unm": float(np.mean(rs[unm]))}
            eu = np.sqrt(np.mean(e[:, unm] ** 2, 1))
            m["bins_unm"] = [float(np.sqrt(np.mean(e[(bins == k) & fin][:, unm] ** 2))) for k in range(5)]
            m["spearman_d_err"] = float(spearmanr(dev_[fin], eu[fin]).statistic)
            t["methods"][name] = m
        out["tiers"][b["tier"]] = t
    # translation invariance of the TRUE plant in x1 (cartpole: cart position)
    plant = Dm.build_plant(cfg["system"], perturb=float(cfg["system"].get("mismatch", 0.0)))
    g = torch.Generator().manual_seed(0)
    xb = torch.tensor(np.asarray(cfg["system"]["ic_box"], float))
    x = xb[:, 0] + (xb[:, 1] - xb[:, 0]) * torch.rand(256, nx, generator=g, dtype=torch.float64)
    nu = int(getattr(plant, "nu", 1))
    u = torch.rand(256, nu, generator=g, dtype=torch.float64) - 0.5
    f0 = plant.f(x, u)
    dev = {}
    for d in (10.0, 100.0):
        xs_ = x.clone(); xs_[:, 0] += d
        dev[str(d)] = float((plant.f(xs_, u) - f0).abs().max())
    out["f_shift_x1_maxabs"] = dev
    out["f_scale"] = float(f0.abs().max())
    return out


def run_old(system, seed, device):
    sys.path.insert(0, str(P / "diag" / "oldcode"))
    from hyperkkl import data as Dm, evaluate as Ev, baselines_data as BD
    from hyperkkl.cache import MODEL_MODULES, module_hash
    from hyperkkl.config import digest
    from hyperkkl.job import method_cfg
    from hyperkkl.systems import build_plant
    from hyperkkl import systems_extra  # noqa: F401
    assert Dm.CACHE.resolve() != LIVE_CACHE.resolve()

    def _ro(path, build, **kw):
        live = LIVE_CACHE / Path(path).relative_to(Dm.CACHE)
        if live.exists():
            return torch.load(live, weights_only=False, map_location="cpu")
        raise FileNotFoundError(live)
    Dm._cached = _ro
    cfg = json.load(open(LIVE / "experiments" / "tree" / "TEST-C3" / "runs" / system / f"seed{seed}" / "results.json"))["config"]
    cfg["seed"] = seed
    scales = Dm.system_scales(cfg, device); des = Dm.make_design(cfg["system"]); data = Dm.training_data(cfg, seed, device)
    mh = module_hash(MODEL_MODULES)
    k1 = digest(cfg, "system", "data", "phase1", extra=[seed, mh])
    kind, mc = method_cfg(cfg, "lora")
    k2 = digest(mc, "system", "data", "phase1", "phase2", extra=[seed, kind, mh])
    to = lambda d: {k: (v.to(device) if isinstance(v, torch.nn.Module) else v) for k, v in d.items()}
    base = to(torch.load(LIVE_CACHE / "models" / f"{system}_p1_{k1}.pt", weights_only=False, map_location="cpu"))
    r = to(torch.load(LIVE_CACHE / "models" / f"{system}_p2_{kind}_{k2}.pt", weights_only=False, map_location="cpu"))
    fits = None
    cfg["baselines"] = ["greybox_ukf"] if system == "cartpole" else []
    if cfg["baselines"]:
        fits = BD.fit_all(cfg, data, scales, des, device, seed, print)
    obs = Ev.build_observers(cfg, des, scales, base, {"lora": r}, None, device, fits)
    return analyse(cfg, des, scales, data, obs, device, Ev, Dm, list(build_plant(cfg["system"]).sensors))


def run_new(system, seed, device, camp):
    sys.path.insert(0, str(HERE))
    import run_p2 as R
    from hyperkkl import evaluate as Ev, data as Dm
    from hyperkkl.systems import build_plant
    real = R.Cm.cached

    def ro(path, build):
        if not Path(path).exists():
            raise FileNotFoundError(f"model cache miss (no retraining): {path}")
        return real(path, build)
    R.Cm.cached = ro
    R.OUT = P / "diag" / "coverage" / "jobs"
    node = "C3-partial" if camp else "TEST-C3"
    if camp:
        R.MC = CAMP / "models"
    j = R.Job(node, system, seed, device)
    arms = ("A2", "A4") if system == "cartpole" else ("A2",)
    obs = {a: j.train_arm(a)[0] for a in arms}
    return analyse(j.cfg, j.des0, j.scales, j.data0, obs, device, Ev, Dm, list(build_plant(j.cfg["system"]).sensors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["old", "new", "camp"], required=True)
    ap.add_argument("--system", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.set_num_threads(2)
    out = run_old(a.system, a.seed, a.device) if a.which == "old" else run_new(a.system, a.seed, a.device, a.which == "camp")
    d = P / "diag" / "coverage"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.which}_{a.system}_s{a.seed}.json").write_text(json.dumps(out, indent=1, default=float))
    print("written", flush=True)


if __name__ == "__main__":
    main()
