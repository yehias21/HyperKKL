"""Inclusion check (rule: INCLUSION.md, fixed before computing). CPU only.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/cartpole_bounded/inclusion.py [plants]

Writes experiments/cartpole_bounded/inclusion/<plant>.json; `inclusion.py table` prints the table.
New plants (cartpole_b, two_link_arm_b): config = configs/systems/<name>.yaml merged into base.yaml;
scales and eval_id episodes are generated on CPU into a scratch cache (the live cache is not touched).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "information"))
import infolib as L  # noqa: E402
from hyperkkl import data as Dm, signals as S  # noqa: E402
from hyperkkl.config import load  # noqa: E402
from hyperkkl.systems import build_plant  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "inclusion"
SCRATCH_CACHE = Path(os.environ.get("INCL_CACHE", "/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/"
                                    "7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/incl_cache"))
NEW = ["cartpole_b", "two_link_arm_b"]
ALL = L.ALL + NEW
B = json.loads((ROOT / "experiments" / "information" / "budgets.json").read_text())


def setup(s):
    if s in NEW:
        Dm.CACHE = SCRATCH_CACHE                       # never write into the live cache
        cfg = load(s)
        scales = Dm.system_scales(cfg, "cpu")
        es = Dm.eval_data(cfg, "id", device="cpu")
        return cfg, scales, es
    pay = L.run_payload(s)
    return pay["config"], pay["scales"], L.load_cached_eval(pay["config"], "id")


def gramian(plant, es, cfg, scales, n_ep=32, n_t=12):
    alpha = float(cfg["system"]["alpha"]); dt = float(scales["dt"])
    x, u = es.x[:n_ep].double(), es.u[:n_ep].double()
    N1 = x.shape[1]; settle = int(0.5 * (N1 - 1))
    W = min(int(round(3 / alpha / dt)), settle)
    F, H = L.jacobians(plant, x, u, dt)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float64)
    D, Di = torch.diag(sig), torch.diag(1 / sig)
    _, R = L.noise_matrices(cfg, scales); Rinv = torch.linalg.inv(R)
    ks = torch.linspace(settle, N1 - 1, n_t).round().long()
    ratios = []
    for k in ks.tolist():
        k0 = k - W
        Phi = torch.eye(plant.nx, dtype=torch.float64).expand(x.shape[0], -1, -1).clone()
        Wo = torch.zeros_like(Phi)
        for i in range(W + 1):
            Hn = H[:, k0 + i] @ D
            Pn = Di @ Phi @ D
            Wo = Wo + Pn.transpose(-1, -2) @ Hn.transpose(-1, -2) @ Rinv @ Hn @ Pn
            if i < W:
                Phi = F[:, k0 + i] @ Phi
        ev = torch.linalg.eigvalsh(0.5 * (Wo + Wo.transpose(-1, -2)))
        ratios.append((ev[:, 0].clamp(min=0) / ev[:, -1]).numpy())
    r = np.concatenate(ratios)
    return {"W_samples": W, "ratio_median": float(np.median(r)), "ratio_p10": float(np.percentile(r, 10)),
            "frac_full_rank": float((r > 1e-10).mean()), "cond_median": float(1 / max(np.median(r), 1e-300))}


def bounded(plant, cfg, scales, n=64, mult=5):
    scfg, dcfg = cfg["system"], cfg["data"]
    des0 = Dm.make_design(scfg)
    des = Dm.Design(des0.A, des0.B, des0.dt, mult * des0.N, des0.alpha, des0.step)
    box = np.asarray(scfg["ic_box"], dtype=np.float64)
    sp = float(dcfg.get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
    NT = des0.N; win = max(1, NT // 10)
    out = {}
    for si, (name, fams, regime) in enumerate((("train", S.TRAIN_FAMILIES, "train"),
                                               ("id", S.families_for("id"), "id"),
                                               ("shape_ood", S.families_for("shape_ood"), "shape_ood"),
                                               ("sequence_ood", S.families_for("sequence_ood"), "sequence_ood"))):
        rng = np.random.RandomState(1000 + si)
        sigs = Dm._signals(fams, n, regime, rng, des, scfg)
        x0 = Dm._sobol(box, n, 77 + si)
        x, _, _ = Dm.integrate(plant, des, x0, sigs, sp, seed=5 + si)
        fin = bool(torch.isfinite(x).all())
        xs = torch.nan_to_num(x, nan=1e30, posinf=1e30, neginf=-1e30)
        Sd = xs.std(0)                                                   # (N+1, nx)
        Sw = lambda k: Sd[max(0, k - win):k + 1].mean(0)
        S_T, S_5T = Sw(NT), Sw(mult * NT)
        seg = lambda a, b: Sd[a * NT:b * NT + 1].mean(0)
        g_end = (S_5T / S_T.clamp(min=1e-12)).numpy()
        g_sec = (seg(mult - 1, mult) / seg(1, 2).clamp(min=1e-12)).numpy()
        ok = fin and bool((g_end <= 2).all()) and bool((g_sec <= 2).all())
        out[name] = {"finite": fin, "S_T": S_T.tolist(), "S_5T": S_5T.tolist(), "ratio_end": g_end.tolist(),
                     "ratio_secular": g_sec.tolist(), "absmax": xs.abs().amax((0, 1)).tolist(), "pass": ok}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def one(s):
    torch.set_num_threads(4)
    t0 = time.time()
    cfg, scales, es = setup(s)
    plant = build_plant(cfg["system"])
    sensors = list(plant.sensors); unm = [j for j in range(plant.nx) if j not in sensors]
    if s in B:
        crb = B[s]["dists"]["eval_id"]["bayes"]["full"]["crb_state"]
    else:
        settle = int(0.5 * (es.x.shape[1] - 1))
        Sig = L.climatology(es.x[:, settle:].double())
        alpha = float(cfg["system"]["alpha"])
        b = L.budget_along(plant, es.x, es.u, cfg, scales, sensors, [3 / alpha], n_ep=64, Sigma=Sig)
        crb = b["bayes"]["full"]["crb_state"]
    crb_u = [crb[j] for j in unm]
    o1a = all(math.isfinite(c) and c < 1 for c in crb_u)
    g = gramian(plant, es, cfg, scales)
    o1b = g["frac_full_rank"] >= 0.9
    weak = (max(crb_u) >= 0.1 if crb_u else False) or g["ratio_median"] < 1e-6
    bd = bounded(plant, cfg, scales)
    res = {"plant": s, "unmeasured": unm, "crb_full_state": crb, "crb_unmeasured_max": max(crb_u) if crb_u else 0,
           "obs_1a": o1a, "gramian": g, "obs_1b": o1b, "observable": o1a and o1b, "weak": weak,
           "bounded": bd["pass"], "bounded_detail": bd, "x_std": scales["x_std"], "wall_s": time.time() - t0}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{s}.json").write_text(json.dumps(res, indent=1))
    print(s, "observable", res["observable"], "weak", weak, "bounded", bd["pass"], f"{res['wall_s']:.0f}s", flush=True)


def table():
    rows = ["| plant | observable? | bounded? | evidence: max unmeasured full-past CRB; Gramian median lmin/lmax (frac full rank) | evidence: worst S(5T)/S(T), worst secular ratio (state, input set) |",
            "|---|---|---|---|---|"]
    for s in ALL:
        p = OUT / f"{s}.json"
        if not p.exists():
            rows.append(f"| {s} | - | - | missing | |"); continue
        r = json.loads(p.read_text()); bd = r["bounded_detail"]
        worst = max(((v["ratio_end"][j], j, k) for k, v in bd.items() if k != "pass" for j in range(len(v["ratio_end"]))))
        worst2 = max(((v["ratio_secular"][j], j, k) for k, v in bd.items() if k != "pass" for j in range(len(v["ratio_secular"]))))
        ob = ("yes" + (" (weak)" if r["weak"] else "")) if r["observable"] else "NO"
        g = r["gramian"]
        rows.append(f"| {s} | {ob} | {'yes' if r['bounded'] else 'NO'} | {r['crb_unmeasured_max']:.4f}; {g['ratio_median']:.1e} ({100 * g['frac_full_rank']:.0f}%) | "
                    f"{worst[0]:.2f} (x{worst[1] + 1}, {worst[2]}); {worst2[0]:.2f} (x{worst2[1] + 1}, {worst2[2]}) |")
    print("\n".join(rows))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "table":
        table(); sys.exit()
    for s in a or ALL:
        one(s)
