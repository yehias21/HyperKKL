"""Pre-registered INCLUSION check (experiments/cartpole_bounded/INCLUSION.md rules) for the case-study plants. CPU.

    CUDA_VISIBLE_DEVICES="" python inclusion_cs.py quad_planar cstr_exo

(1) OBSERVABLE: (1a) full-past Bayesian PCRB of every unmeasured state < 1 (sigma-normalised) and
    (1b) linearised observability Gramian over 3/alpha full rank (lmin/lmax > 1e-10) at >= 90 % of 12 x 32
    (episode, time) points; (2) BOUNDED: 5T rollouts, S_j(5T) <= 2 S_j(T) and no secular growth, all
    four input sets. Exactly the functions of experiments/cartpole_bounded/inclusion.py.
Settings: "rule" = the literal INCLUSION.md setting (exact model, white Gaussian 0.03 y_std on every output at
every step, process noise 0.005); "case" = the case's own exact-level sensor model (per-output physical noise,
multi-rate hold -- Gramian with held outputs only at fresh samples, PCRB via infolib_cs) and its process noise.
BOUNDED is also run on the 30 % mismatched true plant (used by the mm30 / realistic levels).
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(C.LIVE / "experiments" / "cartpole_bounded"))
import infolib_cs as I  # noqa: E402
import inclusion as INC  # noqa: E402  (gramian, bounded: unchanged functions)
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl.config import deep_merge  # noqa: E402
from hyperkkl.systems import build_plant  # noqa: E402

INC.Dm = Dm                                   # the package copy's data module (local cache)
OUT = C.HERE / "inclusion"
CASE_OF = {v["system"]: k for k, v in C.CASES.items()}


def gramian_hold(plant, es, cfg, scales, hold, n_ep=32, n_t=12):
    """INC.gramian with held outputs contributing only at fresh samples (k % P == 0)."""
    alpha = float(cfg["system"]["alpha"]); dt = float(scales["dt"])
    x, u = es.x[:n_ep].double(), es.u[:n_ep].double()
    N1 = x.shape[1]; settle = int(0.5 * (N1 - 1))
    W = min(int(round(3 / alpha / dt)), settle)
    F, H = I.L.jacobians(plant, x, u, dt)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float64)
    D, Di = torch.diag(sig), torch.diag(1 / sig)
    _, s, _, _ = None, None, None, None
    rho, Jt, s, hold = I.sensor_terms(cfg, scales)
    Rinv = torch.diag(torch.as_tensor(1.0 / s ** 2))
    ks = torch.linspace(settle, N1 - 1, n_t).round().long()
    ratios = []
    for k in ks.tolist():
        k0 = k - W
        Phi = torch.eye(plant.nx, dtype=torch.float64).expand(x.shape[0], -1, -1).clone()
        Wo = torch.zeros_like(Phi)
        for i in range(W + 1):
            fresh = torch.as_tensor([1.0 if (k0 + i) % int(P) == 0 else 0.0 for P in hold], dtype=torch.float64)
            Hn = (H[:, k0 + i] * fresh[None, :, None]) @ D
            Pn = Di @ Phi @ D
            Wo = Wo + Pn.transpose(-1, -2) @ Hn.transpose(-1, -2) @ Rinv @ Hn @ Pn
            if i < W:
                Phi = F[:, k0 + i] @ Phi
        ev = torch.linalg.eigvalsh(0.5 * (Wo + Wo.transpose(-1, -2)))
        ratios.append((ev[:, 0].clamp(min=0) / ev[:, -1]).numpy())
    r = np.concatenate(ratios)
    return {"W_samples": W, "ratio_median": float(np.median(r)), "ratio_p10": float(np.percentile(r, 10)),
            "frac_full_rank": float((r > 1e-10).mean()), "cond_median": float(1 / max(np.median(r), 1e-300))}


def one(system):
    torch.set_num_threads(8)
    t0 = time.time()
    case = CASE_OF[system]
    res = {"plant": system, "case": case, "settings": {}}
    for setting in ("rule", "case"):
        cfg = C.case_cfg(case, "exact", 0, "cpu")
        if setting == "rule":
            cfg = deep_merge(cfg, {"data": {"noise_proc": 0.005}})
            cfg["data"]["sensor"] = {"noise": 0.03}
            cfg["data"].pop("labels", None)
        scales = Dm.system_scales(cfg, "cpu")
        es = Dm.eval_data(cfg, "id", device="cpu")
        plant = build_plant(cfg["system"])
        sensors = list(plant.sensors); unm = [j for j in range(plant.nx) if j not in sensors]
        settle = int(0.5 * (es.x.shape[1] - 1))
        Sig = I.L.climatology(es.x[:, settle:].double())
        alpha = float(cfg["system"]["alpha"])
        b = I.budget(plant, es.x, es.u, cfg, scales, sensors, [3 / alpha], Sig, n_ep=48)
        crb = b["bayes"]["full"]["crb_state"]
        crb_u = [crb[j] for j in unm]
        o1a = all(math.isfinite(c) and c < 1 for c in crb_u)
        g = INC.gramian(plant, es, cfg, scales) if setting == "rule" else \
            gramian_hold(plant, es, cfg, scales, cfg["data"]["sensor"].get("hold"))
        o1b = g["frac_full_rank"] >= 0.9
        weak = (max(crb_u) >= 0.1 if crb_u else False) or g["ratio_median"] < 1e-6
        bd = {"mm0": INC.bounded(plant, cfg, scales),
              "mm30": INC.bounded(build_plant(cfg["system"], perturb=0.3), cfg, scales)}
        res["settings"][setting] = {
            "unmeasured": unm, "crb_full_state": crb, "crb_unmeasured_max": max(crb_u),
            "A_3a_unmeasured": b["bayes"]["windows"][f"{3 / alpha:g}"]["unmeasured_lb"],
            "obs_1a": o1a, "gramian": g, "obs_1b": o1b, "observable": o1a and o1b, "weak": weak,
            "bounded": bd["mm0"]["pass"], "bounded_mm30": bd["mm30"]["pass"], "bounded_detail": bd,
            "x_std": scales["x_std"], "sensor": cfg["data"]["sensor"]}
        print(system, setting, "observable", o1a and o1b, "weak", weak, "crb_u", [round(c, 4) for c in crb_u],
              "gram", f"{g['ratio_median']:.1e}/{g['frac_full_rank']:.2f}",
              "bounded", bd["mm0"]["pass"], "bounded_mm30", bd["mm30"]["pass"], f"{time.time() - t0:.0f}s", flush=True)
    res["wall_s"] = time.time() - t0
    OUT.mkdir(exist_ok=True)
    (OUT / f"{system}.json").write_text(json.dumps(res, indent=1, default=C.jdefault))


def table():
    rows = ["| plant | setting | observable? | bounded? (mm0 / mm30) | max unmeasured full-past CRB; Gramian median lmin/lmax (frac full rank) | worst S(5T)/S(T); worst secular ratio (state, input set) |",
            "|---|---|---|---|---|---|"]
    for s in ("quad_planar", "cstr_exo"):
        p = OUT / f"{s}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for st, v in r["settings"].items():
            bd = v["bounded_detail"]["mm0"]
            worst = max(((w["ratio_end"][j], j, k) for k, w in bd.items() if k != "pass" for j in range(len(w["ratio_end"]))))
            worst2 = max(((w["ratio_secular"][j], j, k) for k, w in bd.items() if k != "pass" for j in range(len(w["ratio_secular"]))))
            ob = ("yes" + (" (weak)" if v["weak"] else "")) if v["observable"] else "NO"
            g = v["gramian"]
            rows.append(f"| {s} | {st} | {ob} | {'yes' if v['bounded'] else 'NO'} / {'yes' if v['bounded_mm30'] else 'NO'} | "
                        f"{v['crb_unmeasured_max']:.4f}; {g['ratio_median']:.1e} ({100 * g['frac_full_rank']:.0f}%) | "
                        f"{worst[0]:.2f} (x{worst[1] + 1}, {worst[2]}); {worst2[0]:.2f} (x{worst2[1] + 1}, {worst2[2]}) |")
    print("\n".join(rows))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "table":
        table(); sys.exit()
    for s in a:
        one(s)
