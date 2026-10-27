"""'Coverage, part 3' tables (joint NN coverage, PREREG addendum 4) from diag/coverage/*.json."""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / "diag" / "coverage"
RUNS = [("new", "cartpole", ["A2", "A4"]), ("old", "cartpole", ["lora", "greybox_ukf"]), ("new", "fhn", ["A2"]),
        ("old", "two_link_arm", ["lora"]), ("old", "highway_traffic", ["lora"]), ("camp", "quad_tank", ["A2"]),
        ("camp", "battery", ["A2"])]
TIERS = ("id", "shape_ood", "sequence_ood")
f = lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.4f}"
L = ["| plant | model | tier | null d50 / d99 (val) | eval d50 | eval beyond d99 | unm err near | unm err far | far/near | err by distance quintile (near -> far) | Spearman(d, err) | rule |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
X = ["| plant | x_std | std p1 episodes / x_std | std train episodes / x_std | std collocation / x_std | std eval seq / x_std |", "|---|---|---|---|---|---|"]
R = ["| plant | set | radius about eval-mean (x_std units): q1 / q10 / q50 / q90 / q99 |", "|---|---|---|"]
summ = {}
for which, system, models in RUNS:
    js = [json.loads(p.read_text()) for p in sorted(D.glob(f"{which}_{system}_s?.json"))]
    js = [j for j in js if "joint" in j]
    if len(js) < 3:
        L.append(f"| {system} | {models} | - | missing ({len(js)} seeds) |"); continue
    for m in models:
        for t in TIERS:
            null = np.mean([j["joint"]["null_q50_q99"] for j in js], 0)
            fb = np.mean([j["tiers"][t]["joint"]["frac_beyond_d99"] for j in js])
            d50 = np.mean([j["tiers"][t]["joint"]["d_q50_q99"][0] for j in js])
            g = lambda lab: [j["tiers"][t]["methods"][m][lab]["unm"] for j in js if j["tiers"][t]["methods"][m].get(lab)]
            near = np.mean(g("near")) if g("near") else np.nan
            far = np.mean(g("far")) if g("far") else np.nan
            ratio = far / near if np.isfinite(far) else np.nan
            bins = np.mean([j["tiers"][t]["methods"][m]["bins_unm"] for j in js], 0)
            rho = np.mean([j["tiers"][t]["methods"][m]["spearman_d_err"] for j in js])
            v = "coverage-limited" if (fb > 0.10 and np.isfinite(ratio) and ratio >= 2) else "no"
            summ.setdefault(f"{system}/{m}", {})[t] = {"frac_beyond": fb, "near": near, "far": far, "rho": rho, "verdict": v}
            L.append(f"| {system} | {m} | {t} | {null[0]:.4f} / {null[1]:.4f} | {d50:.4f} | {fb:.1%} | {f(near)} | {f(far)} | "
                     f"{'-' if not np.isfinite(ratio) else f'{ratio:.2f}'} | {' / '.join(f'{b:.3f}' for b in bins)} | {rho:+.2f} | {v} |")
    if which != "old" or system != "cartpole":
        j0 = js[0]; sig = np.asarray(j0["x_std"])
        st = j0["joint"]["std"]
        fs = lambda a: "[" + ", ".join(f"{x:.2f}" for x in np.asarray(a) / sig) + "]"
        X.append(f"| {system} | [{', '.join(f'{x:.3g}' for x in sig)}] | {fs(st['p1'])} | {fs(st['train'])} | {fs(st['coll'])} | {fs(j0['tiers']['sequence_ood']['joint']['eval_std'])} |")
        rq = j0["tiers"]["sequence_ood"]["radius_q_about_eval_mean"]
        for k in ("eval", "p1", "train", "coll"):
            R.append(f"| {system} | {k} | {' / '.join(f'{x:.2f}' for x in rq[k])} |")
out = ["Joint coverage (mean over seeds 0-2; d = NN distance to the training states in x_std units; null = validation episodes):", ""] + L + \
      ["", "Exploration: per-state std relative to x_std (seed 0):", ""] + X + ["", "Radius about the eval-state mean (sequence_ood, seed 0):", ""] + R
(D / "tables_part3.md").write_text("\n".join(out))
(D / "summary_part3.json").write_text(json.dumps(summ, indent=1))
print("\n".join(out))
