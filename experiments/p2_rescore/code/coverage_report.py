"""Tables for 'Cartpole diagnosis, part 2' from diag/coverage/*.json -> diag/coverage/tables.md"""
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / "diag" / "coverage"
RUNS = [("new", "cartpole", ["A2", "A4"]), ("old", "cartpole", ["lora", "greybox_ukf"]),
        ("old", "highway_traffic", ["lora"]), ("camp", "quad_tank", ["A2"]), ("camp", "battery", ["A2"])]
TIERS = ("id", "shape_ood", "sequence_ood")
L = ["| plant | model | tier | joint outside min/max | unm err inside | unm err outside | ratio | rule | outside 1-99% band (secondary) | unm err in band | unm err out of band | x3 in / out band | x4 in / out band |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
summ = {}
env_lines = []
for which, system, models in RUNS:
    js = [json.loads(p.read_text()) for p in sorted(D.glob(f"{which}_{system}_s*.json"))]
    if not js:
        continue
    if which != "old" or system != "cartpole":
        j0 = js[0]
        env_lines.append(f"- {system} (seed 0): envelope lo {np.round(j0['envelope']['lo'], 3).tolist()} hi {np.round(j0['envelope']['hi'], 3).tolist()}; "
                         f"train 1/99% {np.round(j0['train_q'][0], 3).tolist()} / {np.round(j0['train_q'][2], 3).tolist()}; "
                         f"seq eval 1/99% {np.round(j0['tiers']['sequence_ood']['eval_q'][0], 3).tolist()} / {np.round(j0['tiers']['sequence_ood']['eval_q'][2], 3).tolist()}; "
                         f"max |f(x + d e1) - f(x)| for d = 10, 100: {j0['f_shift_x1_maxabs']} (|f| max {j0['f_scale']:.3g})")
    for m in models:
        for t in TIERS:
            fo = np.mean([j["tiers"][t]["frac_out_joint"] for j in js])
            fq = np.mean([j["tiers"][t]["frac_out_q01_q99_joint"] for j in js])
            g = lambda lab, k="unm": [j["tiers"][t]["methods"][m][lab] for j in js if j["tiers"][t]["methods"][m][lab] is not None]
            def mean(lab, idx=None):
                v = g(lab)
                if not v:
                    return float("nan")
                return float(np.mean([x["unm"] if idx is None else x["rms_state"][idx] for x in v]))
            ei, eo = mean("inside"), mean("outside")
            ratio = eo / ei if np.isfinite(eo) else float("nan")
            verdict = "extrapolation" if (fo > 0.30 and np.isfinite(ratio) and ratio >= 3) else "no"
            summ.setdefault(f"{system}/{m}", {})[t] = {"frac_out": fo, "err_in": ei, "err_out": eo, "verdict": verdict, "frac_out_q": fq}
            x34 = f"{mean('in_q', 2):.3f} / {mean('out_q', 2):.3f} | {mean('in_q', 3):.3f} / {mean('out_q', 3):.3f}" if system == "cartpole" else "- | -"
            L.append(f"| {system} | {m} | {t} | {fo:.1%} | {ei:.3f} | {eo:.3f} | {ratio:.2f} | {verdict} | {fq:.1%} | {mean('in_q'):.3f} | {mean('out_q'):.3f} | {x34} |")
drift = []
for which, system, _ in RUNS[:1] + RUNS[2:]:
    js = [json.loads(p.read_text()) for p in sorted(D.glob(f"{which}_{system}_s*.json"))]
    if js:
        for t in TIERS:
            a = js[0]["tiers"][t]["x1_abs_q50_q99_at"]
            drift.append(f"| {system} | {t} | " + " | ".join(f"{a[k][0]:.3g} / {a[k][1]:.3g}" for k in ("t0", "tmid", "tend")) + " |")
out = ["Coverage (settled eval steps, mean over seeds 0-2; errors = RMS |xhat - x| / x_std, unmeasured = mean over unmeasured states):", ""] + L + \
      ["", "Envelopes:", ""] + env_lines + ["", "|x1| (cart position for cartpole) median / 99% at t = 0, T/2, T (seed 0):", "",
                                            "| plant | tier | t0 | T/2 | T |", "|---|---|---|---|---|"] + drift
(D / "tables.md").write_text("\n".join(out))
(D / "summary.json").write_text(json.dumps(summ, indent=1))
print("\n".join(out))
