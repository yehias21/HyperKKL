"""Mechanical scoring of the dose-response predictions P3a-P3d (PREREG.md) and assembly of RESULTS.md
from RESULTS_core.md (score.py), the dose JSONs (dose/) and real_battery.json.

    .venv/bin/python experiments/information/score_dose.py
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
TOL = 0.95
PLANTS = ["quad_tank", "vdp", "two_link_arm"]
SNAME = {"quad_tank": "window-starved", "vdp": "rich", "two_link_arm": "absolutely starved"}


SUMMARY = """## 0. Summary (written by the agent; everything below it is mechanical)

* **What was computed.** For each of the 12 plants, the information about the current state that the measurements
  and inputs carry (bits beyond climatology) and the implied floor on any observer's NRMSE per state, from the
  Bayesian posterior Cramer-Rao recursion (a theorem for Gaussian noises; `infolib.py`) along the cached evaluation
  episodes, for windows 1-10/alpha and the full past (`budgets.json`); the same on the training episodes (equal to
  within 0.3 bits everywhere). The local (EKF-covariance) recursion is reported next to it.
* **Guarantee (P0).** 448 (method, tier) cells: no learned observer and no grey-box filter is below any floor. 3 cells
  of the exact-model EKF on slow states (quad_tank x3, cartpole x2/total) are 5-8% below the full-past floor, each
  inside the bootstrap 95% CI of the achieved pooled RMS (per-episode CV 0.3-0.5). By the pre-registered 5% rule
  this is a FAIL; the honest reading is that the floors are attained by the optimal filter to within sampling error
  and never beaten beyond it. The exact-model EKF sits at 0.95-1.07 of the floor on the near-linear plants.
* **Grouping.** Window-starved (rho >= 1.4: a fading-memory observer loses >= 40% of the attainable precision on the
  unmeasured states): battery, duffing, quad_tank, cartpole, highway_traffic. Rich: bioreactor, lorenz, vdp, fhn,
  network, rossler. Absolutely starved (poor even with unbounded memory): two_link_arm.
* **quad_tank is window-starved, not information-starved in the absolute sense:** the full-past floor on the
  unmeasured tanks is 0.0198 and the grey-box EKF achieves 0.0185-0.0197 (0.95-1.01 of the floor at every
  excitation level): it is information-optimal, and *no observer of any kind can beat it by more than ~1%*. The
  question is why HyperKKL is 3.4x above the floor: the 1/alpha and 3/alpha windows explain a factor 2.6 and 1.5
  (floors 0.052 and 0.030); HyperKKL sits at 1.9-2.6x the 3/alpha floor. The rest is learning-side.
* **Excitation.** For a plant with a known model the state information is a Gramian property: the evaluation-side
  excitation does not change the floor (quad_tank: 21.27 bits at every input time scale from x1/4 to x4), and the
  exact-model EKF's error is flat (P3b PASS 3/3). HyperKKL's error moved by only 1.1-1.3x (P3c FAIL: less than
  predicted; on quad_tank slower inputs lowered it from 0.076 to 0.058 with no change in the budget, a small
  learning-side effect). Training-side enrichment (experiments/quadtank) cannot change the evaluation floor either
  (P4); it can only change coverage. Prediction kept: no arm brings HyperKKL below 0.020 (10/alpha floor).
* **Rank correlations (P1).** HyperKKL's unmeasured error tracks the 3/alpha floor across the 12 plants
  (Spearman 0.65, p = 0.02; 0.76 on sequence OOD): PASS. The exact-model EKF tracks the full-past floor at 0.58
  (FAIL at the 0.7 bar: the Bayesian floor is loose on the chaotic plants); the grey-box does not track it at all
  (-0.05, FAIL): where its residual model is wrong (bioreactor, lorenz, rossler, two_link_arm) it fails outright,
  so its error is not information-limited.
* **Where learned observers lose (P2).** The gap to the best grey-box on the unmeasured states is +464% (median) in
  the window-starved group against -66% in the rich group, 5/5 window-starved plants above the rich median (PASS on
  both tiers); the one unexplained case is fhn (rich, +533%).
* **Cross-check with experiments/quadtank/DIAG.md.** Their no-prior, no-process-noise CRB on the window-start state
  (10 s: h3 0.080, h4 0.045) agrees in magnitude with the 1/alpha floor here (0.066, 0.038); at long windows theirs
  keeps falling (0.009 at 49 s) while the floor here saturates at the process-noise level (0.025, 0.015), which is
  the value the grey-box filter actually reaches. Both analyses conclude observability-/window-limited.
* **Kolmogorov vs Shannon/Fisher.** Kolmogorov complexity is uncomputable (no algorithm computes it, every computable
  lower bound is bounded) and machine-dependent up to a constant, so it cannot yield a per-system number or guarantee.
  The computable content is Fisher/Shannon (above); the honest link is MDL (two-part code: model bits + residual bits,
  the latter bounded below by the rate-distortion bits), which interprets why a structured grey-box needs fewer
  informative samples but adds no quantity. Stated as a remark, not a theorem (theory/m4_information.tex).
* **Noise-model caveat (test plants).** The TEST-C3 runs use coloured Student-t noise on a 30%-perturbed plant; the
  budgets here were computed under the TEST-C1 white-noise exact-plant setting. The post-hoc recomputation
  (`BUDGET_FIX.md`, coordinating agent) gives A = 0.065 (cartpole), 0.097 (highway_traffic), 0.020 (two_link_arm);
  groups and verdicts are unchanged except that two_link_arm becomes "marginal" (see the PREREG addendum).
* **Real battery.** Information floor 0.02-0.11% SOC (sensor-level noise) against achieved 0.7-9%: model error, not
  information, limits every observer on the real cell.
"""


def load(system):
    files = sorted((HERE / "dose").glob(f"{system}_*_seed*.json"))
    return [json.loads(f.read_text()) for f in files]


def agg(runs):
    """Per level: budget (shared), per method mean over seeds of nrmse / unmeasured / rms / rms_state (methods run
    once are taken from the seed that has them)."""
    levels = sorted({lv for r in runs for lv in r["levels"]}, key=float)
    out = {}
    for lv in levels:
        acc = {}
        bud = None
        for r in runs:
            if lv not in r["levels"]:
                continue
            bud = r["levels"][lv]["budget"]
            for m, v in r["levels"][lv]["methods"].items():
                acc.setdefault(m, []).append(v)
        meth = {}
        for m, vs in acc.items():
            meth[m] = {k: float(np.mean([v[k] for v in vs])) for k in ("nrmse", "unmeasured", "measured", "rms")}
            meth[m]["rms_state"] = np.mean([v["rms_state"] for v in vs], 0).tolist()
            meth[m]["n_seeds"] = len(vs)
        out[lv] = {"budget": bud, "methods": meth}
    return out


def main():
    lines = ["", "## 5. P3 - dose-response: evaluation-input excitation (dose/*.json; `dose_response.py`)", ""]
    scores = {}
    any_data = False
    for s in PLANTS:
        runs = load(s)
        if not runs:
            lines.append(f"**{s}**: no dose-response results yet (runs pending).")
            continue
        any_data = True
        alpha = runs[0]["alpha"]; k3 = f"{3 / alpha:g}"
        D = agg(runs)
        levels = list(D)
        seeds = sorted({r["seed"] for r in runs})
        lines += [f"### {s} ({SNAME[s]}; seeds {seeds}; {len(levels)} levels)", "",
                  "| s | slow-band frac | mean switch (s) | bits full | bits 3/a | unm lb full | unm lb 3/a | " +
                  " | ".join(f"{m} unm" for m in ("lora", "ekf_full", "ekf", "greybox_ekf", "greybox_ukf")) + " |",
                  "|---|---|---|---|---|---|---|" + "---|" * 5]
        bfull, b3, e_full, e_lora, e_gb = [], [], [], [], []
        viol = []
        for lv in levels:
            b = D[lv]["budget"]; bb = b["bayes"]; me = D[lv]["methods"]
            bf, w3 = bb["full"], bb["windows"][k3]
            bfull.append(bf["unmeasured_lb"]); b3.append(w3["unmeasured_lb"])
            e_full.append(me.get("ekf_full", {}).get("unmeasured", float("nan")))
            e_lora.append(me.get("lora", {}).get("unmeasured", float("nan")))
            gbs = [me[m]["unmeasured"] for m in ("greybox_ekf", "greybox_ukf") if m in me]
            e_gb.append(min(gbs) if gbs else float("nan"))
            f = lambda m: f"{me[m]['unmeasured']:.4f}" if m in me else "-"
            lines.append(f"| {lv} | {b['input']['slow_band_fraction']:.3f} | {b['input']['mean_switch_interval_s']:.1f} | {bf['bits']:.2f} | {w3['bits']:.2f} | "
                         f"{bf['unmeasured_lb']:.4f} | {w3['unmeasured_lb']:.4f} | " + " | ".join(f(m) for m in ("lora", "ekf_full", "ekf", "greybox_ekf", "greybox_ukf")) + " |")
            # P3a: pooled rms per state and total above 0.95 x full-past bound
            for m, v in me.items():
                if v["rms"] < TOL * bf["nrmse_lb"]:
                    viol.append((s, lv, m, "total", v["rms"], bf["nrmse_lb"]))
                for j, (e, bj) in enumerate(zip(v["rms_state"], bf["crb_state"])):
                    if e < TOL * bj:
                        viol.append((s, lv, m, f"x{j + 1}", e, bj))
        lines.append("")
        p3a = len(viol) == 0
        lines.append(f"**P3a ({s}): {'PASS' if p3a else 'FAIL'}** - {len(viol)} violations of `rms >= 0.95 x full-past bound` over "
                     f"{sum(len(D[lv]['methods']) for lv in levels)} (level, method) cells.")
        if viol:
            lines += ["", "| level | method | quantity | achieved rms | bound |", "|---|---|---|---|---|"]
            lines += [f"| {v[1]} | {v[2]} | {v[3]} | {v[4]:.4f} | {v[5]:.4f} |" for v in viol]
        # P3b
        var_b = max(bfull) / min(bfull)
        var_e = (np.nanmax(e_full) / np.nanmin(e_full)) if np.isfinite(e_full).all() else float("nan")
        if var_b >= 1.2:
            rho = spearmanr(bfull, e_full).statistic
            p3b = rho >= 0.6
            txt = f"budget varies x{var_b:.2f} (>= 1.2): Spearman(bound, ekf_full) = {rho:.2f} (need >= 0.6)"
        else:
            rho = float("nan"); p3b = var_e < 1.3
            txt = f"budget flat (max/min x{var_b:.3f} < 1.2): ekf_full error max/min x{var_e:.2f} (need < 1.3)"
        lines.append(f"**P3b ({s}): {'PASS' if p3b else 'FAIL'}** - {txt}.")
        # P3c
        var_l = np.nanmax(e_lora) / np.nanmin(e_lora)
        i_min = int(np.nanargmin(e_lora)); i_one = levels.index("1")
        near_one = abs(i_min - i_one) <= 1
        p3c_ratio = var_l >= 1.5 * var_e
        lines.append(f"**P3c ({s})**: HyperKKL error max/min x{var_l:.2f} vs ekf_full x{var_e:.2f} -> ratio criterion (>= 1.5x) "
                     f"{'met' if p3c_ratio else 'NOT met'}; HyperKKL minimum at s = {levels[i_min]} ({'at/next to' if near_one else 'away from'} s = 1).")
        # P3d (quad_tank)
        if s == "quad_tank":
            r3 = [e / b for e, b in zip(e_lora, b3)]; rf = [e / b for e, b in zip(e_lora, bfull)]
            lines.append(f"**P3d (quad_tank)**: HyperKKL unm / A(3/a) over levels: {np.round(r3, 2).tolist()} (within x2 of each other: "
                         f"{'yes' if max(r3) / min(r3) <= 2 else 'no'}); HyperKKL unm / full-past bound: {np.round(rf, 2).tolist()}; "
                         f"best grey-box / full-past bound: {np.round([e / b for e, b in zip(e_gb, bfull)], 2).tolist()}.")
        scores[s] = {"P3a": p3a, "violations": viol, "P3b": bool(p3b), "P3b_detail": txt, "P3c_ratio": bool(p3c_ratio),
                     "P3c_min_level": levels[i_min], "P3c_near_one": bool(near_one), "levels": levels,
                     "bound_full_unm": bfull, "bound_3a_unm": b3, "ekf_full_unm": e_full, "lora_unm": e_lora, "best_greybox_unm": e_gb,
                     "n_seeds": len(seeds)}
        lines.append("")
    if any_data:
        n_plants = [s for s in PLANTS if s in scores]
        p3c_pass = sum(scores[s]["P3c_ratio"] for s in n_plants) == len(n_plants) and sum(scores[s]["P3c_near_one"] for s in n_plants) >= 2
        lines.append(f"**P3c overall: {'PASS' if p3c_pass else 'FAIL'}** (ratio criterion on every plant and minimum at/next to s = 1 on >= 2 of 3).")
        scores["P3c_overall"] = bool(p3c_pass)
    # real battery
    lines += ["", "## 6. Real battery (Panasonic 18650PF; `real_battery.py`)", ""]
    rb = HERE / "real_battery.json"
    if rb.exists():
        R = json.loads(rb.read_text())
        lines += ["SOC RMSE floor (% SOC) from the identified 2-RC model driven by the measured current, 30-min windows, settled half; "
                  "(i) the tuned EKF's own (Q, R) [R_vm = 0.1 V], (ii) sensor-level noise [R_vm = 3 mV]; achieved settled SOC RMSE of the real-data run alongside.", "",
                  "| cycle | windows | floor (i) local | floor (i) Bayes | floor (ii) local | floor (ii) Bayes | HyperKKL | EKF | UKF | grey-box EKF | grey-box UKF |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for cyc, c in R["cycles"].items():
            a = R["achieved_soc_rmse_pct"].get(cyc, {})
            if not a and cyc.startswith("HWFT"):          # the real-data tables pool HWFTa/HWFTb as HWFET
                try:
                    import json as _j
                    t = _j.loads((HERE.parents[1] / "experiments" / "real" / "results.json").read_text())["tables"]["test25"]["HWFET"]
                    a = {m: v["rmse_settled"]["mean"] for m, v in t.items() if "rmse_settled" in v}
                except Exception:
                    a = {}
            g = lambda m: f"{a[m]:.2f}" if m in a else "-"
            t, sl = c["soc_rmse_bound_pct"]["tuned_ekf"], c["soc_rmse_bound_pct"]["sensor_level"]
            lines.append(f"| {cyc} | {c['n_windows']} | {t['local_pooled']:.2f} | {t['bayes']:.2f} | {sl['local_pooled']:.3f} | {sl['bayes']:.3f} | "
                         f"{g('lora') if 'lora' in a else g('hyperkkl')} | {g('ekf')} | {g('ukf')} | {g('greybox_ekf')} | {g('greybox_ukf')} |")
        lines.append("")
        lines.append("Reading: setting (ii) is the information floor if the identified model were right; every achieved error is 7-100x above it, so on the "
                     "real cell the error is model error (the ECM), not a shortage of information. Setting (i) is the tuned filter's own belief (inflated Q, R), "
                     "not a bound: HyperKKL is below its local value on NN and Cycle_4. Both are statements about the model; neither is used as a guarantee.")
    else:
        lines.append("pending")
    (HERE / "scores_dose.json").write_text(json.dumps(scores, indent=1, default=float))
    core = (HERE / "RESULTS_core.md").read_text()
    full = core.replace("(rules: `PREREG.md`). Nothing hand-edited.", "(rules: `PREREG.md`). Nothing hand-edited except section 0.", 1)
    full = full.replace("## 1. Per-system", SUMMARY + "\n## 1. Per-system", 1)
    full = full + "\n" + "\n".join(lines) + f"\n\n---\nAssembled {time.strftime('%Y-%m-%d %H:%M %Z')} by `score_dose.py`.\n"
    (HERE / "RESULTS.md").write_text(full)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
