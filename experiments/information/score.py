"""Mechanical scoring of the pre-registered predictions (PREREG.md) -> RESULTS.md, scores.json.

    .venv/bin/python experiments/information/score.py
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infolib as L  # noqa: E402

B = json.loads((HERE / "budgets.json").read_text())
A = json.loads((HERE / "achieved.json").read_text())
GREYBOX = ("greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")
TOL = 0.95
GROUPS = {"rich": [], "window-starved": [], "absolutely starved": [], "marginal": []}


def budget(s, dist="eval_id"):
    r = B[s]; d = r["dists"][dist]; a = r["alpha"]
    w = d["bayes"]["windows"]
    return {"A": w[f"{3 / a:g}"]["unmeasured_lb"], "full_unm": d["bayes"]["full"]["unmeasured_lb"],
            "full_tot": d["bayes"]["full"]["nrmse_lb"], "full_state": d["bayes"]["full"]["crb_state"],
            "bits_full": d["bayes"]["full"]["bits"], "bits3": w[f"{3 / a:g}"]["bits"], "bits1": w[f"{1 / a:g}"]["bits"],
            "A1": w[f"{1 / a:g}"]["unmeasured_lb"], "A10": w[f"{10 / a:g}"]["unmeasured_lb"],
            "tot3": w[f"{3 / a:g}"]["nrmse_lb"], "rd_full": d["bayes"]["full"]["rd_lb"],
            "local_full_unm": d["local"]["full"]["unmeasured_lb"], "local_full_state": d["local"]["full"]["crb_state"],
            "rho": w[f"{3 / a:g}"]["unmeasured_lb"] / d["bayes"]["full"]["unmeasured_lb"]}


def group_of(bq):
    if bq["A"] < 0.02 and bq["rho"] < 1.25:
        return "rich"
    if bq["rho"] >= 1.4:
        return "window-starved"
    if bq["A"] >= 0.03:
        return "absolutely starved"
    return "marginal"


def best_greybox(s, tier, key="unmeasured"):
    ms = A[s]["partial"]["methods"]
    cands = {m: ms[m][tier][key] for m in GREYBOX if m in ms and tier in ms[m]}
    if not cands:
        return None, float("nan")
    m = min(cands, key=cands.get)
    return m, cands[m]


def main():
    lines = ["# Information budget vs achieved error: results", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `score.py` from `budgets.json` and `achieved.json` "
             "(rules: `PREREG.md`). Nothing hand-edited.", ""]
    scores = {}
    groups = {}
    # ---- table 1: budgets and groups
    lines += ["## 1. Per-system information budget (eval id, Bayesian PCRB) and group", "",
              "| system | n_x | n_y | bits full | bits 3/a | bits 1/a | unm lb full | unm lb 3/a (A) | unm lb 1/a | rho | total lb full | RD lb full | group |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in L.ALL:
        bq = budget(s); g = group_of(bq); groups[s] = g; GROUPS[g].append(s)
        lines.append(f"| {s} | {B[s]['nx']} | {B[s]['ny']} | {bq['bits_full']:.1f} | {bq['bits3']:.1f} | {bq['bits1']:.1f} | "
                     f"{bq['full_unm']:.4f} | {bq['A']:.4f} | {bq['A1']:.4f} | {bq['rho']:.2f} | {bq['full_tot']:.4f} | {bq['rd_full']:.4f} | {g} |")
    lines += ["", "Groups: " + "; ".join(f"**{g}**: {', '.join(v) or 'none'}" for g, v in GROUPS.items()), ""]
    scores["groups"] = groups

    # ---- P0: guarantee check
    lines += ["## 2. P0 - every achieved error above the full-past bound (tolerance 5%)", ""]
    viol, checked, tight = [], 0, {}
    for s in L.ALL:
        bq = budget(s)
        for know in ("full", "partial"):
            for m, tiers in A[s][know]["methods"].items():
                for tier, v in tiers.items():
                    checked += 1
                    if v["rms"] < TOL * bq["full_tot"]:
                        viol.append((s, know, m, tier, "total", v["rms"], bq["full_tot"]))
                    for j, (e, b) in enumerate(zip(v["rms_state"], bq["full_state"])):
                        if e < TOL * b:
                            viol.append((s, know, m, tier, f"x{j + 1}", e, b))
        ekf = A[s]["full"]["methods"].get("ekf", {}).get("id")
        gbm, _ = best_greybox(s, "id", "rms")
        gb = A[s]["partial"]["methods"][gbm]["id"] if gbm else None
        lora = A[s]["full"]["methods"].get("lora", {}).get("id")
        tight[s] = {"ekf_over_bound_total": ekf["rms"] / bq["full_tot"] if ekf else None,
                    "ekf_over_bound_unmeasured": (np.mean([ekf["rms_state"][j] for j in range(B[s]["nx"]) if j not in B[s]["sensors"]]) / bq["full_unm"]) if ekf else None,
                    "greybox_over_bound_total": gb["rms"] / bq["full_tot"] if gb else None,
                    "greybox": gbm,
                    "lora_over_bound_total": lora["rms"] / bq["full_tot"] if lora else None,
                    "lora_over_A": (lora["unmeasured"] / bq["A"]) if lora else None,
                    "lora_over_bound_unmeasured": (lora["unmeasured"] / bq["full_unm"]) if lora else None}
    p0 = len(viol) == 0
    scores["P0"] = {"pass": p0, "checked_method_tier_cells": checked, "violations": viol}
    lines.append(f"**P0: {'PASS' if p0 else 'FAIL'}** - {checked} (system, knowledge, method, tier) cells, {len(viol)} violations "
                 f"of `rms >= 0.95 x bound` (total or any state).")
    if viol:
        lines += ["", "| system | runs | method | tier | quantity | achieved rms | bound |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {v[0]} | {v[1]} | {v[2]} | {v[3]} | {v[4]} | {v[5]:.4f} | {v[6]:.4f} |" for v in viol]
    lines += ["", "Tightness (tier id, pooled RMS / bound; unmeasured = mean per-state ratio):", "",
              "| system | EKF exact / bound (total) | EKF exact / bound (unmeasured) | best grey-box / bound (total) | HyperKKL / bound (total) | HyperKKL unm / A(3/a) | HyperKKL unm / bound full |",
              "|---|---|---|---|---|---|---|"]
    for s in L.ALL:
        t = tight[s]
        f = lambda v: "-" if v is None else f"{v:.2f}"
        lines.append(f"| {s} | {f(t['ekf_over_bound_total'])} | {f(t['ekf_over_bound_unmeasured'])} | {f(t['greybox_over_bound_total'])} ({t['greybox']}) | "
                     f"{f(t['lora_over_bound_total'])} | {f(t['lora_over_A'])} | {f(t['lora_over_bound_unmeasured'])} |")
    scores["tightness"] = tight

    # ---- P1: rank correlations
    lines += ["", "## 3. P1 - rank correlation across the 12 plants", ""]
    rows = []
    for s in L.ALL:
        bq = budget(s)
        lora = A[s]["full"]["methods"].get("lora", {}); ekf = A[s]["full"]["methods"].get("ekf", {})
        gbm, gbv = best_greybox(s, "id"); _, gbv_seq = best_greybox(s, "sequence_ood")
        rows.append({"system": s, "A": bq["A"], "full_unm": bq["full_unm"], "bits3": bq["bits3"], "full_tot": bq["full_tot"],
                     "lora_unm_id": lora.get("id", {}).get("unmeasured"), "lora_tot_id": lora.get("id", {}).get("nrmse"),
                     "lora_unm_seq": lora.get("sequence_ood", {}).get("unmeasured"),
                     "ekf_unm_id": ekf.get("id", {}).get("unmeasured"), "ekf_tot_id": ekf.get("id", {}).get("nrmse"),
                     "gb_unm_id": gbv, "gb_unm_seq": gbv_seq, "gb": gbm, "group": groups[s]})
    scores["rows"] = rows

    def sp(xk, yk, subset=None):
        r = [q for q in rows if q[xk] is not None and q[yk] is not None and not (isinstance(q[yk], float) and math.isnan(q[yk]))
             and (subset is None or q["system"] in subset)]
        if len(r) < 4:
            return float("nan"), float("nan"), len(r)
        res = spearmanr([q[xk] for q in r], [q[yk] for q in r])
        return float(res.statistic), float(res.pvalue), len(r)
    tests = [("P1a", "A", "lora_unm_id", 0.5, "HyperKKL unmeasured (id) vs A = unm lb(3/a)"),
             ("P1b", "full_unm", "ekf_unm_id", 0.7, "exact-model EKF unmeasured (id) vs unm lb(full)"),
             ("P1c", "full_unm", "gb_unm_id", 0.5, "best grey-box unmeasured (id) vs unm lb(full)")]
    lines += ["| test | x | y | n | Spearman rho_s | p (two-sided) | threshold | verdict |", "|---|---|---|---|---|---|---|---|"]
    for name, xk, yk, thr, desc in tests:
        r, p, n = sp(xk, yk)
        ok = (not math.isnan(r)) and r >= thr
        scores[name] = {"rho_s": r, "p": p, "n": n, "threshold": thr, "pass": bool(ok)}
        lines.append(f"| **{name}** | {xk} | {yk} | {n} | {r:.3f} | {p:.3g} | >= {thr} | **{'PASS' if ok else 'FAIL'}** |")
    lines += ["", "Secondary (reported, no verdict):", "", "| x | y | n | rho_s | p |", "|---|---|---|---|---|"]
    no_duff = [s for s in L.ALL if s != "duffing"]
    for xk, yk, sub, tag in [("A", "lora_unm_id", no_duff, "11 plants"), ("bits3", "lora_unm_id", None, ""),
                             ("full_tot", "lora_tot_id", None, ""), ("full_tot", "ekf_tot_id", None, ""),
                             ("A", "lora_unm_seq", None, ""), ("full_unm", "gb_unm_seq", None, ""),
                             ("full_unm", "lora_unm_id", None, "")]:
        r, p, n = sp(xk, yk, sub)
        lines.append(f"| {xk} | {yk} {tag} | {n} | {r:.3f} | {p:.3g} |")
    lines += ["", "Per-plant values used (tier id unless noted):", "",
              "| system | group | A | unm lb full | HyperKKL unm | HyperKKL unm (seq) | EKF exact unm | best grey-box unm | best grey-box unm (seq) |",
              "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
    for q in rows:
        lines.append(f"| {q['system']} | {q['group']} | {q['A']:.4f} | {q['full_unm']:.4f} | {f(q['lora_unm_id'])} | {f(q['lora_unm_seq'])} | "
                     f"{f(q['ekf_unm_id'])} | {f(q['gb_unm_id'])} ({q['gb']}) | {f(q['gb_unm_seq'])} |")

    # ---- P2: gap by group
    lines += ["", "## 4. P2 - the learned-vs-grey-box gap by group (partial-knowledge runs, unmeasured NRMSE)", ""]
    for tier, tag in (("sequence_ood", "primary"), ("id", "secondary")):
        gaps = {}
        for s in L.ALL:
            ms = A[s]["partial"]["methods"]
            if "lora" not in ms or tier not in ms["lora"]:
                continue
            gbm, gbv = best_greybox(s, tier)
            if gbm is None:
                continue
            gaps[s] = (ms["lora"][tier]["unmeasured"] - gbv) / gbv
        ws = [gaps[s] for s in GROUPS["window-starved"] if s in gaps]
        rich = [gaps[s] for s in GROUPS["rich"] if s in gaps]
        med_ws, med_rich = float(np.median(ws)), float(np.median(rich))
        n_above = int(sum(g > med_rich for g in ws))
        ok = med_ws > med_rich and n_above >= 4
        scores[f"P2_{tier}"] = {"pass": bool(ok), "median_gap_window_starved": med_ws, "median_gap_rich": med_rich,
                                "n_window_starved_above_rich_median": n_above, "n_window_starved": len(ws), "gaps": gaps}
        lines.append(f"**P2 ({tier}, {tag}): {'PASS' if ok else 'FAIL'}** - median gap window-starved {med_ws:+.0%} "
                     f"vs rich {med_rich:+.0%}; {n_above}/{len(ws)} window-starved plants above the rich median (need >= 4).")
        lines += ["", "| system | group | gap (lora - best grey-box) / best grey-box |", "|---|---|---|"]
        lines += [f"| {s} | {groups[s]} | {gaps[s]:+.0%} |" for s in L.ALL if s in gaps]
        lines.append("")
    (HERE / "scores.json").write_text(json.dumps(scores, indent=1, default=float))
    (HERE / "RESULTS_core.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
