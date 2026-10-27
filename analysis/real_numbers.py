"""Real-data (battery SOC) numbers for the paper: experiments/real/results.json -> LaTeX macros.

    .venv/bin/python -I -m analysis.real_numbers      # -> paper/generated_real/numbers_real.tex, tab_real.tex

Read-only on experiments/. Every number the paper quotes about the real-data experiment is a macro
written here (never typed by hand); the table is written as booktabs LaTeX. Re-run after
experiments/real/results.json changes (e.g. when the same-data baselines koopman_kf / greybox_*
are added: the macro \\RealSameDataBaselines then switches from "pending" to "present").
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "real" / "results.json"
OUT = ROOT / "paper" / "generated_real"

TEST25 = ("US06", "HWFET", "NN")
METHODS = ("ws_best", "lora", "hyper", "latent", "concat", "autonomous", "ekf", "ukf", "cc")
LABEL = {"ws_best": "WS-best (HyperKKL $r=4$ or dense ablation, chosen per seed on validation)", "lora": "HyperKKL ($r=4$)", "hyper": "dense-subspace ablation",
         "latent": "latent injection", "concat": "concat (additive)", "autonomous": "autonomous (no input)",
         "ekf": "EKF (identified 2-RC ECM, $Q,R$ tuned on validation)", "ukf": "UKF (same model, tuned)",
         "cc": "coulomb counting from the wrong prior",
         "koopman_kf": "Koopman KF (same data)", "koopman_kf_bilinear": "bilinear Koopman KF (same data)",
         "greybox_ekf": "grey-box EKF (ECM + learned residual)", "greybox_ukf": "grey-box UKF",
         "greybox_ekf_oracle": "grey-box EKF, oracle-tuned"}
LEARNED = {"ws_best", "lora", "hyper", "latent", "concat", "autonomous"}
TRAIN_CYCLES = ("25/Cycle_1", "25/Cycle_2", "25/Cycle_3", "25/UDDS", "25/LA92")
VALTEST_CYCLES = ("25/Cycle_4", "25/US06", "25/HWFTa", "25/HWFTb", "25/NN")


def _name(s: str) -> str:
    """Macro-safe CamelCase (letters only)."""
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    words = []
    for w in s.split():
        if w.isdigit():
            w = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "25": "TwentyFive",
                 "06": "ZeroSix", "92": "NineTwo"}.get(w, "".join({"0": "Zero", "1": "One", "2": "Two", "3": "Three",
                                                                 "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
                                                                 "8": "Eight", "9": "Nine"}[c] for c in w))
        else:
            w = re.sub(r"\d", lambda m: {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
                                         "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}[m.group()], w)
        words.append(w[:1].upper() + w[1:])
    return "".join(words)


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def signed(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}"


def main() -> None:
    d = json.loads(SRC.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    M: dict[str, str] = {}

    def put(name: str, value: str) -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        M[name] = value

    # ---- settled RMSE tables (25 degC test cycles + validation), %
    t25 = d["tables"]["test25"]
    present = set()
    for cyc in TEST25:
        for m, v in t25[cyc].items():
            present.add(m)
            r = v["rmse_settled"]
            put(f"Real{_name(m)}{_name(cyc)}Rmse", fmt(r["mean"]))
            if "sd" in r:
                put(f"Real{_name(m)}{_name(cyc)}RmseSd", fmt(r["sd"]))
            if "max_settled" in v:
                put(f"Real{_name(m)}{_name(cyc)}Max", fmt(v["max_settled"]["mean"], 1))
            if "t_conv_median_min" in v:
                put(f"Real{_name(m)}{_name(cyc)}Tconv", fmt(v["t_conv_median_min"], 1))
    val = d["tables"]["val"]["Cycle_4"]
    for m, v in val.items():
        put(f"Real{_name(m)}ValRmse", fmt(v["rmse_settled"]["mean"]))
    put("RealSameDataBaselines", "present" if any(k.startswith(("koopman", "greybox")) for k in present) else "pending")

    # ---- pre-registered decisions
    dec = d["decision"]
    put("RealHOneLearnedBetter", str(dec["H1_ws_best_vs_ekf"]["cycles_learned_better"]))
    put("RealHOneEkfBetter", str(dec["H1_ws_best_vs_ekf"]["cycles_ekf_better"]))
    put("RealHTwoWsBetter", str(dec["H2_ws_best_vs_latent"]["cycles_ws_better"]))
    put("RealHTwoLatentBetter", str(dec["H2_ws_best_vs_latent"]["cycles_latent_better"]))
    put("RealWsChoice", ", ".join(f"seed {k}: {v}" for k, v in dec["ws_choice_per_seed"].items()))
    put("RealNTestCycles", str(len(TEST25)))

    # ---- paired differences (settled RMSE, % SOC), 25 degC
    for pair in ("ws_best-ekf", "ws_best-ukf", "ws_best-latent", "lora-ekf", "lora-latent", "hyper-ekf", "hyper-latent"):
        if pair not in d["paired"]:
            continue
        P = d["paired"][pair]["test25"]
        # cycles decided for the first-named method by the pre-registered rule (CI excludes 0 and all seeds agree)
        put(f"Real{_name(pair)}CyclesABetter", str(sum(1 for c in TEST25 if P[c].get("a_better"))))
        put(f"Real{_name(pair)}CyclesBBetter", str(sum(1 for c in TEST25 if P[c].get("b_better"))))
        for cyc in TEST25:
            q = P[cyc]
            base = f"Real{_name(pair)}{_name(cyc)}"
            put(base + "Diff", signed(q["mean_diff"]))
            put(base + "CiLo", signed(q["ci95"][0]))
            put(base + "CiHi", signed(q["ci95"][1]))
            put(base + "Seeds", ", ".join(signed(s) for s in q["per_seed_diff"]))
            put(base + "Nwin", str(q["n_windows"]))
        # smallest / largest |diff| over the test cycles
        diffs = [P[c]["mean_diff"] for c in TEST25]
        put(f"Real{_name(pair)}DiffMinAbs", fmt(min(abs(x) for x in diffs)))
        put(f"Real{_name(pair)}DiffMaxAbs", fmt(max(abs(x) for x in diffs)))

    # ---- 0 degC (secondary): per cycle settled RMSE and who is better
    t0 = d["tables"]["test0"]
    P0 = d["paired"]["ws_best-ekf"]["test0"]
    ekf_better = sum(1 for c in P0.values() if c.get("b_better"))
    ws_better = sum(1 for c in P0.values() if c.get("a_better"))
    put("RealZeroNCycles", str(len(P0)))
    put("RealZeroEkfBetter", str(ekf_better))
    put("RealZeroWsBetter", str(ws_better))
    put("RealZeroWsBetterCycles", ", ".join(c for c, q in P0.items() if q.get("a_better")) or "none")
    ws0 = [t0[c]["ws_best"]["rmse_settled"]["mean"] for c in P0]
    ekf0 = [t0[c]["ekf"]["rmse_settled"]["mean"] for c in P0]
    put("RealZeroWsRmseMin", fmt(min(ws0), 1)); put("RealZeroWsRmseMax", fmt(max(ws0), 1))
    put("RealZeroEkfRmseMin", fmt(min(ekf0), 1)); put("RealZeroEkfRmseMax", fmt(max(ekf0), 1))
    P0l = d["paired"]["ws_best-latent"]["test0"]
    put("RealZeroWsLatentTies", str(sum(1 for q in P0l.values() if not q.get("a_better") and not q.get("b_better"))))

    # ---- whole-recording runs (prior 80 %, RMSE over t >= 15 min), 25 degC
    W = d["whole_recordings"]
    for key, v in W.items():
        if not key.startswith("25/"):
            continue
        cyc = key.split("/", 1)[1]
        for m in ("ws_best", "ekf", "ukf"):
            put(f"RealWhole{_name(m)}{_name(cyc)}", fmt(v[m]["rmse"]))

    # ---- identified model and its open-loop fit
    ecm = d["identification"]["ecm"]
    put("RealEcmTauOne", fmt(ecm["tau1_s"], 0)); put("RealEcmTauTwo", fmt(ecm["tau2_s"], 0))
    put("RealEcmRZero", fmt(1e3 * ecm["R0"], 1))
    put("RealQAh", fmt(d["Q_Ah"], 3))
    fit = d["open_loop_model_fit"]
    tr = [fit[c]["vm_rmse_mV"] for c in TRAIN_CYCLES]
    vt = [fit[c]["vm_rmse_mV"] for c in VALTEST_CYCLES]
    z = [v["vm_rmse_mV"] for k, v in fit.items() if k.startswith("0/")]
    put("RealVmFitTrainMin", fmt(min(tr), 0)); put("RealVmFitTrainMax", fmt(max(tr), 0))
    put("RealVmFitValTestMin", fmt(min(vt), 0)); put("RealVmFitValTestMax", fmt(max(vt), 0))
    put("RealVmFitUsZeroSix", fmt(fit["25/US06"]["vm_rmse_mV"], 0))
    put("RealVmFitZeroMin", fmt(min(z), 0)); put("RealVmFitZeroMax", fmt(max(z), 0))
    put("RealArrheniusFactor", fmt(d["arrhenius_factor_0C"], 1))
    put("RealEcmVmax", fmt(ecm["v_max_mV"] / 1e3, 1))

    # ---- sizes
    nw = d["n_windows"]
    put("RealNwinUsZeroSix", str(nw["test25/US06"]))
    put("RealNwinHwfet", str(nw["test25/HWFTa"] + nw["test25/HWFTb"]))
    put("RealNwinNn", str(nw["test25/NN"]))
    put("RealNwinVal", str(nw["val"]))
    tr_ = d["train"]
    seeds = sorted(tr_.keys())
    put("RealNSeeds", str(len(seeds)))
    params = [tr_[s][m]["n_params"] for s in seeds for m in ("latent", "concat", "lora", "hyper")]
    put("RealParamsMin", f"{min(params)/1e3:.1f}k"); put("RealParamsMax", f"{max(params)/1e3:.1f}k")
    put("RealAlpha", fmt(d["scales"]["design_check"]["alpha"], 1))
    put("RealEkfRvm", fmt(d["filter_tuning"]["ekf"]["choice"]["r_vm"], 2))
    put("RealUkfRvm", fmt(d["filter_tuning"]["ukf"]["choice"]["r_vm"], 2))
    put("RealGenerated", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))

    # ---- write macros
    lines = ["% generated by analysis/real_numbers.py from experiments/real/results.json -- do not edit",
             f"% generated {M['RealGenerated']}"]
    for k in sorted(M):
        v = M[k].replace("%", r"\%").replace("_", r"\_")
        lines.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    (OUT / "numbers_real.tex").write_text("\n".join(lines) + "\n")

    # ---- table: settled SOC RMSE [%] per method and cycle
    order = [m for m in METHODS if m in present] + sorted(m for m in present if m not in METHODS)
    rows = []
    for m in order:
        cells = []
        for cyc in TEST25:
            v = t25[cyc].get(m)
            if v is None:
                cells.append("--"); continue
            r = v["rmse_settled"]
            cells.append(f"{r['mean']:.2f}" + (f" $\\pm$ {r['sd']:.2f}" if "sd" in r else ""))
        vv = val.get(m)
        cells.append(f"{vv['rmse_settled']['mean']:.2f}" if vv else "--")
        lab = LABEL.get(m, m.replace("_", r"\_"))
        if m == "ws_best":
            cells = [f"\\textbf{{{c}}}" for c in cells[:3]] + cells[3:]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    tab = [
        "% generated by analysis/real_numbers.py -- do not edit by hand",
        r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Real-data validation: settled SOC RMSE [\% SOC] ($t\ge15$ min of 30-min windows, prior SOC 20\% off) on the three held-out 25$^\circ$C drive cycles and on the validation cycle (Cycle 4, used for model selection and filter tuning only). Learned: mean $\pm$ sd over " + M["RealNSeeds"] + r" seeds. WS-best = HyperKKL ($r=4$) or its dense-subspace ablation, whichever has the lower validation RMSE per seed (the pre-registered selection). Same-data baselines (Koopman KF, grey-box EKF/UKF): " + M["RealSameDataBaselines"] + r".}",
        r"\label{tab:real}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"method & US06 & HWFET & NN & validation \\", r"\midrule",
        *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_real.tex").write_text("\n".join(tab) + "\n")
    print(f"wrote {len(M)} macros and tab_real.tex to {OUT} (same-data baselines: {M['RealSameDataBaselines']})")


if __name__ == "__main__":
    main()
