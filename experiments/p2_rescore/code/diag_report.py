"""Cartpole diagnosis tables from diag/*.json -> diag/summary.json, printed markdown."""
import json, math
from pathlib import Path
import numpy as np
P = Path(__file__).resolve().parent.parent
D = P / "diag"
B = json.loads((P.parent / "information" / "budgets.json").read_text())["cartpole"]
w3 = B["dists"]["eval_id"]["bayes"]["windows"]["2"]["crb_state"]; wf = B["dists"]["eval_id"]["bayes"]["full"]["crb_state"]
C = ["stored", "ct_redraw", "noiseless", "gauss_white", "gauss_col"]
T = ["id", "shape_ood", "sequence_ood"]
rows = {}
for which, models in (("old", ["lora", "greybox_ukf"]), ("new", ["A2", "A4"])):
    js = [json.loads((D / f"{which}_s{s}.json").read_text()) for s in range(3)]
    for m in models:
        rows[m] = {"chk": [], "ldec": [j["models"].get(m, {}).get("l_dec_q95") for j in js]}
        for t in T:
            for c in C:
                rr = [j["results"][m][t][c] for j in js]
                rows[m][(t, c)] = {"unm": np.mean([r["unm"] for r in rr]), "tot": np.mean([r["nrmse"] for r in rr]),
                                   "state": np.mean([r["state"] for r in rr], 0), "unm_seeds": [r["unm"] for r in rr]}
            rows[m]["chk"].append((t, [j["stored_clean_check"][m][t] for j in js], [j["results"][m][t]["stored"]["nrmse"] for j in js]))
L = []
L.append("Reproduction (total NRMSE, stored clean vs re-evaluated `stored`, seeds 0-2):")
for m in rows:
    for t, a, b in rows[m]["chk"]:
        if t == "sequence_ood":
            L.append(f"- {m}: {' / '.join(f'{x:.4f}' for x in a)} vs {' / '.join(f'{x:.4f}' for x in b)}")
L += ["", "| model | tier | condition | total | unmeasured | x1 (meas.) | x2 cart vel | x3 (meas.) | x4 ang. vel |", "|---|---|---|---|---|---|---|---|---|"]
summ = {}
for m in rows:
    for t in T:
        for c in C:
            r = rows[m][(t, c)]
            L.append(f"| {m} | {t} | {c} | {r['tot']:.4f} | {r['unm']:.4f} | " + " | ".join(f"{v:.4f}" for v in r["state"]) + " |")
        st = rows[m][(t, "stored")]["unm"]; nl = rows[m][(t, "noiseless")]["unm"]
        summ.setdefault(m, {})[t] = {"drop_noiseless": 1 - nl / st,
                                     "gw_vs_ct": rows[m][(t, "gauss_white")]["unm"] / rows[m][(t, "ct_redraw")]["unm"],
                                     "gc_vs_ct": rows[m][(t, "gauss_col")]["unm"] / rows[m][(t, "ct_redraw")]["unm"],
                                     "x2_over_crb3": rows[m][(t, "noiseless")]["state"][1] / w3[1],
                                     "x4_over_crb3": rows[m][(t, "noiseless")]["state"][3] / w3[3]}
L += ["", "| model | l_dec_q95 (s0/s1/s2) | tier | unm drop noiseless vs stored | gauss white / ct redraw | gauss coloured / ct redraw | x2 (noiseless) / CRB(3/a) | x4 (noiseless) / CRB(3/a) |", "|---|---|---|---|---|---|---|---|"]
for m in summ:
    for t in T:
        s = summ[m][t]; ld = rows[m]["ldec"]
        lds = "-" if ld[0] is None else " / ".join(f"{v:.0f}" for v in ld)
        L.append(f"| {m} | {lds} | {t} | {s['drop_noiseless']:+.0%} | {s['gw_vs_ct']:.2f} | {s['gc_vs_ct']:.2f} | {s['x2_over_crb3']:.1f} | {s['x4_over_crb3']:.0f} |")
L += ["", f"Bayesian PCRB per state (x_std units, eval_id, budgets.json): window 3/alpha = 2 s: {', '.join(f'{v:.4f}' for v in w3)}; full past: {', '.join(f'{v:.4f}' for v in wf)}."]
(D / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
print("\n".join(L))
