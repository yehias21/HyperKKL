"""Aggregate out12/ and out34/ -> ../ERROR_FLOOR.json (+ tables printed as markdown)."""
import json, glob, numpy as np
from pathlib import Path
D = Path(__file__).resolve().parent
SYS = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]
NODES = ["C1-where", "C3-partial", "C3-mismatch30"]
M = ["latent", "lora", "hyper", "ekf", "greybox_ekf", "greybox_ukf"]
TIERS = ["id", "sequence_ood"]
agg = {"exp12": {}, "exp34": {}, "missing12": [], "missing34": []}
for n in NODES:
    for s in SYS:
        for k in range(3):
            f = D / "out12" / f"{n}_{s}_s{k}.json"
            if not f.exists():
                agg["missing12"].append(f.name); continue
            d = json.loads(f.read_text())
            for t in TIERS:
                for c, ms in d["res"][t].items():
                    for m, v in ms.items():
                        agg["exp12"].setdefault(n, {}).setdefault(s, {}).setdefault(t, {}).setdefault(c, {}).setdefault(m, []).append(float(np.mean(v["nrmse"])))
for f in sorted((D / "out34").glob("*.json")):
    d = json.loads(f.read_text())
    agg["exp34"].setdefault(d["node"], {}).setdefault(d["system"], []).append(d)
gm = lambda a: float(np.exp(np.mean(np.log(np.asarray(a)))))
mean = lambda a: float(np.mean(a))

def fmt(x): return "-" if x is None else (f"{x:.4f}" if x < 0.1 else f"{x:.3f}")

lines = []
P = lines.append
# ---- Table 1: noise-free vs 3% ------------------------------------------------------------
for t in TIERS:
    P(f"\n#### Exp 1 ({t}): NRMSE at 3% sensor noise (stored clean episodes) -> noise-free (same episodes), seed mean\n")
    P("| node | system | " + " | ".join(M) + " | best KKL / best grey-box: 3% -> 0% |")
    P("|---|---|" + "---|" * (len(M) + 1))
    for n in NODES:
        for s in SYS:
            e = agg["exp12"].get(n, {}).get(s, {}).get(t)
            if not e: continue
            row, best = [], {}
            for m in M:
                a, b = e["clean"].get(m), e["n0"].get(m)
                if a is None: row.append("-"); continue
                row.append(f"{fmt(mean(a))} -> {fmt(mean(b))}")
            kk = min(("latent", "lora", "hyper"), key=lambda m: mean(e["clean"][m]))
            kz = min(("latent", "lora", "hyper"), key=lambda m: mean(e["n0"][m]))
            cand = [m for m in ("greybox_ekf", "greybox_ukf") + (("ekf",) if n != "C3-partial" else ()) if m in e["clean"]]
            gb = min(cand, key=lambda m: mean(e["clean"][m]))
            gz = min(cand, key=lambda m: mean(e["n0"][m]))
            r3 = mean(e["clean"][kk]) / mean(e["clean"][gb]); r0 = mean(e["n0"][kz]) / mean(e["n0"][gz])
            P(f"| {n} | {s} | " + " | ".join(row) + f" | {r3:.1f}x -> {r0:.1f}x |")
# ---- Table 2: noise scaling, C1-where ------------------------------------------------------
P("\n#### Exp 2: C1-where, NRMSE vs sensor noise 0 / 1 / 3 / 10 % (re-drawn noise, same episodes), sequence_ood; slope b = d log e / d log sigma fitted on 1-10 %; e(0) = noise-free floor\n")
P("| system | method | 0% | 1% | 3% | 10% | slope b (1-10%) | noise part e_n(3%) = sqrt(e3^2 - e0^2) | slope of e_n (1-10%) | noise share of MSE at 3% |")
P("|---|---|---|---|---|---|---|---|---|---|")
slopes = {}
for s in SYS:
    e = agg["exp12"].get("C1-where", {}).get(s, {}).get("sequence_ood")
    if not e: continue
    for m in [m for m in M if m in e["n0"]]:
        v = [mean(e[c][m]) for c in ("n0", "n1", "n3", "n10")]
        b = float(np.polyfit(np.log([1, 3, 10]), np.log(v[1:]), 1)[0])
        en = [float(np.sqrt(max(x ** 2 - v[0] ** 2, 1e-12))) for x in v[1:]]
        bn = float(np.polyfit(np.log([1, 3, 10]), np.log(en), 1)[0])
        slopes.setdefault(s, {})[m] = {"e": v, "slope": b, "noise_part": en, "slope_noise_part": bn, "noise_share3": 1 - (v[0] / v[2]) ** 2}
        P(f"| {s} | {m} | " + " | ".join(fmt(x) for x in v) + f" | {b:.2f} | {fmt(en[1])} | {bn:.2f} | {100 * (1 - (v[0] / v[2]) ** 2):.0f}% |")
agg["slopes_C1"] = slopes
# ---- Table 3: probe ------------------------------------------------------------------------
for n in agg["exp34"]:
    for t in TIERS:
        P(f"\n#### Exp 3 ({n}, {t}): held-out pooled NRMSE (k >= settle) of the trained decoders vs probes fit on the training episodes, seed mean\n")
        P("| system | lora dec (z_obs) | hyper dec | lora dec on noise-free z | MLP [z_obs,c] | kNN [z_obs,c] | MLP [z_win5,c] | MLP [z_win20,c] | MLP [z_win20 stride10,c] | MLP [z_win20 stride30,c] | MLP [z_clean,c] | MLP [z_clean win20,c] |")
        P("|---|" + "---|" * 11)
        for s in SYS:
            L = agg["exp34"][n].get(s)
            if not L: continue
            g = lambda f: mean([f(d) for d in L])
            pr = lambda key: (g(lambda d: d["probe"][key][t]) if all(key in d["probe"] for d in L) else None)
            P(f"| {s} (n={len(L)}) | {fmt(g(lambda d: d['decoder']['lora'][t]['obs']))} | {fmt(g(lambda d: d['decoder']['hyper'][t]['obs']))} | "
              f"{fmt(g(lambda d: d['decoder']['lora'][t]['clean']))} | {fmt(pr('z_c|obs'))} | {fmt(g(lambda d: d['knn']['z_c|obs'][t]))} | "
              f"{fmt(pr('zwin5_c|obs'))} | {fmt(pr('zwin20_c|obs'))} | {fmt(pr('zwin20s10_c|obs'))} | {fmt(pr('zwin20s30_c|obs'))} | {fmt(pr('z_c|clean'))} | {fmt(pr('zwin20_c|clean'))} |")
# ---- Table 4: Jacobian ---------------------------------------------------------------------
for n in agg["exp34"]:
    P(f"\n#### Exp 4 ({n}): decoder Jacobian ||diag(1/sigma) dT+/dz||_2 along evaluation trajectories (sequence_ood, settled), seed mean; noise part = RMS of T+(z_obs,c) - T+(z_clean,c); linearised = RMS of J (z_obs - z_clean)\n")
    P("| system | method | ||J|| median | ||J|| q95 | ||J diag(z_std)|| median | RMS z noise | noise part (actual) | noise part (linearised J e_z) | total error | noise-free error |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for s in SYS:
        L = agg["exp34"][n].get(s)
        if not L: continue
        for m in ("lora", "hyper", "autonomous_base"):
            g = lambda f: mean([f(d) for d in L])
            j = lambda key: g(lambda d: d["jac"][m]["sequence_ood"][key])
            P(f"| {s} | {m} | {j('spec_median'):.0f} | {j('spec_q95'):.0f} | {j('jac_scaled_by_zstd_median'):.2f} | {j('ez_rms'):.2g} | "
              f"{fmt(g(lambda d: d['decoder'][m]['sequence_ood']['noise_part']))} | {fmt(j('lin_noise_nrmse'))} | "
              f"{fmt(g(lambda d: d['decoder'][m]['sequence_ood']['obs']))} | {fmt(g(lambda d: d['decoder'][m]['sequence_ood']['clean']))} |")
fn = json.loads((D / "faults_noise10.json").read_text())
P("\n#### Exp 2b: C3-faults (exact model), noise10 / clean ratio and implied slope b = log(ratio)/log(10/3), seed mean (3 seeds)\n")
MM = ["latent", "lora", "hyper", "nlox", "ekf", "ukf", "pf", "ekf_oracle"]
P("| system | tier | " + " | ".join(MM) + " |"); P("|---|---|" + "---|" * len(MM))
for s in SYS:
    for t in TIERS:
        r = fn.get(s, {}).get(t, {})
        P(f"| {s} | {t} | " + " | ".join(f"x{r[m]['ratio']:.2f} (b {r[m]['slope']:.2f})" if m in r else "-" for m in MM) + " |")
agg["faults_noise10"] = fn
(D.parent / "ERROR_FLOOR.json").write_text(json.dumps(agg, indent=1, default=float))
(D / "tables.md").write_text("\n".join(lines))
print("\n".join(lines)); print("missing12", agg["missing12"])
