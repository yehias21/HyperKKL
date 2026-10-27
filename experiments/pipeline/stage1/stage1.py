"""Stage 1 (eval-only): does phase-1 base-map quality predict KKL's losses?  Read-only on
experiments/tree.  Usage: python -I stage1.py ROOT OUT_JSON"""
import json, sys, glob, os
import numpy as np
from scipy.stats import spearmanr, t as student_t

ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
from hyperkkl.metrics import paired_relative  # noqa: E402

TREE = os.path.join(ROOT, "experiments", "tree")
NODES = ["C3-partial", "C3-mismatch30", "C1-where"]
TIERS = ["id", "shape_ood", "sequence_ood"]
KKL_COND = ["latent", "concat", "film", "lora", "hyper"]
DATA = ["nlox", "koopman_kf", "koopman_kf_bilinear", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
MODEL = ["ekf", "ukf", "pf", "kkl_mb", "ekf_huber", "ukf_huber", "ekf_oracle", "ukf_oracle"]
DIAG = ["eps_dec", "eps_enc_rel", "eps_pde_rms", "eps_pde_ss_rms", "l_dec_q95"]

runs = {}
for n in NODES:
    for f in sorted(glob.glob(f"{TREE}/{n}/runs/*/seed*/results.json")):
        d = json.load(open(f))
        if d["seed"] > 2:
            continue                       # seeds 0-2 everywhere (C1-where has 0-4)
        r = {"diag": d["phase1"]["diag"], "m": {}}
        for t in TIERS:
            r["m"][t] = {k: np.asarray(v["nrmse"], float) for k, v in d["results"][t]["clean"]["methods"].items()}
        runs[(n, d["system"], d["seed"])] = r

cells = {}
for (n, s, k), r in runs.items():
    cells.setdefault((n, s), []).append((k, r))


def seedmean(lst, t, m):
    return float(np.mean([r["m"][t][m].mean() for _, r in lst]))


def best_of(lst, t, names):
    have = [m for m in names if all(m in r["m"][t] for _, r in lst)]
    if not have:
        return None, None
    b = min(have, key=lambda m: seedmean(lst, t, m))
    return b, seedmean(lst, t, b)


rows = []
for (n, s), lst in sorted(cells.items()):
    lst.sort(key=lambda a: a[0])
    row = {"node": n, "system": s, "seeds": [k for k, _ in lst]}
    for dk in DIAG:
        row[dk] = float(np.mean([r["diag"][dk] for _, r in lst]))
    for t in TIERS:
        bk, vk = best_of(lst, t, KKL_COND)
        bd, vd = best_of(lst, t, DATA)
        bm, vm = best_of(lst, t, MODEL)
        row[t] = {"kkl": bk, "kkl_nrmse": vk, "data": bd, "data_nrmse": vd, "model": bm, "model_nrmse": vm,
                  "log_gap_data": (float(np.log(vk / vd)) if vd else None)}
        if bd:
            row[t]["rel_vs_data"] = paired_relative([r["m"][t][bk] for _, r in lst], [r["m"][t][bd] for _, r in lst])
        if all("autonomous" in r["m"][t] for _, r in lst):
            row[t]["auto_nrmse"] = seedmean(lst, t, "autonomous")
            row[t]["best_vs_auto"] = paired_relative([r["m"][t][bk] for _, r in lst],
                                                     [r["m"][t]["autonomous"] for _, r in lst])
            # per conditioned method vs autonomous
            row[t]["each_vs_auto"] = {m: paired_relative([r["m"][t][m] for _, r in lst],
                                                         [r["m"][t]["autonomous"] for _, r in lst])["rel"]
                                      for m in KKL_COND if all(m in r["m"][t] for _, r in lst)}
    rows.append(row)


def corr(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 4:
        return None
    rho, p = spearmanr(np.log(xs[ok]), ys[ok])
    return {"rho": float(rho), "p": float(p), "n": int(ok.sum())}


corrs = {}
for t in TIERS:
    for scope in NODES + ["pooled"]:
        sel = [r for r in rows if scope == "pooled" or r["node"] == scope]
        for dk in DIAG:
            corrs[f"{t}|{scope}|{dk}|abs"] = corr([r[dk] for r in sel], [np.log(r[t]["kkl_nrmse"]) for r in sel])
            sg = [r for r in sel if r[t]["log_gap_data"] is not None]
            corrs[f"{t}|{scope}|{dk}|gap"] = corr([r[dk] for r in sg], [r[t]["log_gap_data"] for r in sg])

# within-system, across-seed: does the seed with the worse base map have the worse KKL? (pooled
# Spearman of within-cell centred logs)
within = {}
for t in TIERS:
    for dk in DIAG:
        xs, ys = [], []
        for (n, s), lst in cells.items():
            bk, _ = best_of(lst, t, KKL_COND)
            lx = np.log([r["diag"][dk] for _, r in lst]); ly = np.log([r["m"][t][bk].mean() for _, r in lst])
            xs += list(lx - lx.mean()); ys += list(ly - ly.mean())
        rho, p = spearmanr(xs, ys)
        within[f"{t}|{dk}"] = {"rho": float(rho), "p": float(p), "n": len(xs)}

json.dump({"rows": rows, "corr": corrs, "within": within}, open(OUT, "w"), indent=1, default=float)

# ---- printout -------------------------------------------------------------------------
for t in ["sequence_ood", "shape_ood", "id"]:
    print(f"\n=== {t}")
    print(f"{'node':14s} {'system':10s} {'eps_dec':>8s} {'eps_enc':>8s} {'pde':>8s} {'l_q95':>7s} | {'KKL':>8s} {'best':6s} | {'data':>8s} {'which':12s} {'rel vs data [CI]':>24s} | {'auto':>7s} {'best vs auto [CI]':>22s}")
    for r in rows:
        x = r[t]
        rv = x.get("rel_vs_data"); ra = x.get("best_vs_auto")
        f = lambda q: f"{100*q['rel']:+.0f}% [{100*q['ci'][0]:+.0f},{100*q['ci'][1]:+.0f}]{'*' if q['better'] or q['worse'] else ''}" if q else "-"
        print(f"{r['node']:14s} {r['system']:10s} {r['eps_dec']:8.3f} {r['eps_enc_rel']:8.4f} {r['eps_pde_rms']:8.3f} {r['l_dec_q95']:7.0f} | "
              f"{x['kkl_nrmse']:8.4f} {x['kkl']:6s} | {(x['data_nrmse'] or float('nan')):8.4f} {str(x['data']):12s} {f(rv):>24s} | "
              f"{x.get('auto_nrmse', float('nan')):7.4f} {f(ra):>22s}")
print("\n=== Spearman (log diag vs log KKL NRMSE | log gap to best data-driven)")
for k, v in corrs.items():
    if v and ("sequence_ood" in k or "pooled" in k):
        print(f"{k:50s} rho {v['rho']:+.2f}  p {v['p']:.3f}  n {v['n']}")
print("\n=== within-cell across seeds")
for k, v in within.items():
    print(f"{k:30s} rho {v['rho']:+.2f} p {v['p']:.3f} n {v['n']}")
