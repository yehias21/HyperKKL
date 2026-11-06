"""Collect the per-plant difficulty axes from the research results into
src/hyperkkl_bench/resources/difficulty.json, each value with its provenance (file + key path).
Read-only on experiments/. Re-run after a source changes; tests check the file is consistent.

Axes
  n_x, n_y, measured / unmeasured             metadata
  info_budget                                 Bayesian posterior CRB lower bound on the settled
                                              unmeasured NRMSE (experiments/information): window
                                              3/alpha (what a fading-memory observer can reach)
                                              vs the full past; white = C1 sensor, realistic = C3
                                              colored Student-t (test plants only; BUDGET_FIX.md)
  window_ratio                                window / full-past bound (>= 1; large = needs memory)
  repairability                               rho = 1 - MSE_greybox / MSE_model of the one-t_ref
                                              open-loop output prediction (experiments/structural)
  lyapunov                                    largest Lyapunov exponent (experiments/chaos_refs)
  observability                               linearised observability Gramian over 3/alpha:
                                              median condition number, rank-deficient fraction
                                              (experiments/cartpole_bounded/inclusion)
  inclusion                                   observable / weak / bounded flags (same files)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parent
EXP = CHECKOUT.parent / "experiments"
sys.path.insert(0, str(CHECKOUT / "src"))
from hyperkkl_bench.data import metadata          # noqa: E402
from hyperkkl_bench.registry import PLANTS       # noqa: E402


def rel(p):
    return str(Path(p).relative_to(CHECKOUT.parent))


def bound(bayes, alpha):
    w = bayes["windows"]
    key = min(w, key=lambda k: abs(float(k) - 3.0 / alpha))
    a, f = w[key]["unmeasured_lb"], bayes["full"]["unmeasured_lb"]
    return {"window_3_over_alpha": a, "full_past": f, "window_ratio": a / f if f else None,
            "bits_window": w[key].get("bits"), "bits_full": bayes["full"].get("bits"), "window_key_s": key}


def main():
    budgets = json.loads((EXP / "information" / "budgets.json").read_text())
    c3 = json.loads((EXP / "information" / "budgets_c3.json").read_text())
    cpb = json.loads((EXP / "cartpole_bounded" / "budget_cartpole_b.json").read_text())
    struct = json.loads((EXP / "structural" / "scores.json").read_text())["P-S3"]
    lyap = json.loads((EXP / "chaos_refs" / "results.json").read_text())["lyapunov"]
    out = {"_about": __doc__, "plants": {}}
    for p in PLANTS:
        meta = metadata(p)
        alpha = float(meta["alpha"])
        src = "cartpole" if p == "legacy_cartpole" else p
        d = {"n_x": meta["n_x"], "n_y": meta["n_y"], "measured": meta["sensor_names"],
             "unmeasured": [meta["state_names"][j] for j in meta["unmeasured"]],
             "group": meta.get("group"), "role": PLANTS[p]["role"], "alpha": alpha}
        ib = {}
        if p == "cartpole_b":
            ib["white"] = bound(cpb["dists"]["eval_id"]["white_gauss"]["bayes"], alpha)
            ib["white"]["provenance"] = rel(EXP / "cartpole_bounded/budget_cartpole_b.json") + ":dists.eval_id.white_gauss.bayes"
            ib["realistic"] = bound(cpb["dists"]["eval_id"]["c3"]["bayes"], alpha)
            ib["realistic"]["provenance"] = rel(EXP / "cartpole_bounded/budget_cartpole_b.json") + ":dists.eval_id.c3.bayes"
        elif src in budgets:
            ib["white"] = bound(budgets[src]["dists"]["eval_id"]["bayes"], alpha)
            ib["white"]["provenance"] = rel(EXP / "information/budgets.json") + f":{src}.dists.eval_id.bayes"
            if src in c3:
                ib["realistic"] = bound(c3[src]["dists"]["eval_id"]["c3"]["bayes"], alpha)
                ib["realistic"]["provenance"] = rel(EXP / "information/budgets_c3.json") + f":{src}.dists.eval_id.c3.bayes"
        d["info_budget"] = ib
        if src in struct["plants"]:
            i = struct["plants"].index(src)
            d["repairability"] = {"rho": 1 - struct["nonrep"][i], "rho_ref": struct["rho_ref"][i],
                                  "greybox_degradation_D": struct["D"][i],
                                  "provenance": rel(EXP / "structural/scores.json") + f":P-S3[{i}]"
                                  + (" (unbounded cartpole)" if src == "cartpole" else "")}
        else:
            d["repairability"] = None
        lk = f"{src}/nominal"
        if lk in lyap:
            d["lyapunov"] = {"lambda_max_mean_id": lyap[lk]["tiers"]["id"]["lam_long"]["mean"],
                             "provenance": rel(EXP / "chaos_refs/results.json") + f":lyapunov.{lk}.tiers.id.lam_long.mean"}
        else:
            d["lyapunov"] = None
        inc = EXP / "cartpole_bounded" / "inclusion" / f"{src}.json"
        if inc.exists():
            j = json.loads(inc.read_text())
            g = j.get("gramian", {})
            d["observability"] = {"gramian_cond_median": g.get("cond_median"), "frac_full_rank": g.get("frac_full_rank"),
                                  "crb_unmeasured_max": j.get("crb_unmeasured_max"),
                                  "provenance": rel(inc) + ":gramian"}
            d["inclusion"] = {"observable": j.get("observable"), "weakly_observable": j.get("weak"),
                              "bounded": j.get("bounded"), "provenance": rel(inc)}
        out["plants"][p] = d
    res = CHECKOUT / "src" / "hyperkkl_bench" / "resources"
    res.mkdir(exist_ok=True)
    (res / "difficulty.json").write_text(json.dumps(out, indent=1))
    print("wrote", res / "difficulty.json")


if __name__ == "__main__":
    main()
