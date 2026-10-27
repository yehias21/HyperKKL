"""POST HOC (2026-10-26): bits-sufficiency rule with the corrected budgets (BUDGET_FIX.md).

    cd experiments && ../.venv/bin/python information/bits_rule_c3.py

Same rule as bits_rule_posthoc.py (HyperKKL 'cannot win' if A > eps; A = unmeasured Bayesian PCRB
for the window 3/alpha; eps = best grey-box unmeasured error of the same tier; HyperKKL outcome from
p2_rescore SOAP arms A2/A4 where present, else the original lora run) with two fixes:
  * test plants: A from budgets_c3.json (TEST-C3 noise model: AR(1) rho 0.9 Student-t nu 3, true plant
    with 0.3 mismatch); dev plants: budgets.json (their comparison C3-regimes clean = white Gaussian
    0.03, mismatch 0 = the C1 setting, unchanged);
  * tier sequence_ood uses the eval_seq distribution (was eval_id for both tiers).
Writes bits_rule_c3.json.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score as S  # noqa: E402

B = json.loads((HERE / "budgets.json").read_text())
BC = json.loads((HERE / "budgets_c3.json").read_text())
R = json.loads((HERE.parent / "p2_rescore" / "results.json").read_text())["plants"]
DIST = {"id": "eval_id", "sequence_ood": "eval_seq"}


def A_of(s, tier, which="new"):
    if which == "old":
        return S.budget(s, "eval_id")["A"], S.budget(s, "eval_id")["full_unm"]
    if s in BC and "error" not in BC[s]:
        d = BC[s]["dists"][DIST[tier]]["c3"]["bayes"]; a = BC[s]["alpha"]
        return d["windows"][f"{3 / a:g}"]["unmeasured_lb"], d["full"]["unmeasured_lb"]
    b = S.budget(s, DIST[tier])
    return b["A"], b["full_unm"]


def outcome(s, tier):
    gbm, eps = S.best_greybox(s, tier)
    if gbm is None or eps != eps:
        return None
    ms = S.A[s]["partial"]["methods"]
    fold, ci, src = None, None, "old"
    p = R.get(s, {}).get(f"{tier}/unm")
    if p:
        eps = p["greybox_mean"]
        c = [(p[a]["fold"], a, p[a]["fold_ci"]) for a in ("A2", "A4") if a in p and p[a]]
        if c:
            fold, src, ci = min(c)
    if fold is None and "lora" in ms and tier in ms["lora"]:
        fold = ms["lora"][tier]["unmeasured"] / eps
    if ci:
        out = "win" if ci[1] < 1 else ("lose" if ci[0] > 1 else "tie")
    else:
        out = "win" if fold < 1 else "lose"
    return {"greybox": gbm, "eps": eps, "fold": fold, "fold_ci": ci, "src": src, "outcome": out}


def main():
    res, lines = {}, []
    for tier in ("id", "sequence_ood"):
        lines += [f"\n== tier {tier}", f"{'plant':16s} {'A_old':>7s} {'A_new':>7s} {'full_new':>8s} {'eps_gb':>7s} "
                  f"{'log2(A/eps)':>11s} {'pred_old':>10s} {'pred_new':>10s} {'fold':>6s} {'src':>4s} outcome"]
        for s in S.L.ALL:
            o = outcome(s, tier)
            if o is None:
                continue
            Ao, _ = A_of(s, tier, "old"); An, fn = A_of(s, tier, "new")
            po = "CANNOT" if Ao > o["eps"] else "can"; pn = "CANNOT" if An > o["eps"] else "can"
            ok = not (pn == "CANNOT" and o["outcome"] == "win")
            res.setdefault(tier, {})[s] = dict(o, A_old=Ao, A_new=An, full_new=fn, pred_old=po, pred_new=pn,
                                              violates=not ok, log2_A_over_eps=math.log2(An / o["eps"]))
            lines.append(f"{s:16s} {Ao:7.4f} {An:7.4f} {fn:8.4f} {o['eps']:7.4f} {math.log2(An / o['eps']):+11.2f} "
                         f"{po:>10s} {pn:>10s} {o['fold']:6.2f} {o['src']:>4s} {o['outcome']}{'' if ok else '  <-- VIOLATES'}")
    print("\n".join(lines))
    (HERE / "bits_rule_c3.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
