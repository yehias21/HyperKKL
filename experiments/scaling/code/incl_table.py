"""Inclusion / budget table from inclusion/ring*.json (markdown to stdout)."""
import json
from pathlib import Path

import numpy as np

SC = Path(__file__).resolve().parent.parent
print("| n_x | n_y | n_z | observable? | bounded? (rule) | max unmeasured full-past CRB; Gramian median lmin/lmax (frac full rank) | worst S(5T)/S(T); secular | perturbed-30% plant bounded? (extra; worst ratio) | A = window PCRB unmeasured D0 / D30 / DR |")
print("|---|---|---|---|---|---|---|---|---|")
for nx in (4, 8, 16, 32, 64, 128):
    f = SC / "inclusion" / f"ring{nx}.json"
    if not f.exists():
        print(f"| {nx} | | | pending | | | | | |"); continue
    r = json.loads(f.read_text())
    def worst(bd):
        e = max(max(v["ratio_end"]) for k, v in bd.items() if k != "pass")
        s = max(max(v["ratio_secular"]) for k, v in bd.items() if k != "pass")
        return e, s
    e, s = worst(r["bounded_detail"]); e3, s3 = worst(r["bounded_perturbed30_detail"])
    g = r["gramian"]
    ob = ("yes" + (" (weak)" if r["weak"] else "")) if r["observable"] else "NO"
    B = r.get("budget", {})
    A = " / ".join(f"{B[d]['A_window_unmeasured_lb']:.4f}" for d in ("D0", "D30", "DR") if d in B) or "pending"
    print(f"| {nx} | {len(r['system']['sensors'])} | {r['design_check']['nz']} | {ob} | {'yes' if r['bounded'] else 'NO'} | "
          f"{r['crb_unmeasured_max']:.4f}; {g['ratio_median']:.1e} ({100 * g['frac_full_rank']:.0f}%) | {e:.2f}; {s:.2f} | "
          f"{'yes' if r['bounded_perturbed30'] else 'no'} ({e3:.2f}) | {A} |")
