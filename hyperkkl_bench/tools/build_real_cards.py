"""Write docs/cards/real_<dataset>.md (+ REAL_EXCLUDED.md, REAL_SMOKE.md) from hyperkkl_bench.real.cards and the
smoke reports in reference/real/smoke/."""
import json
from pathlib import Path

from hyperkkl_bench.real import list_real
from hyperkkl_bench.real.cards import card_markdown, excluded_markdown, neurobem_table, smoke_results

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "docs" / "cards"
for d in list_real():
    (out / f"real_{d}.md").write_text(card_markdown(d))
(out / "REAL_EXCLUDED.md").write_text(excluded_markdown())
ORDER = ["prior_mean", "coulomb", "finite_diff", "imu_static", "ekf", "ekf_gated", "ude_ekf_gated", "gru", "hyperkkl"]
L = ["# Real-data smoke runs (1 seed, through the benchmark API)", "",
     "Each cell is the track's headline (battery: settled SOC RMSE [% SOC], mean of the 25 degC test families; "
     "emps: velocity NRMSE during encoder outages; silverbox: settled velocity NRMSE, mean of 3 test splits; neurobem: settled hidden NRMSE, mean of test_id and test_ood). "
     "Lower is better. One training seed, reduced budgets for UDE (1000 steps x 2 learning rates) and GRU (64 units, "
     "1500 steps); HyperKKL = A3 recipe at full budget (phase 1 6000 + 6000 steps, phase 2 3000 steps), 1 seed, GPU. "
     "These show that every path runs end to end and gives sane numbers; they are NOT benchmark results "
     "(PROTOCOL: >= 3 seeds).", "",
     "| dataset | " + " | ".join(ORDER) + " |", "|---|" + "---|" * len(ORDER)]
for d in list_real():
    sm = smoke_results(d)
    L.append(f"| {d} | " + " | ".join(f"{sm[m]['headline']:.4g}" if m in sm and sm[m]["headline"] is not None else "-"
                                      for m in ORDER) + " |")
L += ["", "## neurobem in physical units", ""] + neurobem_table()
(out / "REAL_SMOKE.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
