"""Progress of the case-study runs: done jobs per (case, level, family); deletes filters.json without ekf_oracle
(written by jobs started before the ekf_oracle addition) so the dispatcher re-runs them."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "runs"
cnt, stale = {}, []
for f in R.glob("[AB]/*/seed*/*.json"):
    case, level = f.parts[-4], f.parts[-3]
    if f.stem == "filters":
        if "ekf_oracle" not in json.loads(f.read_text())["methods"]:
            stale.append(f); f.unlink(); continue
    cnt[(case, level, f.stem)] = cnt.get((case, level, f.stem), 0) + 1
for f in R.glob("[AB]/*/seed*/*.error.txt"):
    print("ERROR FILE", f)
print("stale filters removed:", len(stale))
fams = sorted({k[2] for k in cnt})
for case in "AB":
    for level in ["realistic", "exact", "partial", "mm30", "mm10"]:
        print(case, f"{level:9s}", " ".join(f"{fm}:{cnt.get((case, level, fm), 0)}" for fm in fams))
print("total", sum(cnt.values()))
