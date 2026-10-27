#!/usr/bin/env bash
# Freeze the system selection once the C3 data-baseline re-evaluation has merged 66/66 runs.
set -u
T=/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/tree
OUT=/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/archive_recipe
until [ "$(ls $T/C3-{partial,realistic,mismatch30}/runs/*/seed*/results.json.bak 2>/dev/null | wc -l)" -ge 66 ]; do sleep 60; done
sleep 120   # let the last merge's results.json rename settle
N=$(ls $T/C3-{partial,realistic,mismatch30}/runs/*/seed*/results.json.bak | wc -l)
/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python -I /mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/archive_recipe/study_data/select_systems.py $T $OUT/SELECTION.tmp.json > $OUT/SELECTION.log 2>&1
/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python -I - $OUT/SELECTION.tmp.json $OUT/SELECTION.json "$N" <<'PY'
import json, sys, time
d = json.load(open(sys.argv[1]))
d["frozen_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
d["frozen_at_dubai"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 4 * 3600)) + " +04"
d["reeval_merged_runs"] = int(sys.argv[3])
d["rule"] = ("scenarios C3-partial, C3-mismatch30; tier sequence_ood; r = min(lora, hyper) / min(non-HyperKKL "
             "methods present on all seeds); per scenario the 2 largest r (LOSE) and 2 smallest r (WIN)")
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
rm -f $OUT/SELECTION.tmp.json
echo FROZEN; cat $OUT/SELECTION.log
