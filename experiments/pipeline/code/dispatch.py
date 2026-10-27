"""Run every pipeline job on GPUs 5,6,7 (slots per GPU), skipping finished ones; one retry."""
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
OUT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/pipeline/runs")
SYSTEMS = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]
SCEN = ["C3-partial", "C3-mismatch30", "C1-where"]
GPUS = [g for g in sys.argv[1].split(",")] if len(sys.argv) > 1 else ["5", "6", "7"]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 3
assert set(GPUS) <= {"5", "6", "7"}

jobs = [(sc, s, k) for sc in SCEN for k in (0, 1, 2) for s in SYSTEMS]
todo = [j for j in jobs if not (OUT / j[0] / j[1] / f"seed{j[2]}" / "results.json").exists()]
tries = {j: 0 for j in todo}
slots = {(g, i): None for g in GPUS for i in range(PER)}
logd = OUT.parent / "dispatch_logs"
logd.mkdir(parents=True, exist_ok=True)
print(f"{len(todo)} jobs, slots {list(slots)}", flush=True)
while todo or any(slots.values()):
    for sl, cur in list(slots.items()):
        if cur is not None:
            p, j = cur
            if p.poll() is None:
                continue
            ok = p.returncode == 0
            print(f"{time.strftime('%H:%M:%S')} done {j} rc {p.returncode}", flush=True)
            if not ok and tries[j] < 2:
                todo.append(j)
            slots[sl] = None
        if slots[sl] is None and todo:
            j = todo.pop(0)
            tries[j] += 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=sl[0], PYTHONUNBUFFERED="1")
            lf = open(logd / f"{j[0]}_{j[1]}_seed{j[2]}.log", "a")
            p = subprocess.Popen([PY, str(HERE / "run_pipeline.py"), "--scenario", j[0], "--system", j[1],
                                  "--seed", str(j[2]), "--device", "cuda:0"], env=env, stdout=lf, stderr=lf)
            slots[sl] = (p, j)
            print(f"{time.strftime('%H:%M:%S')} start {j} on GPU {sl[0]} (try {tries[j]})", flush=True)
    time.sleep(10)
print("all done", flush=True)
