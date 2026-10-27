"""Stage-B jobs (run_b.py) on GPUs 5-7, SLOTS per GPU."""
import argparse, os, subprocess, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--base-arm", required=True); ap.add_argument("--slots", type=int, default=2)
ap.add_argument("--script", default="run_b.py"); ap.add_argument("--prefix", default="B1")
ap.add_argument("--jobs", default="", help="node:system;node:system ... (default: the 12 C3 cells)")
a = ap.parse_args()
gpus = [5, 6, 7]
cells = [tuple(x.split(":")) for x in a.jobs.split(";") if x] or [(n, s) for n in ("C3-partial", "C3-mismatch30")
        for s in ("quad_tank", "duffing", "battery", "lorenz", "vdp", "bioreactor")]
jobs = [(n, s, sd) for sd in (0, 1, 2) for n, s in cells]
done = lambda j: (HERE / "runs" / j[0] / j[1] / f"seed{j[2]}" / f"{a.prefix}-{a.base_arm}.json").exists()
jobs = [j for j in jobs if not done(j)]
slots = [(g, i) for i in range(a.slots) for g in gpus]
running, tries = {}, {}
print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs", flush=True)
(HERE / "logs").mkdir(exist_ok=True)
while jobs or running:
    for sl in slots:
        if sl in running and running[sl][0].poll() is not None:
            p, j = running.pop(sl)
            print(f"[{time.strftime('%H:%M:%S')}] done {j} rc={p.returncode}", flush=True)
            if p.returncode and not done(j):
                tries[j] = tries.get(j, 0) + 1
                if tries[j] < 3: jobs.append(j); print("   FAILED, requeued", flush=True)
                else: print("   FAILED 3x", flush=True)
        if sl not in running and jobs:
            j = jobs.pop(0)
            log = open(HERE / "logs" / f"{a.prefix}_{j[0]}_{j[1]}_s{j[2]}.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(sl[0]), OMP_NUM_THREADS="2")
            running[sl] = (subprocess.Popen([PY, str(HERE / a.script), "--node", j[0], "--system", j[1], "--seed", str(j[2]),
                                             "--base-arm", a.base_arm, "--device", "cuda:0"], stdout=log, stderr=subprocess.STDOUT,
                                            env=env, cwd=HERE), j)
            print(f"[{time.strftime('%H:%M:%S')}] start {j} gpu {sl[0]}", flush=True)
    time.sleep(15)
print("ALL DONE", flush=True)
