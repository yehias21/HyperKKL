"""Run the archive-recipe jobs on GPUs 5-7 only, SLOTS per GPU, skipping finished runs.

    python dispatch_ar.py SELECTION.json --seeds 0 1 2 [--slots 3]
Jobs are ordered seed-major so that every cell gets seed 0..2 before extra seeds."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
NODE = {"C3-partial": "AR-partial", "C3-mismatch30": "AR-mismatch30"}
ap = argparse.ArgumentParser()
ap.add_argument("selection"); ap.add_argument("--suffix", default=""); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--slots", type=int, default=3); ap.add_argument("--gpus", default="5,6,7")
ap.add_argument("--temp_gpus", default="", help="extra GPUs usable only until --temp_stop (local time HH:MM)")
ap.add_argument("--temp_last_start", default="03:45"); ap.add_argument("--temp_stop", default="05:40")
ap.add_argument("--temp_slots", type=int, default=2)
a = ap.parse_args()
gpus = [int(g) for g in a.gpus.split(",")]
assert set(gpus) <= {5, 6, 7}, "permanent GPUs: 5, 6, 7 only"
tgpus = [int(g) for g in a.temp_gpus.split(",") if g]
assert set(tgpus) <= {0, 1, 2, 3, 4}
import datetime as _dt
def _today(hm):
    h, m = map(int, hm.split(":")); n = _dt.datetime.now()
    t = n.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > n - _dt.timedelta(hours=12) else t + _dt.timedelta(days=1)
T_LAST, T_STOP = _today(a.temp_last_start), _today(a.temp_stop)
sel = json.load(open(a.selection))["selected"]
jobs = [(NODE[c["node"]] + a.suffix, c["system"], s) for s in a.seeds for c in sel]
done = lambda j: (HERE / "experiments/tree" / j[0] / "runs" / j[1] / f"seed{j[2]}" / "results.json").exists()
jobs = [j for j in jobs if not done(j)]
(HERE / "logs").mkdir(exist_ok=True)
running = {}
slots = [(g, i) for i in range(a.slots) for g in gpus] + [(g, i) for i in range(a.temp_slots) for g in tgpus]
print(f"{len(jobs)} jobs on {len(slots)} slots", flush=True)
while jobs or running:
    now = _dt.datetime.now()
    for sl in slots:
        if sl[0] in tgpus and sl in running and now >= T_STOP:
            p, j, t0 = running.pop(sl)
            p.terminate()
            try: p.wait(60)
            except Exception: p.kill()
            jobs.insert(0, j)                      # cached phases make the restart cheap
            print(f"[{time.strftime('%H:%M:%S')}] STOPPED {j} on temp gpu {sl[0]} (deadline); requeued", flush=True)
            continue
        if sl[0] in tgpus and sl not in running and now >= T_LAST:
            continue
        if sl in running and running[sl][0].poll() is not None:
            p, j, t0 = running.pop(sl)
            print(f"[{time.strftime('%H:%M:%S')}] done {j} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
            if p.returncode != 0 and not done(j):
                print(f"   FAILED {j}; see logs", flush=True)
        if sl not in running and jobs:
            j = jobs.pop(0)
            log = open(HERE / "logs" / f"{j[0]}_{j[1]}_s{j[2]}.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(sl[0]), OMP_NUM_THREADS="2")
            p = subprocess.Popen([PY, str(HERE / "run_ar_job.py"), "--node", j[0], "--system", j[1],
                                  "--seed", str(j[2]), "--device", "cuda:0"], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[sl] = (p, j, time.time())
            print(f"[{time.strftime('%H:%M:%S')}] start {j} on gpu {sl[0]}", flush=True)
    time.sleep(10)
print("ALL DONE", flush=True)
