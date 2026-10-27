"""Run the quad_tank PE jobs on GPUs 5-7 only, SLOTS per GPU (default 2).
A (node, seed) is never run twice at once (its arms share generated data files)."""
import argparse, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--slots", type=int, default=2); ap.add_argument("--gpus", default="5,6,7")
ap.add_argument("--arms", default="X1,X3,X2,X0"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--nodes", default="C3-partial,C3-mismatch30")
a = ap.parse_args()
gpus = [int(g) for g in a.gpus.split(",")]
assert set(gpus) <= {5, 6, 7}, "GPUs 5, 6, 7 only"
jobs = [{"node": n, "seed": int(s), "arm": arm} for s in a.seeds.split(",") for arm in a.arms.split(",")
        for n in a.nodes.split(",")]
done = lambda j: (HERE / "runs" / j["node"] / f"seed{j['seed']}" / f"{j['arm']}.json").exists()
jobs = [j for j in jobs if not done(j)]
(HERE / "logs").mkdir(exist_ok=True)
slots = [(g, i) for i in range(a.slots) for g in gpus]
running, fails = {}, {}
print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs on {len(slots)} slots", flush=True)
while jobs or running:
    for sl in slots:
        if sl in running and running[sl][0].poll() is not None:
            p, j, t0 = running.pop(sl)
            print(f"[{time.strftime('%H:%M:%S')}] done {j} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
            if p.returncode != 0 and not done(j):
                k = (j["node"], j["seed"], j["arm"])
                fails[k] = fails.get(k, 0) + 1
                if fails[k] < 3:
                    jobs.append(j); print("   FAILED, requeued", flush=True)
                else:
                    print("   FAILED 3x, dropped", flush=True)
        if sl not in running:
            busy = {(r[1]["node"], r[1]["seed"]) for r in running.values()}
            ready = [j for j in jobs if (j["node"], j["seed"]) not in busy]
            if ready:
                j = ready[0]; jobs.remove(j)
                tag = f"{j['node']}_s{j['seed']}_{j['arm']}"
                log = open(HERE / "logs" / f"{tag}.log", "a")
                cmd = [PY, str(HERE / "run_qt.py"), "--node", j["node"], "--seed", str(j["seed"]),
                       "--arm", j["arm"], "--device", "cuda:0", "--no-gb"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(sl[0]), OMP_NUM_THREADS="2")
                running[sl] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE), j, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] start {tag} on gpu {sl[0]}", flush=True)
    time.sleep(20)
print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)
