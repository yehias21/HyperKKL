"""Run beat-the-grey-box jobs on GPUs 5-7 only, SLOTS per GPU.

    python dispatch_bg.py --plan wave1 [--slots 3]
A job is (node, system, seed, arms, d2select, needs) and starts once its 'needs' file exists.
Finished arms are skipped by run_bg.py itself (per-arm result files)."""
import argparse, json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
NODES = ["C3-partial", "C3-mismatch30"]
SYSTEMS = ["quad_tank", "duffing", "battery", "lorenz", "vdp", "bioreactor"]   # slowest first
W1 = ["A0", "A1", "A2", "A3", "A4", "A4-S", "A4-P", "A4-H", "D1", "D3"]

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True); ap.add_argument("--slots", type=int, default=3)
ap.add_argument("--gpus", default="5,6,7"); ap.add_argument("--arms", default="")
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--nodes", default=",".join(NODES))
ap.add_argument("--systems", default=",".join(SYSTEMS))
a = ap.parse_args()
gpus = [int(g) for g in a.gpus.split(",")]
assert set(gpus) <= {5, 6, 7}, "GPUs 5, 6, 7 only"
seeds = [int(s) for s in a.seeds.split(",")]
nodes, systems = a.nodes.split(","), a.systems.split(",")
jobs = []
if a.plan == "wave1":
    for s in seeds:
        for n in nodes:
            for sy in systems:
                jobs.append({"node": n, "system": sy, "seed": s, "arms": W1, "d2select": s == 0, "needs": None})
    for s in seeds:
        for n in nodes:
            for sy in systems:
                jobs.append({"node": n, "system": sy, "seed": s, "arms": ["D2"], "d2select": False,
                             "needs": str(HERE / "runs" / n / sy / "d2_selection.json")})
elif a.plan == "split":                      # seeds 1-2: main stack first, then D2, then leave-one-out
    MAIN = ["A0", "A1", "A2", "A3", "A4", "D1", "D3"]
    for arms in (MAIN, ["D2"], ["A4-S", "A4-P", "A4-H"]):
        for s in seeds:
            for n in nodes:
                for sy in systems:
                    jobs.append({"node": n, "system": sy, "seed": s, "arms": arms, "d2select": False,
                                 "needs": str(HERE / "runs" / n / sy / "d2_selection.json") if arms == ["D2"] else None})
elif a.plan == "arms":                       # explicit arm list on every (node, system, seed)
    arms = a.arms.split(",")
    for s in seeds:
        for n in nodes:
            for sy in systems:
                jobs.append({"node": n, "system": sy, "seed": s, "arms": arms, "d2select": False, "needs": None})
else:
    raise SystemExit(f"unknown plan {a.plan}")


def done(j):
    d = HERE / "runs" / j["node"] / j["system"] / f"seed{j['seed']}"
    ok = all((d / f"{x}.json").exists() for x in j["arms"])
    if j["d2select"]:
        ok = ok and (HERE / "runs" / j["node"] / j["system"] / "d2_selection.json").exists()
    return ok


jobs = [j for j in jobs if not done(j)]
(HERE / "logs").mkdir(exist_ok=True)
slots = [(g, i) for i in range(a.slots) for g in gpus]
running, fails = {}, {}
print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs on {len(slots)} slots", flush=True)
while jobs or running:
    for sl in slots:
        if sl in running and running[sl][0].poll() is not None:
            p, j, t0 = running.pop(sl)
            print(f"[{time.strftime('%H:%M:%S')}] done {j['node']} {j['system']} s{j['seed']} {','.join(j['arms'])} "
                  f"rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
            if p.returncode != 0 and not done(j):
                k = (j["node"], j["system"], j["seed"], tuple(j["arms"]))
                fails[k] = fails.get(k, 0) + 1
                if fails[k] < 3:
                    jobs.append(j)
                    print(f"   FAILED, requeued ({fails[k]})", flush=True)
                else:
                    print("   FAILED 3x, dropped", flush=True)
        ext = subprocess.run(["pgrep", "-af", "run_bg.py --node"], capture_output=True, text=True).stdout.splitlines()
        ext = [l for l in ext if "pgrep" not in l]
        own = {r[0].pid for r in running.values()}
        ext_busy = set()
        for l in ext:
            pid, *args = l.split()
            if int(pid) in own:
                continue
            try:
                ext_busy.add((args[args.index("--node") + 1], args[args.index("--system") + 1], int(args[args.index("--seed") + 1])))
            except Exception:
                pass
        if sl not in running and len(ext) < len(slots):
            ready = [j for j in jobs if j["needs"] is None or Path(j["needs"]).exists()]
            busy = {(r[1]["node"], r[1]["system"], r[1]["seed"]) for r in running.values()}
            ready = [j for j in ready if (j["node"], j["system"], j["seed"]) not in busy | ext_busy]
            if ready:
                j = ready[0]
                jobs.remove(j)
                tag = f"{j['node']}_{j['system']}_s{j['seed']}_{'-'.join(j['arms'])[:40]}"
                log = open(HERE / "logs" / f"{tag}.log", "a")
                cmd = [PY, str(HERE / "run_bg.py"), "--node", j["node"], "--system", j["system"],
                       "--seed", str(j["seed"]), "--arms", ",".join(j["arms"]), "--device", "cuda:0"]
                if j["d2select"]:
                    cmd.append("--d2select")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(sl[0]), OMP_NUM_THREADS="2")
                running[sl] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE), j, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] start {tag} on gpu {sl[0]}", flush=True)
    time.sleep(15)
print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)
