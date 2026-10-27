"""Dispatcher for the memory-design jobs: <= SLOTS of our processes per GPU, GPUs 0-4 preferred.
Yield rule: if a process NOT started from this project's .venv appears on GPUs 0-4, only 5-7 are used.

    python dispatch_md.py [--gpus 0,1,2,3,4] [--fallback 5,6,7] [--slots 2] [--systems ...] [--groups ...]
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
VENV = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv"
GROUPS = {"A": "M0,M3", "B": "M1", "C": "M2,M5,M4"}

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", default="0,1,2,3,4"); ap.add_argument("--fallback", default="5,6,7")
ap.add_argument("--slots", type=int, default=2)
ap.add_argument("--systems", default="quad_tank,battery,vdp,bioreactor")
ap.add_argument("--nodes", default="C3-partial,C3-mismatch30"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--groups", default="C,A,B")
a = ap.parse_args()
prim = [int(g) for g in a.gpus.split(",")]
fb = [int(g) for g in a.fallback.split(",")]


def ts():
    return time.strftime("%H:%M:%S")


def gpu_state():
    """(foreign process on 0-4?, {index: used MiB})"""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    idx, used = {}, {}
    for line in q:
        i, u, m = [x.strip() for x in line.split(",")]
        idx[u] = int(i); used[int(i)] = int(m)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    foreign = False
    for line in apps:
        if not line.strip():
            continue
        u, pid = [x.strip() for x in line.split(",")]
        if idx.get(u, 99) > 4:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except Exception:
            foreign = True; continue
        prog = cmd.split(" ")[0]
        own = prog.startswith(VENV) or (prog.endswith(".venv/bin/python") and cwd.startswith(str(Path(VENV).parent)))
        if not own:
            foreign = True
    return foreign, used


def done(j):
    out = ROOT / "runs" / j["node"] / j["system"] / f"seed{j['seed']}"
    return all((out / f"{arm}.json").exists() for arm in GROUPS[j["group"]].split(","))


jobs = [{"node": n, "system": s, "seed": int(sd), "group": g}
        for s in a.systems.split(",") for g in a.groups.split(",") for sd in a.seeds.split(",") for n in a.nodes.split(",")]
jobs = [j for j in jobs if not done(j)]
(ROOT / "logs").mkdir(exist_ok=True)
running, fails = {}, {}
print(f"[{ts()}] {len(jobs)} jobs", flush=True)
while jobs or running:
    for key in list(running):
        p, j, t0 = running[key]
        if p.poll() is not None:
            running.pop(key)
            print(f"[{ts()}] done {j} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
            if p.returncode != 0 and not done(j):
                k = (j["node"], j["system"], j["seed"], j["group"])
                fails[k] = fails.get(k, 0) + 1
                if fails[k] < 3:
                    jobs.append(j); print("   FAILED, requeued", flush=True)
                else:
                    print("   FAILED 3x, dropped", flush=True)
    foreign, used = gpu_state()
    allowed = fb if foreign else prim
    if foreign:
        print(f"[{ts()}] foreign process on GPUs 0-4: using {fb} only for new jobs", flush=True)
    for g in allowed:
        for i in range(a.slots):
            key = (g, i)
            if key in running or not jobs or used.get(g, 0) > 70000:
                continue
            j = jobs.pop(0)
            tag = f"{j['node']}_{j['system']}_s{j['seed']}_{j['group']}"
            log = open(ROOT / "logs" / f"{tag}.log", "a")
            cmd = [PY, str(HERE / "run_md.py"), "--node", j["node"], "--system", j["system"], "--seed", str(j["seed"]),
                   "--arms", GROUPS[j["group"]], "--device", "cuda:0"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS="2")
            running[key] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE), j, time.time())
            print(f"[{ts()}] start {tag} on gpu {g}", flush=True)
    time.sleep(30)
print(f"[{ts()}] ALL DONE", flush=True)
