"""Run every baselines2 job on GPUs 0-7, <= PER_GPU of our processes per GPU.

Before each launch: if nvidia-smi shows a compute process on GPUs 0-4 whose command line does not
contain 'TMLR/hyperkkl', GPUs 0-4 are dropped for every later launch (running jobs finish).

    python dispatch.py [--baselines a,b] [--gpus 0-7] [--per-gpu 4]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAG = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/GPU_0_4_BLOCKED")   # gpu_guard.sh
ROOT = HERE.parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"

DEV = [("C3-partial", s) for s in ("battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp")] + \
      [("C3-mismatch30", s) for s in ("battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp")]
TEST = [("TEST-C3", s) for s in ("fhn", "highway_traffic", "network", "rossler", "two_link_arm")] + \
       [("TEST-C3-B", "cartpole_b")]
CELLS = DEV + TEST
ORDER = ["greybox", "gru140", "ude5k", "node5k", "nlox140", "ncde140", "ude140k", "node140k", "gru500", "nlox500"]
SEEDS = [0, 1, 2, 3, 4]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def foreign_on_low() -> list:
    """(gpu index, pid, cmd) of compute processes on GPUs 0-4 that are not ours / not TMLR/hyperkkl."""
    try:
        idx = dict(l.split(", ") for l in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"], text=True).strip().splitlines())
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                       text=True).strip().splitlines()
    except Exception as e:                           # be conservative: treat as foreign
        return [(-1, -1, f"nvidia-smi failed: {e}")]
    out = []
    for l in apps:
        if not l.strip():
            continue
        uuid, pid = [x.strip() for x in l.split(",")]
        g = int(idx.get(uuid, -1))
        if g < 0 or g > 4:
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            cmd = "?"
        if not cmd.strip() or cmd == "?":           # exiting process (cmdline already gone)
            continue
        if "TMLR/hyperkkl" not in cmd and "b2.job" not in cmd:
            out.append((g, int(pid), cmd[:120]))
    return out


def adopted() -> dict:
    """b2.job processes already running (e.g. started by a previous dispatcher): pid -> (job, gpu)."""
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            a = (p / "cmdline").read_bytes().split(b"\0")
            env = dict(x.split(b"=", 1) for x in (p / "environ").read_bytes().split(b"\0") if b"=" in x)
        except OSError:
            continue
        a = [x.decode() for x in a if x]
        if "b2.job" not in a or "--smoke" in a:
            continue
        kv = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--")}
        try:
            g = int(env.get(b"CUDA_VISIBLE_DEVICES", b"-1"))
            out[int(p.name)] = ((kv["--baseline"], kv["--node"], kv["--system"], int(kv["--seed"])), g)
        except (KeyError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=",".join(ORDER))
    ap.add_argument("--gpus", default="0-7")
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.gpus.split("-"))
    gpus = list(range(lo, hi + 1))
    cells = CELLS if a.cells == "all" else [c for c in CELLS if c[1] in a.cells.split(",")]
    jobs = [(b, n, s, sd) for b in a.baselines.split(",") for sd in SEEDS for (n, s) in cells]
    if a.real:                                       # real battery GRU / NCDE (PREREG section 5), first
        jobs = [(f"real_{k}", "REAL", "battery", sd) for k in ("gru", "ncde") for sd in (0, 1, 2)] + jobs
    run_dir = ROOT / "runs"

    def done_file(j):
        if j[0].startswith("real_"):
            return ROOT / "real" / f"{j[0][5:]}_seed{j[3]}.json"
        return run_dir / j[1] / j[2] / f"seed{j[3]}" / f"{j[0]}.json"
    todo = [j for j in jobs if not done_file(j).exists()]
    log(f"{len(jobs)} jobs, {len(todo)} to run, gpus {gpus}, per gpu {a.per_gpu}")
    running = {}                                     # Popen -> (job, gpu)
    adopt = adopted()                                # pid -> (job, gpu), not our children
    todo = [j for j in todo if j not in {v[0] for v in adopt.values()}]
    log(f"adopted {len(adopt)} running jobs; {len(todo)} to launch")
    low_ok = True
    strikes = 0
    tries = {}
    (ROOT / "logs" / "jobs").mkdir(parents=True, exist_ok=True)
    while todo or running or adopt:
        def requeue_if_killed(j, g):
            b, n, s_, sd = j
            d = run_dir / n / s_ / f"seed{sd}"
            if done_file(j).exists() or (d / f"{b}.error.txt").exists() or b.startswith("real_"):
                return
            tries[j] = tries.get(j, 0) + 1
            if tries[j] <= 3:                          # killed (e.g. by gpu_guard): not a failure
                todo.insert(0, j)
                log(f"re-queued killed job {j} (try {tries[j]})")

        for p in list(running):
            if p.poll() is not None:
                j, g = running.pop(p)
                log(f"end rc={p.returncode} gpu{g} {j}")
                requeue_if_killed(j, g)
        for pid in list(adopt):
            if not Path(f"/proc/{pid}").exists():
                j, g = adopt.pop(pid)
                log(f"end (adopted) gpu{g} {j}")
                requeue_if_killed(j, g)
        if todo:
            if low_ok:
                f = foreign_on_low()
                strikes = strikes + 1 if f else 0
                if strikes >= 2:                     # seen on two consecutive checks
                    low_ok = False
                    log(f"FOREIGN process on GPUs 0-4 {f}: no more launches on 0-4")
            blocked = FLAG.exists()
            if blocked and not getattr(main, "_said", False):
                log("GPU_0_4_BLOCKED flag present: launches on GPUs 5-7 only while it exists")
            main._said = blocked
            usable = [g for g in gpus if (low_ok and not blocked) or g >= 5]
            load = {g: 0 for g in usable}
            for _, g in list(running.values()) + list(adopt.values()):
                if g in load:
                    load[g] += 1
            free = sorted((n, g) for g, n in load.items() if n < a.per_gpu)
            if free:
                g = free[0][1]
                j = todo.pop(0)
                b, n, s, sd = j
                lf = open(ROOT / "logs" / "jobs" / f"{n}_{s}_seed{sd}_{b}.log", "a")
                cmd = [PY, "real_run.py", "--kind", b[5:], "--seed", str(sd)] if b.startswith("real_") else \
                    [PY, "-m", "b2.job", "--node", n, "--system", s, "--seed", str(sd), "--baseline", b]
                p = subprocess.Popen(cmd, cwd=str(HERE), stdout=lf, stderr=subprocess.STDOUT,
                                     env={"CUDA_VISIBLE_DEVICES": str(g), "PATH": "/usr/bin:/bin",
                                          "HOME": "/home/demo", "OMP_NUM_THREADS": "2"})
                running[p] = (j, g)
                log(f"start gpu{g} {j} ({len(todo)} left)")
                time.sleep(2)
                continue
        time.sleep(10)
    log("all done")


if __name__ == "__main__":
    main()
