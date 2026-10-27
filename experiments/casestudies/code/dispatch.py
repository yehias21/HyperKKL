"""GPU dispatcher for the case-study jobs (run_cs.py). <= PER_GPU of our processes per GPU, least-loaded GPU first;
yield rule: a GPU in 0-4 on which a compute process whose command line lacks 'TMLR/hyperkkl' was seen on two
consecutive checks gets no new job (running ones finish). cuFFT / CUDA-graph transients are retried (max 3).

    nohup ../../../.venv/bin/python dispatch.py [--cases A,B] [--levels ...] [--families ...] [--seeds 0,1,2,3,4] \
        > ../logs/dispatch.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CS = HERE.parent
ROOT = CS.parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
LOG = CS / "logs"
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PER_GPU = 3
ORDER_LEVELS = ["realistic", "exact", "partial", "mm30", "mm10"]
ORDER_FAMS = ["filters", "greybox", "hk", "ude5k", "gru140", "node5k", "nlox140"]


def status(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    with open(CS / "STATUS.md", "a") as f:
        f.write(f"{datetime.now(ZoneInfo('Asia/Dubai')):%Y-%m-%d %H:%M} Dubai  [dispatch] {msg}\n")


def done(j):
    return (CS / "runs" / j["case"] / j["level"] / f"seed{j['seed']}" / f"{j['family']}.json").exists()


def gpu_procs():
    """{gpu: [(pid, cmdline)]} of every compute process."""
    out = {g: [] for g in GPUS}
    try:
        idx = {}
        for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                            text=True).splitlines():
            i, u = [x.strip() for x in line.split(",")]
            idx[u] = int(i)
        for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                             "--format=csv,noheader"], text=True).splitlines():
            pid, u = [x.strip() for x in line.split(",")]
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore").replace("\0", " ")
            except Exception:
                cmd = None
            if u in idx:
                out[idx[u]].append((pid, cmd))
    except Exception:
        pass
    return out


_seen_foreign = {}
GUARD_FLAG = ROOT / "experiments" / "GPU_0_4_BLOCKED"     # written by experiments/gpu_guard.sh


def foreign(procs):
    bad = set()
    for g in range(5):
        f = any(c is not None and c.strip() and "TMLR/hyperkkl" not in c for _, c in procs.get(g, []))
        _seen_foreign[g] = _seen_foreign.get(g, 0) + 1 if f else 0
        if _seen_foreign[g] >= 2:
            bad.add(g)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="A,B")
    ap.add_argument("--levels", default=",".join(ORDER_LEVELS))
    ap.add_argument("--families", default=",".join(ORDER_FAMS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--per_gpu", type=int, default=PER_GPU)
    a = ap.parse_args()
    jobs = [{"case": c, "level": l, "seed": int(s), "family": f}
            for s in a.seeds.split(",") for l in a.levels.split(",") for c in a.cases.split(",")
            for f in a.families.split(",")]
    for j in jobs:
        j["name"] = f"{j['case']}_{j['level']}_s{j['seed']}_{j['family']}"
    running, tries = {}, {j["name"]: 0 for j in jobs}
    status(f"dispatcher started: {len(jobs)} jobs ({sum(done(j) for j in jobs)} already done), <= {a.per_gpu}/GPU")
    last_bad = set()
    while True:
        for name, (p, g, j, lf) in list(running.items()):
            if p.poll() is None:
                continue
            del running[name]
            txt = Path(lf).read_text(errors="ignore")[-20000:]
            if done(j):
                status(f"done {name} (gpu {g})")
            elif p.returncode in (-15, -9, 143, 137) and g <= 4 and tries[name] < 6:
                tries[name] = max(0, tries[name] - 1)          # killed by the GPU guard: re-run (on 5-7), not a failure
                status(f"{name} killed on gpu {g} (guard / SIGTERM), re-queued")
            elif ("CUFFT_INTERNAL_ERROR" in txt or "CUDA error" in txt or "cudaErrorLaunchFailure" in txt) \
                    and tries[name] < 3:
                status(f"transient on {name}, retry")
            else:
                tries[name] = 99
                status(f"FAILED {name} rc {p.returncode} (see {lf})")
        pending = [j for j in jobs if not done(j) and j["name"] not in running and tries[j["name"]] < 4]
        if not pending and not running:
            status("dispatcher finished")
            return
        procs = gpu_procs()
        bad = foreign(procs)
        if GUARD_FLAG.exists():
            bad = bad | {0, 1, 2, 3, 4}
        if bad != last_bad:
            status(f"foreign processes on GPUs {sorted(bad)}: no new jobs there" if bad else "GPUs 0-4 clear again")
            last_bad = bad
        for j in pending:
            ours = {g: sum(1 for v in running.values() if v[1] == g) for g in GPUS}
            total = {g: len(procs.get(g, [])) for g in GPUS}
            if GUARD_FLAG.exists():                         # checked before every launch
                bad = bad | {0, 1, 2, 3, 4}
            free = [g for g in GPUS if ours[g] < a.per_gpu and g not in bad]
            if not free:
                break
            g = min(free, key=lambda x: (ours[x], total[x], x))
            tries[j["name"]] += 1
            lf = str(LOG / f"{j['name']}.try{tries[j['name']]}.log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            cmd = [PY, str(HERE / "run_cs.py"), "--case", j["case"], "--level", j["level"], "--seed", str(j["seed"]),
                   "--family", j["family"], "--device", "cuda:0"]
            p = subprocess.Popen(cmd, cwd=str(HERE), stdout=open(lf, "w"), stderr=subprocess.STDOUT, env=env)
            running[j["name"]] = (p, g, j, lf)
            status(f"start {j['name']} on gpu {g} (try {tries[j['name']]})")
            time.sleep(10)
            procs = gpu_procs()
        time.sleep(30)


if __name__ == "__main__":
    main()
