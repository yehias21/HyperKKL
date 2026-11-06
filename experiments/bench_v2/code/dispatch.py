"""Queue dispatcher for bench_v2: GPUs 2-7 ONLY, <= 3 processes per GPU, counting EVERY compute process that
nvidia-smi reports on a GPU (ours or other agents'). Spreads round-robin over all six GPUs.

    python dispatch.py jobs.txt [--name NAME]

One job per line:  <done-file> :: <command run in code/>.  A job whose done-file exists is skipped; a failed job is
retried once. GPU n is skipped while experiments/GPU_n_RESERVED_* or (for n <= 4) experiments/GPU_0_4_BLOCKED exists.
Re-reads jobs.txt every cycle, so appended lines are picked up. Log: logs/dispatch_<name>.log. Stop it with its PID.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
HERE = CODE.parent
EXP = HERE.parent
PY = str(EXP.parent / ".venv" / "bin" / "python")
GPUS = [2, 3, 4, 5, 6, 7]
PER_GPU = 3
CPU_SLOTS = 0          # --cpu N: run jobs on the CPU only, N at a time (no GPU)


def stamp():
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 4 * 3600))      # Dubai


def gpu_load():
    try:
        uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                            text=True, timeout=60).stdout.strip().splitlines()
        idx = {u.split(", ")[1]: int(u.split(", ")[0]) for u in uu}
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    except Exception:
        return None
    n = {g: 0 for g in idx.values()}
    for a in apps:
        if a.strip():
            n[idx[a.split(", ")[0]]] += 1
    return n


def blocked(g):
    if list(EXP.glob(f"GPU_{g}_RESERVED_*")):
        return True
    return g <= 4 and (EXP / "GPU_0_4_BLOCKED").exists()


def host_running():
    """Commands of this queue already running on the host (e.g. left by a previous dispatcher): never start twice."""
    try:
        out = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "args="], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    pre = PY + " "
    return {l[len(pre):].strip() for l in out.splitlines() if l.startswith(pre)}


def read_jobs(f):
    out = []
    for l in open(f):
        l = l.strip()
        if not l or l.startswith("#") or "::" not in l:
            continue
        d, c = l.split("::", 1)
        out.append((d.strip(), c.strip()))
    return out


def main():
    jf = sys.argv[1]
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else Path(jf).stem
    log = open(HERE / "logs" / f"dispatch_{name}.log", "a")
    tries, running, mine = {}, {}, {g: 0 for g in GPUS}
    log.write(f"{stamp()} dispatcher pid {os.getpid()} start {jf}\n"); log.flush()
    rr = 0
    while True:
        for p in list(running):
            if p.poll() is not None:
                (d, c), g = running.pop(p)
                if g >= 0:
                    mine[g] -= 1
                log.write(f"{stamp()} done rc {p.returncode} gpu {g}: {c}\n"); log.flush()
        jobs = read_jobs(jf)
        active = {c for (d, c), g in running.values()} | host_running()
        pend = [(d, c) for d, c in jobs if not (HERE / d).exists() and c not in active and tries.get(c, 0) < 2]
        if not pend and not running:
            break
        load = gpu_load() if CPU_SLOTS == 0 else {}
        if CPU_SLOTS:
            while pend and len(running) < CPU_SLOTS:
                d, c = pend.pop(0)
                tries[c] = tries.get(c, 0) + 1
                tag = "".join(ch if ch.isalnum() else "_" for ch in c.split(" ", 1)[-1])[:120]
                out = open(HERE / "logs" / "jobs" / f"{tag}.out", "a")
                p = subprocess.Popen([PY] + shlex.split(c), cwd=CODE, stdout=out, stderr=subprocess.STDOUT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                                          "MKL_NUM_THREADS": "2"})
                running[p] = ((d, c), -1)
                log.write(f"{stamp()} start cpu pid {p.pid} try {tries[c]}: {c}\n"); log.flush()
        elif load is not None:
            order = GPUS[rr % len(GPUS):] + GPUS[:rr % len(GPUS)]
            for g in order:
                if not pend:
                    break
                if blocked(g):
                    continue
                if max(load.get(g, 0), mine[g]) < PER_GPU:
                    d, c = pend.pop(0)
                    tries[c] = tries.get(c, 0) + 1
                    tag = "".join(ch if ch.isalnum() else "_" for ch in c.split(" ", 1)[-1])[:120]
                    out = open(HERE / "logs" / "jobs" / f"{tag}.out", "a")
                    p = subprocess.Popen([PY] + shlex.split(c), cwd=CODE, stdout=out, stderr=subprocess.STDOUT,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "OMP_NUM_THREADS": "2",
                                              "MKL_NUM_THREADS": "2"})
                    running[p] = ((d, c), g)
                    mine[g] += 1
                    load[g] = load.get(g, 0) + 1
                    log.write(f"{stamp()} start gpu {g} pid {p.pid} try {tries[c]}: {c}\n"); log.flush()
            rr += 1
        time.sleep(15)
    log.write(f"{stamp()} queue empty\n"); log.flush()


if __name__ == "__main__":
    (HERE / "logs" / "jobs").mkdir(parents=True, exist_ok=True)
    if "--cpu" in sys.argv:
        CPU_SLOTS = int(sys.argv[sys.argv.index("--cpu") + 1])
    main()
