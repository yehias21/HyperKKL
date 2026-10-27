"""Scaling-study launcher.

    nohup .venv/bin/python experiments/scaling/code/launch.py > experiments/scaling/logs/launch.log 2>&1 &

Rules (task + coordinator): GPUs 0-7, <= 3 of OUR processes per GPU (least-loaded first);
GPUs 0-4 get no new job while experiments/GPU_0_4_BLOCKED exists or a process whose cmdline lacks
TMLR/hyperkkl is visible there; a job killed by the GPU guard (signal while the flag exists) is
re-queued for GPUs 5-7, not counted as a failure; cuFFT transients retried (max 3).
Data units run on the CPU (<= CPU_SLOTS at once) before the GPU units that need them.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SC = HERE.parent
ROOT = SC.parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
FLAG = ROOT / "experiments" / "GPU_0_4_BLOCKED"
GPUS = list(range(8))
PER_GPU = 3
CPU_SLOTS = 10
sys.path.insert(0, str(HERE))
from plan import jobs  # noqa: E402


def status(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    with open(SC / "STATUS.md", "a") as f:
        f.write(f"- {datetime.now(ZoneInfo('Asia/Dubai')):%Y-%m-%d %H:%M} Dubai: [launcher] {msg}\n")


def gpu_procs():
    """(ours per GPU, GPUs 0-4 with a foreign process)."""
    ours, bad = {}, set()
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
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
            except Exception:
                continue
            g = idx.get(u, 99)
            if "experiments/scaling" in cmd:
                ours[g] = ours.get(g, 0) + 1
            elif cmd and "TMLR/hyperkkl" not in cmd and g <= 4:
                bad.add(g)
    except Exception:
        pass
    return ours, bad


def main():
    J = jobs()
    if "--data-only" in sys.argv:
        J = [j for j in J if j["gpu"] is False]
    running = {}                         # name -> (proc, gpu or None, job, logfile)
    tries = {j["name"]: 0 for j in J}
    avoid04 = set()
    bad_prev = set()
    status(f"started: {len(J)} jobs ({sum(1 for j in J if not j['done']())} to do)")
    while True:
        for name, (p, g, j, lf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[name]
            txt = Path(lf).read_text(errors="ignore")[-20000:]
            if j["done"]():
                status(f"done {name} (gpu {g})")
            elif rc < 0 and g is not None and g <= 4 and FLAG.exists():
                avoid04.add(name); tries[name] -= 1
                status(f"{name} killed on gpu {g} by the GPU guard: re-queued for GPUs 5-7")
            elif "CUFFT_INTERNAL_ERROR" in txt and tries[name] < 3:
                status(f"cuFFT transient on {name}, retry")
            else:
                tries[name] = 99
                status(f"FAILED {name} rc {rc} (see {lf})")
        pending = [j for j in J if not j["done"]() and j["name"] not in running and tries[j["name"]] < 4
                   and all(d() for d in j["deps"])]
        if not pending and not running:
            status("finished")
            return
        ours, bad = gpu_procs()
        bad_now = bad & bad_prev                 # foreign seen on 2 consecutive checks
        bad_prev = bad
        blocked = FLAG.exists()
        n_cpu = sum(1 for v in running.values() if v[1] is None)
        for j in pending:
            if j["gpu"] is False:
                if n_cpu >= CPU_SLOTS:
                    continue
                g = None
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
                n_cpu += 1
            else:
                load = {x: max(ours.get(x, 0), sum(1 for v in running.values() if v[1] == x)) for x in GPUS}
                ok = [x for x in GPUS if load[x] < PER_GPU and not (x <= 4 and (blocked or x in bad_now
                                                                                   or j["name"] in avoid04))]
                if not ok:
                    continue
                g = min(ok, key=lambda x: (load[x], -x))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
                ours[g] = load[g] + 1
            tries[j["name"]] += 1
            lf = str(SC / "logs" / f"{j['name']}.try{tries[j['name']]}.log")
            p = subprocess.Popen(j["cmd"], cwd=str(ROOT), stdout=open(lf, "w"), stderr=subprocess.STDOUT, env=env)
            running[j["name"]] = (p, g, j, lf)
            status(f"start {j['name']} on {'cpu' if g is None else f'gpu {g}'} (try {tries[j['name']]})")
            time.sleep(5)
        time.sleep(30)


if __name__ == "__main__":
    main()
