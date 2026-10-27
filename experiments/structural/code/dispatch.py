"""Job queue for the structural campaign: <= 2 of our processes per GPU on 0-7; a GPU in 0-4 is
skipped while any process of another project (cmdline without /papers/TMLR/) runs on it.

    nohup python dispatch.py > ../dispatch.log 2>&1 &
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE.parents[2] / ".venv" / "bin" / "python")
RUNS = HERE.parent / "runs"
LOGS = HERE.parent / "logs"
LOGS.mkdir(exist_ok=True)
ORDER = ["quad_tank", "battery", "cartpole", "bioreactor", "vdp", "lorenz", "highway_traffic"]
JOBS = [(s, t, k) for k in (0, 1, 2) for s in ORDER for t in ("ref", "struct")]
JOBS = sorted(JOBS, key=lambda j: (j[0] == "highway_traffic", j[2]))      # highway last (PREREG)
GPUS = [int(g) for g in os.environ.get("STRUCT_GPUS", "0,1,2,3,4,5,6,7").split(",")]
PER_GPU = 2


def foreign_on(gpu: int) -> bool:
    try:
        uu = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout
        for line in q.splitlines():
            u, pid = [x.strip() for x in line.split(",")]
            if u != uu:
                continue
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore")
            except OSError:
                cmd = ""
            if "/papers/TMLR/" not in cmd and "hyperkkl" not in cmd:
                return True
    except Exception:
        return False
    return False


def done(j):
    s, t, k = j
    return (RUNS / s / t / f"seed{k}" / "results.json").exists()


def main():
    running = {}            # (s,t,k) -> (Popen, gpu)
    fails = {}
    todo = [j for j in JOBS if not done(j)]
    print(time.strftime("%H:%M:%S"), f"{len(todo)} jobs to run on GPUs {GPUS}", flush=True)
    while todo or running:
        for j, (p, g) in list(running.items()):
            if p.poll() is not None:
                del running[j]
                ok = done(j)
                print(time.strftime("%H:%M:%S"), "finished", j, "gpu", g, "ok" if ok else f"FAILED rc={p.returncode}", flush=True)
                if not ok:
                    fails[j] = fails.get(j, 0) + 1
                    if fails[j] < 2:
                        todo.insert(0, j)
        load = {g: 0 for g in GPUS}
        for _, g in running.values():
            load[g] += 1
        for g in GPUS:
            while todo and load[g] < PER_GPU:
                if g <= 4 and foreign_on(g):
                    break
                j = todo.pop(0)
                s, t, k = j
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                lf = open(LOGS / f"{s}_{t}_s{k}.log", "a")
                p = subprocess.Popen([PY, str(HERE / "run_struct.py"), "--system", s, "--tier", t, "--seed", str(k),
                                      "--device", "cuda:0"], env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE))
                running[j] = (p, g)
                load[g] += 1
                print(time.strftime("%H:%M:%S"), "started", j, "gpu", g, flush=True)
        time.sleep(20)
    print(time.strftime("%H:%M:%S"), "all done; failures:", fails, flush=True)


if __name__ == "__main__":
    main()
