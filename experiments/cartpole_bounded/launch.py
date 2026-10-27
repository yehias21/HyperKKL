"""Run the cartpole_b / two_link_arm_b jobs: <= 2 of our processes per GPU, cuFFT-transient retries (max 3),
yield rule (a GPU in 0-4 showing a process that is not from this project's .venv gets no new job).

    nohup .venv/bin/python experiments/cartpole_bounded/launch.py > experiments/cartpole_bounded/logs/launch.log 2>&1 &
"""
import subprocess
import time
from pathlib import Path

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
HERE = ROOT / "experiments" / "cartpole_bounded"
LOG = HERE / "logs"
PY = str(ROOT / ".venv" / "bin" / "python")
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
PER_GPU = 2
import sys
PLANTS = sys.argv[1:] or ["two_link_arm_b", "cartpole_b"]
SEEDS = [0, 1, 2]


def status(msg):
    with open(HERE / "STATUS.md", "a") as f:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        f.write(f"{datetime.now(ZoneInfo('Asia/Dubai')):%Y-%m-%d %H:%M} Dubai  {msg}\n")


def tree_done(node, s, k):
    return (ROOT / "experiments" / "tree" / node / "runs" / s / f"seed{k}" / "results.json").exists()


def soap_done(s, k):
    d = HERE / "soap_runs" / "TEST-C3-B" / s / f"seed{k}"
    return (d / "A2.json").exists() and (d / "A4.json").exists()


JOBS = []
for s in PLANTS:
    for k in SEEDS:
        JOBS.append({"name": f"C3_{s}_{k}", "done": (lambda s=s, k=k: tree_done("TEST-C3-B", s, k)), "dep": None,
                     "cmd": [PY, "-m", "hyperkkl.job", "--node", "TEST-C3-B", "--system", s, "--seed", str(k), "--device", "cuda:0"]})
for s in PLANTS:
    for k in SEEDS:
        JOBS.append({"name": f"C1_{s}_{k}", "done": (lambda s=s, k=k: tree_done("TEST-C1-B", s, k)), "dep": None,
                     "cmd": [PY, "-m", "hyperkkl.job", "--node", "TEST-C1-B", "--system", s, "--seed", str(k), "--device", "cuda:0"]})
for s in PLANTS:
    for k in SEEDS:
        JOBS.append({"name": f"SOAP_{s}_{k}", "done": (lambda s=s, k=k: soap_done(s, k)),
                     "dep": (lambda s=s, k=k: tree_done("TEST-C3-B", s, k)),
                     "cmd": [PY, str(HERE / "code" / "run_soap.py"), "--node", "TEST-C3-B", "--system", s, "--seed", str(k),
                             "--arms", "A2,A4", "--device", "cuda:0"]})


def foreign_gpus():
    """GPUs (0-4) with a compute process not started from this project's .venv."""
    try:
        idx = {}
        for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines():
            i, u = [x.strip() for x in line.split(",")]
            idx[u] = int(i)
        bad = set()
        for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True).splitlines():
            pid, u = [x.strip() for x in line.split(",")]
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
                cwd = str(Path(f"/proc/{pid}/cwd").resolve())
            except Exception:
                continue
            if "hyperkkl" not in cmd and "TMLR" not in cwd and idx.get(u, 9) <= 4:
                bad.add(idx[u])
        return bad
    except Exception:
        return set()


def ours_per_gpu():
    """Our processes per GPU as nvidia-smi sees them (both launchers): cmdline names a bounded plant."""
    out = {}
    try:
        idx = {}
        for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines():
            i, u = [x.strip() for x in line.split(",")]
            idx[u] = int(i)
        for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True).splitlines():
            pid, u = [x.strip() for x in line.split(",")]
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
            except Exception:
                continue
            if "cartpole_b" in cmd or "two_link_arm_b" in cmd:
                out[idx[u]] = out.get(idx[u], 0) + 1
    except Exception:
        pass
    return out


def main():
    running = {}          # name -> (proc, gpu, job, logfile)
    tries = {j["name"]: 0 for j in JOBS}
    status(f"launcher started: {len(JOBS)} jobs")
    while True:
        for name, (p, g, j, lf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[name]
            txt = Path(lf).read_text(errors="ignore")[-20000:]
            if j["done"]():
                status(f"done {name} (gpu {g})")
            elif "CUFFT_INTERNAL_ERROR" in txt and tries[name] < 3:
                status(f"cuFFT transient on {name}, retry")
            else:
                tries[name] = 99
                status(f"FAILED {name} rc {rc} (see {lf})")
        pending = [j for j in JOBS if not j["done"]() and j["name"] not in running and tries[j["name"]] < 4
                   and (j["dep"] is None or j["dep"]())]
        if not pending and not running:
            if all(j["done"]() for j in JOBS) or all(tries[j["name"]] >= 4 or j["done"]() for j in JOBS):
                status("launcher finished")
                return
        bad = foreign_gpus()
        for j in pending:
            ext = ours_per_gpu()
            load = {g: max(ext.get(g, 0), sum(1 for v in running.values() if v[1] == g)) for g in GPUS}
            free = [g for g in GPUS if load[g] < PER_GPU and g not in bad]
            if not free:
                break
            g = min(free, key=lambda x: (load[x], x))
            tries[j["name"]] += 1
            lf = str(LOG / f"{j['name']}.try{tries[j['name']]}.log")
            env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            p = subprocess.Popen(j["cmd"], cwd=str(ROOT), stdout=open(lf, "w"), stderr=subprocess.STDOUT, env=env)
            running[j["name"]] = (p, g, j, lf)
            status(f"start {j['name']} on gpu {g} (try {tries[j['name']]})")
            time.sleep(20)                  # stagger starts (cuFFT plan errors when many start at once)
        time.sleep(30)


if __name__ == "__main__":
    main()
