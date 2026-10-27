"""Queue runner: one process per (node, system, seed, arm), <= 3 per GPU on GPUs 0-4.
If a GPU process NOT from the project .venv appears on 0-4, new jobs go to 5-7 only.

    python dispatch.py jobs.txt      (lines: node system seed arm)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
PER_GPU = 3


def foreign_on_low():
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout
        uu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout
        idx = {l.split(",")[1].strip(): int(l.split(",")[0]) for l in uu.strip().splitlines()}
        for l in q.strip().splitlines():
            pid, u = [x.strip() for x in l.split(",")]
            if idx.get(u, 9) <= 4:
                exe = subprocess.run(["ps", "-o", "args=", "-p", pid], capture_output=True, text=True).stdout
                try:
                    cwd = str(Path(f"/proc/{pid}/cwd").resolve())
                except Exception:
                    cwd = ""
                ours = "/TMLR/hyperkkl/.venv/" in exe or (".venv/bin/python" in exe and "/TMLR/hyperkkl" in cwd)
                if exe and not ours:
                    return True
    except Exception:
        return False
    return False


def mine():
    """(per-GPU count, set of running job keys) over ALL run_tv.py processes (any dispatcher)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    cnt, keys = {}, set()
    for l in out.splitlines():
        a = l.split()
        if "run_tv.py" not in l or not all(t in a for t in ("--device", "--node", "--system", "--seed", "--arms")):
            continue
        a = l.split()
        g = int(a[a.index("--device") + 1].split(":")[1])
        cnt[g] = cnt.get(g, 0) + 1
        keys.add((a[a.index("--node") + 1], a[a.index("--system") + 1], a[a.index("--seed") + 1],
                  a[a.index("--arms") + 1]))
    return cnt, keys


def main():
    jobs = [l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    running = {}                     # Popen -> gpu
    log = open(ROOT / "logs" / "dispatch.log", "a")
    while jobs or running:
        for p in [p for p in running if p.poll() is not None]:
            log.write(f"{time.strftime('%H:%M:%S')} done rc={p.returncode} {p.args[-6:]}\n"); log.flush()
            running.pop(p)
        gpus = [5, 6, 7] if foreign_on_low() else [0, 1, 2, 3, 4]
        while jobs:
            cnt, keys = mine()
            load = {g: cnt.get(g, 0) for g in gpus}
            g = min(gpus, key=lambda x: load[x])
            if load[g] >= PER_GPU:
                break
            node, system, seed, arm = jobs.pop(0)
            out = ROOT / "runs" / node / system / f"seed{seed}" / f"{arm}.json"
            if out.exists() or (node, system, seed, arm) in keys:
                continue
            lf = open(ROOT / "logs" / f"{arm}_{node}_{system}_s{seed}.log", "a")
            p = subprocess.Popen([PY, str(HERE / "run_tv.py"), "--node", node, "--system", system, "--seed", seed,
                                  "--arms", arm, "--device", f"cuda:{g}"], stdout=lf, stderr=subprocess.STDOUT)
            running[p] = g
            log.write(f"{time.strftime('%H:%M:%S')} start gpu{g} {node} {system} s{seed} {arm}\n"); log.flush()
        time.sleep(20)


if __name__ == "__main__":
    main()
