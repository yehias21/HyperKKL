"""Grey-box refits on the CPU, N concurrent processes; a job starts once its arm's data exist."""
import os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
need = {"X0": [], "X1": ["long_orig"], "X2": ["slow_T1"], "X3": ["long_orig", "slow_T3"]}
jobs = [(n, s, a) for a in ("X0", "X2", "X1", "X3") for s in (0, 1, 2) for n in ("C3-partial", "C3-mismatch30")]
done = lambda n, s, a: (HERE / "runs" / n / f"seed{s}" / f"{a}_gb.json").exists() or \
    ((HERE / "runs" / n / f"seed{s}" / f"{a}.json").exists() and '"greybox_ekf"' in (HERE / "runs" / n / f"seed{s}" / f"{a}.json").read_text())
ready = lambda n, s, a: all((HERE / "data" / f"{n}_s{s}_{p}.pt").exists() for p in need[a])
running = []
while jobs or running:
    running = [r for r in running if r[0].poll() is None]
    jobs = [j for j in jobs if not done(*j)]
    busy = {r[1] for r in running}
    while len(running) < N:
        c = [j for j in jobs if ready(*j) and j not in busy]
        if not c:
            break
        j = c[0]; jobs.remove(j)
        n, s, a = j
        log = open(HERE / "logs" / f"gb_{n}_s{s}_{a}.log", "a")
        p = subprocess.Popen([PY, str(HERE / "run_qt.py"), "--node", n, "--seed", str(s), "--arm", a, "--gb-only", "--device", "cpu"],
                             stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4"))
        running.append((p, j)); print(time.strftime("%H:%M:%S"), "start", j, flush=True)
    time.sleep(20)
print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
