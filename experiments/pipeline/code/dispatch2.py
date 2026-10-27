"""Dispatcher v2: GPUs 5,6,7 always; GPUs 0-4 only until the deadline (2026-10-26 05:45 CET).
Adopts jobs already running (found via /proc). No new job starts on 0-4 after START_CUTOFF, and any job
still on 0-4 at KILL_AT is terminated and requeued (finished sub-models are cached, so it resumes)."""
import datetime as dt
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python"
OUT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/pipeline/runs")
SYSTEMS = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]
SCEN = ["C3-partial", "C3-mismatch30", "C1-where"]
CET = dt.timezone(dt.timedelta(hours=1))
START_CUTOFF = dt.datetime(2026, 10, 26, 4, 45, tzinfo=CET)
KILL_AT = dt.datetime(2026, 10, 26, 5, 38, tzinfo=CET)
SLOTS = {"5": 3, "6": 3, "7": 3, "0": 2, "1": 2, "2": 2, "3": 2, "4": 2}
EARLY = {"0", "1", "2", "3", "4"}
logd = OUT.parent / "dispatch_logs"
logd.mkdir(parents=True, exist_ok=True)


def now():
    return dt.datetime.now(CET)


def log(m):
    print(f"{now():%H:%M:%S} {m}", flush=True)


def running_external():
    """{job: (pid, gpu)} for run_pipeline.py processes not started by this dispatcher."""
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")
            if not any(b"run_pipeline.py" in c for c in cmd) or b"--smoke" in cmd:
                continue
            a = [c.decode() for c in cmd]
            j = (a[a.index("--scenario") + 1], a[a.index("--system") + 1], int(a[a.index("--seed") + 1]))
            env = dict(e.split("=", 1) for e in (d / "environ").read_bytes().decode(errors="ignore").split("\0") if "=" in e)
            out[j] = (int(d.name), env.get("CUDA_VISIBLE_DEVICES"))
        except Exception:
            continue
    return out


def done(j):
    return (OUT / j[0] / j[1] / f"seed{j[2]}" / "results.json").exists()


jobs = [(sc, s, k) for sc in SCEN for k in (0, 1, 2) for s in SYSTEMS]
mine = {}          # job -> (Popen, gpu)
tries = {j: 0 for j in jobs}
log("dispatcher v2 start")
while True:
    ext = running_external()
    for j, (p, g) in list(mine.items()):
        if p.poll() is not None:
            log(f"done {j} rc {p.returncode} gpu {g}")
            del mine[j]
    busy = {j: g for j, (_, g) in ext.items()}
    busy.update({j: g for j, (_, g) in mine.items()})
    # deadline enforcement on GPUs 0-4
    if now() >= KILL_AT:
        for j, (p, g) in list(mine.items()):
            if g in EARLY:
                log(f"deadline: terminating {j} on GPU {g} (requeued)")
                p.send_signal(signal.SIGTERM)
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    p.kill()
                del mine[j]
                busy.pop(j, None)
    todo = [j for j in jobs if not done(j) and j not in busy and tries[j] < 3]
    if not todo and not busy:
        break
    load = {g: 0 for g in SLOTS}
    for g in busy.values():
        if g in load:
            load[g] += 1
    allow_early = now() < START_CUTOFF
    for g in ["5", "6", "7", "0", "1", "2", "3", "4"]:
        if g in EARLY and not allow_early:
            continue
        while load[g] < SLOTS[g] and todo:
            j = todo.pop(0)
            tries[j] += 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTHONUNBUFFERED="1",
                       PIPELINE_EARLY_GPU_UNTIL=KILL_AT.isoformat() if g in EARLY else "")
            lf = open(logd / f"{j[0]}_{j[1]}_seed{j[2]}.log", "a")
            p = subprocess.Popen([PY, str(HERE / "run_pipeline.py"), "--scenario", j[0], "--system", j[1],
                                  "--seed", str(j[2]), "--device", "cuda:0"], env=env, stdout=lf, stderr=lf)
            mine[j] = (p, g)
            load[g] += 1
            log(f"start {j} on GPU {g} (try {tries[j]})")
    time.sleep(15)
log("all done")
