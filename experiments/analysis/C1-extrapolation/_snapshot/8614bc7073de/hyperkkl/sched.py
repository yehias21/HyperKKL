"""GPU worker pool: drains experiments/queue/pending onto the GPUs.

    python -m hyperkkl worker --gpus 0-7 --per-gpu 4

Each job runs as its own process (``python -m hyperkkl.job``) pinned to one GPU via
CUDA_VISIBLE_DEVICES. Claiming is an atomic rename pending/ -> running/, so two workers
can never run the same job. When a node's last job finishes, the worker writes its report
and re-renders TREE.md. ``touch experiments/STOP`` stops claiming new jobs; running jobs
finish normally.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from . import tree as Tr
from .config import ROOT

PENDING, RUNNING, FAILED = Tr.QUEUE / "pending", Tr.QUEUE / "running", Tr.QUEUE / "failed"
PY = sys.executable
REPORT_ASYNC = True        # reports run in a detached process; tests switch this off


def parse_gpus(s: str) -> list:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _next_job():
    jobs = []
    for p in PENDING.glob("*.json"):
        try:
            j = json.loads(p.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        jobs.append((-int(j.get("priority", 0)), j.get("queued", ""), p.name, p))
    for _, _, name, p in sorted(jobs):
        dst = RUNNING / name
        try:
            os.rename(p, dst)                       # atomic claim
            return json.loads(dst.read_text()), dst
        except (FileNotFoundError, OSError):
            continue
    return None, None


def _alive(pid) -> bool:
    try:
        if int(pid) <= 0:
            return False
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:                         # exists, owned by someone else
        return True


def _finish(job, path, rc):
    nid = job["node"]
    try:
        if rc == 0:
            path.unlink(missing_ok=True)
        else:
            err = Tr.run_dir(nid, job["system"], job["seed"]) / "error.txt"
            if not err.exists():                    # killed (OOM, signal): no traceback written,
                err.parent.mkdir(parents=True, exist_ok=True)   # so progress() would not see it
                err.write_text(f"exit code {rc} without a traceback (killed?); see stdout.txt\n")
            FAILED.mkdir(parents=True, exist_ok=True)
            os.replace(path, FAILED / path.name)
        p = Tr.progress(nid)
        st = Tr.refresh_status(nid)
        if p["pending"] == 0 and p["running"] == 0 and p["done"] > 0:
            # in a detached process: a large node's report takes minutes and must not stall
            # job dispatch; a report bug cannot kill the pool either
            if REPORT_ASYNC:
                log = open(Tr.TREE / nid / "report.log", "w")
                subprocess.Popen([PY, "-m", "hyperkkl", "report", nid], cwd=str(ROOT),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            else:
                try:
                    Tr.report(nid)
                except Exception as e:
                    print(f"report {nid} failed: {e!r}", flush=True)
        Tr.render()
    except Exception as e:                          # nor a bookkeeping bug (bad node.yaml, ...)
        print(f"bookkeeping for {nid} failed: {e!r}", flush=True)
        return
    print(f"[{time.strftime('%H:%M:%S')}] {nid} {job['system']} seed{job['seed']} rc={rc} "
          f"-> node {st} {p['done']}/{p['total']}", flush=True)


def worker(gpus: list, per_gpu: int = 4, poll: float = 5.0):
    for d in (PENDING, RUNNING, FAILED):
        d.mkdir(parents=True, exist_ok=True)
    # jobs left in running/ by a dead worker go back to pending, unless their process is
    # still alive (another worker's job, or an orphan) or they finished after it died
    for p in RUNNING.glob("*.json"):
        try:
            j = json.loads(p.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            j = {}
        if _alive(j.get("pid")):
            print(f"worker: {p.name} is still running (pid {j['pid']}), left in running/", flush=True)
            continue
        res = Tr.run_dir(j["node"], j["system"], j["seed"]) / "results.json" if "node" in j else None
        if res is not None and res.exists() and res.stat().st_mtime > p.stat().st_mtime:
            _finish(j, p, 0)
        else:
            os.replace(p, PENDING / p.name)
    slots = [(g, i) for i in range(per_gpu) for g in gpus]      # interleaved across GPUs
    procs = {s: None for s in slots}
    stop = Tr.EXP / "STOP"
    print(f"worker: gpus {gpus} x {per_gpu} slots", flush=True)
    while True:
        for s, cur in procs.items():
            if cur is not None and cur[0].poll() is not None:
                proc, job, path, fh = cur
                fh.close()
                _finish(job, path, proc.returncode)
                procs[s] = None
        busy = sum(v is not None for v in procs.values())
        if stop.exists():
            if busy == 0:
                print("worker: STOP and idle, exiting", flush=True)
                return
        else:
            for s in slots:
                if procs[s] is not None:
                    continue
                job, path = _next_job()
                if job is None:
                    break
                rd = Tr.run_dir(job["node"], job["system"], job["seed"])
                rd.mkdir(parents=True, exist_ok=True)
                fh = open(rd / "stdout.txt", "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s[0]), OMP_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
                proc = subprocess.Popen([PY, "-m", "hyperkkl.job", "--node", job["node"],
                                         "--system", job["system"], "--seed", str(job["seed"]),
                                         "--device", "cuda:0"],
                                        cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
                procs[s] = (proc, job, path, fh)
                tmp = path.with_suffix(".tmp")      # the pid tells a restarted worker it is alive
                tmp.write_text(json.dumps({**job, "pid": proc.pid}))
                os.replace(tmp, path)
                try:
                    Tr.refresh_status(job["node"])
                except Exception as e:
                    print(f"status {job['node']} failed: {e!r}", flush=True)
                print(f"[{time.strftime('%H:%M:%S')}] start {job['node']} {job['system']} "
                      f"seed{job['seed']} on gpu{s[0]}", flush=True)
        time.sleep(poll)
