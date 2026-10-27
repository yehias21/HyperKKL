"""Per-step inference cost vs n_x (PREREG section 6).

    taskset -c CORE .venv/bin/python experiments/scaling/code/latency.py --nx 16 --device cpu
    CUDA_VISIBLE_DEVICES=g .venv/bin/python experiments/scaling/code/latency.py --nx 16 --device cuda:0

Observers are built by the job code itself with smoke-length training (the architecture and the
sizes are the full-run ones; the cost of a step does not depend on the trained weight values).
Cost of one online step x_hat_{k+1} = advance(y_k, u_k, y_{k+1}, u_{k+1}); estimate() at batch 1:
median of 7 repeats of 200 steps (after 20 warm-up steps), torch threads = 1. FLOPs: one step under
torch.utils.flop_counter.FlopCounterMode (counts matmul / bmm / addmm / conv / sdpa only; linalg ops
such as matrix_exp, inv and cholesky are NOT counted, so filter FLOPs are lower bounds).
Writes latency/nx{nx}_{device}.json.
"""
import argparse
import json
import statistics
import time

import common as C  # noqa: F401
import torch
from torch.utils.flop_counter import FlopCounterMode

from job import Job


@torch.no_grad()
def step_cost(ob, ny, nx, device, n=200, reps=7):
    y = torch.zeros(1, ny, device=device); u = torch.zeros(1, 1, device=device)
    ob.reset(1, torch.zeros(1, nx, device=device), u)
    for _ in range(20):
        ob.advance(y, u, y, u); ob.estimate()
    ts, cs = [], []
    for _ in range(reps):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0, c0 = time.perf_counter(), time.process_time()
        for _ in range(n):
            ob.advance(y, u, y, u); ob.estimate()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(1e6 * (time.perf_counter() - t0) / n)
        cs.append(1e6 * (time.process_time() - c0) / n)
    fc = FlopCounterMode(display=False)
    with fc:
        ob.advance(y, u, y, u); ob.estimate()
    # on the CPU every core of this shared host is busy: the process CPU time (us_cpu) excludes the time
    # the pinned core spent on other processes and is the primary CPU latency; wall time is kept too
    prim = statistics.median(cs) if device == "cpu" else statistics.median(ts)
    return {"us_median": prim, "us_wall_median": statistics.median(ts), "us_cpu_median": statistics.median(cs),
            "us_all": ts, "flops": int(fc.get_total_flops())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(1)
    train_dev = a.device                       # observers are built on the device they are timed on
    out = {"nx": a.nx, "device": a.device, "threads": torch.get_num_threads(), "methods": {}}
    for unit in ("kkl:partial", "gb:partial", "mf:D0"):
        j = Job(a.nx, 0, unit, train_dev, data_nx=4, smoke=True)   # small data: cost does not depend on it
        j.out = C.SC / "latency" / "tmp"; j.out.mkdir(parents=True, exist_ok=True)
        j.setup()
        obs, meta = {"kkl": j.run_kkl, "gb": j.run_gb, "mf": j.run_mf}[j.kind]()
        if a.nx >= 64 and "node_ekf" not in obs:            # timing only (NODE is not trained at 64)
            o, _ = j.run_dyn("node", hybrid=False); obs.update(o)
        for name, ob in obs.items():
            try:
                r = step_cost(ob, len(j.model.sensors), a.nx, a.device)
            except Exception as e:                      # report, never silently drop
                r = {"error": repr(e)}
            out["methods"][name] = {**r, "params": meta.get(name, {}).get("params_trainable")}
            print(a.nx, a.device, name, r.get("us_median"), r.get("flops"), flush=True)
    (C.SC / "latency").mkdir(exist_ok=True)
    (C.SC / "latency" / f"nx{a.nx}_{a.device.split(':')[0]}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
