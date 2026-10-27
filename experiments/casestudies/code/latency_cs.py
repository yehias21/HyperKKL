"""Deployment cost of every observer: per-step latency at batch 1 on ONE CPU core (torch, float32, 1 thread),
FLOPs per step, parameter / memory footprint, and an estimate for the target hardware. CPU only.

    CUDA_VISIBLE_DEVICES="" taskset -c <core> python latency_cs.py [A B] [--level realistic] [--seed 0]

Observers: the deployed (fresh) observers each run_cs.py job pickled for seed 0 (models/<case>/<level>/seed0/
obs_<family>.pt). Timing: a 300-step random (y, u) sequence, 5 repeats after 30 warm-up steps; we report the
MIN over repeats (closest to the uncontended cost: the host was shared, load ~150-180 on 96 cores) and the
median. FLOPs: torch.utils.flop_counter (matmul-type ops) on one advance+estimate, plus an analytic count of
what it does not see (filters: Jacobian, matrix exponential, inversions, the physics model's elementwise ops;
learned observers: activation functions). MCU estimate (Case A, STM32H7 Cortex-M7 @ 480 MHz, float32):
t = FLOPs / (0.5 FLOP/cycle x 480 MHz) + 40 cycles per transcendental (tanh/exp/sin/cos/sigmoid) evaluation,
x 1.5 control-flow overhead; stated to +-2x, NOT measured on hardware.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from hyperkkl.observers import _Filter, UKF, KKLObserver  # noqa: E402

torch.set_num_threads(1)
MCU_HZ, MCU_FLOP_PER_CYCLE, MCU_TRANS_CYCLES, MCU_OVERHEAD = 480e6, 0.5, 40, 1.5
BUDGET_US = {"A": 150.0, "B": 100_000.0}
FLASH_BYTES = {"A": 512 * 1024, "B": 50 * 1024 * 1024}
# elementwise cost of one evaluation of the practitioner's physics f (counted by hand from systems_casestudy):
# (flops, transcendentals)
PHYS = {"quad_planar": (70, 6), "cstr_exo": (35, 4)}


def tensors_of(obj, seen=None, depth=0):
    """Every tensor reachable from obj (modules: params + buffers; attributes; dicts / lists)."""
    seen = set() if seen is None else seen
    out = []
    if id(obj) in seen or depth > 6:
        return out
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, torch.nn.Module):
        return list(obj.parameters()) + list(obj.buffers())
    if isinstance(obj, dict):
        for v in obj.values():
            out += tensors_of(v, seen, depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out += tensors_of(v, seen, depth + 1)
        return out
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k in ("jac",):
                continue
            out += tensors_of(v, seen, depth + 1)
    return out


def footprint(ob):
    ts = {id(t): t for t in tensors_of(ob)}
    n = sum(t.numel() for t in ts.values())
    return int(n), int(4 * n)


def mlp_flops(mod, x_dim):
    """matmul FLOPs + activations of one forward of an nn.Module on a (1, x_dim) input (counted)."""
    fc = FlopCounterMode(display=False)
    with fc:
        mod(torch.zeros(1, x_dim))
    return int(fc.get_total_flops())


def count_transcendental(ob):
    """Activation evaluations per step of a learned observer: hidden units of tanh / silu / sigmoid / GRU gates."""
    n = 0
    for t in tensors_of(ob):
        pass
    mods = [m for m in tensors_modules(ob)]
    for m in mods:
        for sub in m.modules():
            if isinstance(sub, torch.nn.Linear):
                n += sub.out_features          # one activation per hidden unit (upper bound: also output layer)
            elif isinstance(sub, torch.nn.GRU):
                n += 3 * sub.hidden_size * sub.num_layers + sub.hidden_size * sub.num_layers
    return n


def tensors_modules(obj, seen=None, depth=0):
    seen = set() if seen is None else seen
    out = []
    if id(obj) in seen or depth > 6:
        return out
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        return [obj]
    if isinstance(obj, dict):
        for v in obj.values():
            out += tensors_modules(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += tensors_modules(v, seen, depth + 1)
    elif hasattr(obj, "__dict__") and not isinstance(obj, torch.Tensor):
        for k, v in vars(obj).items():
            out += tensors_modules(v, seen, depth + 1)
    return out


def filter_flops(ob, nx, ny, plant_name):
    """Analytic FLOPs per step of the EKF / UKF as implemented (observers.EKF / UKF)."""
    pf, pt = PHYS[plant_name]
    mlp = 0
    m = ob.m
    if hasattr(m, "net") or hasattr(m, "mlp"):            # grey-box / UDE / NODE residual MLP
        net = getattr(m, "net", None) or getattr(m, "mlp", None)
        try:
            mlp = mlp_flops(net, nx + 1) if net is not None else 0
        except Exception:
            mlp = sum(2 * p.numel() for p in net.parameters())
    f_eval = pf + mlp
    n, k = nx, ny
    upd = 2 * (2 * n * n * k + 2 * n * k * k) + k ** 3 + 2 * (2 * n ** 3)          # S, K, Joseph P update
    if isinstance(ob, UKF):
        sp = 2 * n + 1
        return int(sp * 4 * f_eval + n ** 3 / 3 + 2 * sp * n * n + upd), int(sp * 4 * pt)
    jac = n * f_eval * 2                                     # reverse-mode / forward-mode Jacobian ~ n f-evals x2
    expm = 10 * 2 * n ** 3                                   # scaling and squaring + Pade
    prop = 2 * 2 * n ** 3
    return int(jac + 4 * f_eval + expm + prop + upd), int((n + 4) * pt)


def time_ob(ob, nx, ny, reps=5, n=300, warm=30):
    g = torch.Generator().manual_seed(0)
    Y = torch.randn(n + warm + 1, 1, ny, generator=g) * 0.1
    U = torch.rand(n + warm + 1, 1, 1, generator=g) - 0.5
    with torch.no_grad():
        ob.reset(1, torch.zeros(1, nx), U[0])
        for k in range(warm):
            ob.advance(Y[k], U[k], Y[k + 1], U[k + 1]); ob.estimate()
        out = []
        for r in range(reps):
            t0 = time.perf_counter()
            for k in range(warm, warm + n):
                ob.advance(Y[k], U[k], Y[k + 1], U[k + 1]); ob.estimate()
            out.append(1e6 * (time.perf_counter() - t0) / n)
    return min(out), statistics.median(out)


def learned_flops(ob, nx, ny):
    fc = FlopCounterMode(display=False)
    y, u = torch.zeros(1, ny), torch.zeros(1, 1)
    with torch.no_grad():
        ob.reset(1, torch.zeros(1, nx), u)
        with fc:
            ob.advance(y, u, y, u); ob.estimate()
    return int(fc.get_total_flops())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["A", "B"])
    ap.add_argument("--level", default="realistic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import os
    core = sorted(os.sched_getaffinity(0))
    res = {"host_load": open("/proc/loadavg").read().split()[:3], "cores": core, "torch": torch.__version__,
           "threads": torch.get_num_threads(), "mcu": {"hz": MCU_HZ, "flop_per_cycle": MCU_FLOP_PER_CYCLE,
                                                       "cycles_per_transcendental": MCU_TRANS_CYCLES,
                                                       "overhead": MCU_OVERHEAD}, "cases": {}}
    import __main__
    import run_cs
    __main__.OmegaInj = run_cs.OmegaInj            # pickled from run_cs.py run as a script
    from hyperkkl.evaluate import build_observers
    from hyperkkl import data as Dm
    for case in a.cases:
        d = C.MODELS / case / a.level / f"seed{a.seed}"
        system = C.CASES[case]["system"]
        rows = {}
        groups = []
        # the practitioner's EKF / UKF: rebuilt from the config (no trained part); the oracle-tuned EKF and the
        # grey-box filters have the cost of the EKF / UKF and UDE filters respectively (same model + same
        # 64-wide residual MLP, only Q / R differ) and are reported as such in RESULTS.md
        cfg = C.case_cfg(case, a.level, a.seed, "cpu")
        cfg["baselines"] = ["ekf", "ukf"]
        des = Dm.make_design(cfg["system"])
        groups.append(("filters(rebuilt)", build_observers(cfg, des, Dm.system_scales(cfg, "cpu"), None, {}, None, "cpu")))
        for p in sorted(d.glob("obs_*.pt")):
            if p.name == "obs_filters.pt":
                continue
            try:
                groups.append((p.name, torch.load(p, weights_only=False, map_location="cpu")))
            except Exception as e:
                print(f"{case} cannot load {p.name}: {e!r}", flush=True)
        for gname, obs in groups:
            for name, ob in obs.items():
                if name == "climatology":
                    continue
                nx = len(C.CASES[case]["sigma"]) and None
                cfg = C.base_cfg(case, a.level)
                nx = len(cfg["system"]["ic_box"]); ny = len(cfg["system"]["sensors"])
                try:
                    tmin, tmed = time_ob(ob, nx, ny)
                except Exception as e:
                    rows[name] = {"error": repr(e)}
                    continue
                npar, nbytes = footprint(ob)
                if isinstance(ob, _Filter):
                    fl, trans = filter_flops(ob, nx, ny, system)
                    kind = "filter"
                else:
                    try:
                        fl = learned_flops(ob, nx, ny)
                    except Exception:
                        fl = 2 * npar
                    trans = count_transcendental(ob)
                    kind = "learned"
                mcu_us = 1e6 * MCU_OVERHEAD * (fl / MCU_FLOP_PER_CYCLE + trans * MCU_TRANS_CYCLES) / MCU_HZ
                rows[name] = {"kind": kind, "us_cpu_min": tmin, "us_cpu_median": tmed, "flops": fl,
                              "transcendentals": trans, "n_tensor_elems": npar, "bytes_f32": nbytes,
                              "mcu_us_est": mcu_us if case == "A" else None,
                              "meets_budget_cpu": tmin <= BUDGET_US[case],
                              "meets_budget_mcu_est": (mcu_us <= BUDGET_US[case] and nbytes <= FLASH_BYTES[case])
                              if case == "A" else (tmin <= BUDGET_US[case] and nbytes <= FLASH_BYTES[case])}
                r = rows[name]
                print(f"{case} {name:22s} cpu {tmin:8.1f} us (med {tmed:8.1f})  flops {fl:9d}  trans {trans:6d}  "
                      f"elems {npar:8d}  mcu_est {r['mcu_us_est'] if r['mcu_us_est'] is None else round(r['mcu_us_est'], 1)} us",
                      flush=True)
        res["cases"][case] = rows
    out = C.HERE / f"latency_{a.level}.json"
    out.write_text(json.dumps(res, indent=1, default=C.jdefault))


if __name__ == "__main__":
    main()
