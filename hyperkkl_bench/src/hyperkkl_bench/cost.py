"""Cost of an estimator: parameters, persistent memory, FLOPs per step, measured latency.

Latency is the wall-clock time of ONE ``step(u, y)`` call at batch 1 on CPU, median over
``n`` calls after ``warmup`` calls, single thread (torch.set_num_threads(1)), on a real episode.
It includes Python overhead by design: it is what a deployed online estimator pays per sample.
Report the CPU model next to it (``platform`` field). It is NOT used for deployment-budget verdicts:
those use ``mcu_us_est``, the FLOP-based estimate on the reference MCU of the research deployment
study (``MCU`` below), because a Python step costs milliseconds whatever the method.

FLOPs: the estimator's ``flops_per_step()`` hook if present; otherwise, for torch estimators,
torch.utils.flop_counter counts the matmul-type FLOPs of one step (elementwise ops are not
counted, so this is a lower bound); otherwise null.
"""

from __future__ import annotations

import platform
import time

import numpy as np

# MCU model of the research deployment study (experiments/casestudies/latency_realistic.json "mcu"):
# Cortex-M7-class, 480 MHz, 0.5 FLOP / cycle sustained, 40 cycles per transcendental, x1.5 overhead.
MCU = {"hz": 480e6, "flop_per_cycle": 0.5, "cycles_per_transcendental": 40, "overhead": 1.5}


def mcu_us(flops, transcendentals=0, mcu=MCU):
    """Estimated per-step latency (us) on the reference MCU from FLOPs (and transcendentals)."""
    cyc = flops / mcu["flop_per_cycle"] + transcendentals * mcu["cycles_per_transcendental"]
    return float(mcu["overhead"] * cyc / mcu["hz"] * 1e6)


def _torch_modules(est):
    try:
        from torch import nn
    except ImportError:
        return []
    return [v for v in vars(est).values() if isinstance(v, nn.Module)]


def n_params(est):
    if hasattr(est, "n_params"):
        v = est.n_params()
        return None if v is None else int(v)
    mods = _torch_modules(est)
    return int(sum(p.numel() for m in mods for p in m.parameters())) if mods else None


def measure(est, es, n: int = 300, warmup: int = 30, budget_us: float | None = None,
            budget_bytes: int | None = None) -> dict:
    out = {"n_params": n_params(est), "state_bytes": int(est.state_bytes()) if hasattr(est, "state_bytes") else None,
           "flops_per_step": None, "flops_source": None, "platform": platform.processor() or platform.machine()}
    if out["state_bytes"] is None and out["n_params"] is not None:
        out["state_bytes"] = 4 * out["n_params"]                 # float32 weights (lower bound)
    if hasattr(est, "flops_per_step"):
        out["flops_per_step"], out["flops_source"] = int(est.flops_per_step()), "hook"
    try:
        import torch
        nth = torch.get_num_threads()
        torch.set_num_threads(1)
    except ImportError:
        torch, nth = None, None
    try:
        ep = es[0]
        try:
            est.reset(np.zeros(es.x0.shape[1]), None)
        except NotImplementedError as e:
            out["latency_note"] = f"not measured: {e}"
            return out
        ctx = getattr(est, "context", None)
        x0 = np.asarray(ctx["prior_mean"], float) if ctx else np.zeros(es.x0.shape[1])
        est.reset(x0, None)
        K = min(len(ep.u), warmup + n)
        ts = []
        if out["flops_per_step"] is None and torch is not None and _torch_modules(est):
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                est.step(float(ep.u[0]), ep.y[0])
            out["flops_per_step"], out["flops_source"] = int(fc.get_total_flops()), "torch.flop_counter (matmul only)"
        for k in range(K):
            t0 = time.perf_counter()
            est.step(float(ep.u[k]), ep.y[k])
            ts.append(time.perf_counter() - t0)
        ts = np.asarray(ts[warmup:]) * 1e6
        out.update({"latency_us_median": float(np.median(ts)), "latency_us_p95": float(np.percentile(ts, 95)),
                    "latency_n": int(len(ts))})
    finally:
        if torch is not None:
            torch.set_num_threads(nth)
    if out["flops_per_step"] is not None:
        tr = int(est.transcendentals_per_step()) if hasattr(est, "transcendentals_per_step") else 0
        out["mcu_us_est"] = mcu_us(out["flops_per_step"], tr)
        out["mcu_model"] = dict(MCU, transcendentals_counted=hasattr(est, "transcendentals_per_step"))
    if budget_us is not None:
        # the budget verdict uses the MCU estimate (what the budget is about); the measured CPU latency
        # (Python / torch overhead on the evaluation host) is reported separately and never used for it
        out["within_latency_budget"] = (bool(out["mcu_us_est"] <= budget_us) if "mcu_us_est" in out else None)
        out["latency_budget_basis"] = "mcu_us_est (FLOPs-based)"
    if budget_bytes is not None and out["state_bytes"] is not None:
        out["within_memory_budget"] = bool(out["state_bytes"] <= budget_bytes)
    return out

