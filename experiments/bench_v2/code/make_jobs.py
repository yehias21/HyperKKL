"""Write the job lists in PREREG section-6 priority order.

    python make_jobs.py gpu   > ../logs/jobs_gpu.txt      (v2run jobs: every learned method needs a GPU)
    python make_jobs.py cpu   > ../logs/jobs_cpu.txt      (real-track CPU methods)

Within a priority level, phase-1 jobs come first (later jobs of the same cell/seed wait for the maps), then the
expensive peers, so the long jobs start early and the short ones fill in.
"""
import sys

import v2cfg

SEEDS = [0, 1, 2]
B_CELLS = [("driven", p, "realistic") for p in v2cfg.FAMILY_B]
A_CELLS = [("autonomous", p, "white") for p in v2cfg.FAMILY_A]
REAL = ["battery_pf", "battery_lg", "emps", "silverbox"]
NCDE_CELLS = ["vdp", "battery", "cartpole_b", "lorenz96"]
INJ_CELLS = ["vdp", "battery", "cartpole_b"]


def run(fam, p, v, m, s, lr=None):
    tag = m + (f"_lr{lr:g}" if lr is not None else "")
    lrs = f" --lr {lr:g}" if lr is not None else ""
    return f"runs/{fam}/{p}/{v}/{tag}/seed{s}.json :: v2run.py {fam} {p} {v} {m} --seed {s}{lrs}"


def level(cells, family):
    out = []
    for s in SEEDS:
        for c in cells:
            out.append(run(*c, "p1", s))
    for s in SEEDS:
        for c in cells:
            for m in ("nlox_m", "dkkl_m"):
                for lr in (1e-2, 1e-3):
                    out.append(run(*c, m, s, lr))
    for c in cells:
        out.append(run(*c, "filters", 0))
    for s in SEEDS:
        for c in cells:
            for lr in (1e-2, 1e-3):
                out.append(run(*c, "hk", s, lr))
            dyn = ("ude1", "ude32", "node32") if family == "driven" else ("ude1", "ude32")
            for m in dyn + ("gru",):
                out.append(run(*c, m, s))
            if family == "driven":
                for m in ("latent", "concat", "film"):
                    for lr in (2e-3, 1e-3):
                        out.append(run(*c, m, s, lr))
    return out


def gpu():
    out = ["# priority 1: family B core (realistic)"] + level(B_CELLS, "driven")
    out += ["# priority 2: real tracks, GPU part (HyperKKL A3)"]
    for d in REAL:
        for s in SEEDS:
            out.append(f"runs/real/{d}/hyperkkl_a3/seed{s}.json :: real_run.py {d} hyperkkl_a3 --seed {s}")
    out += ["# priority 3: family A (partial)"] + level(A_CELLS, "autonomous")
    out += ["# priority 4: ncde subset"]
    for s in SEEDS:
        for p in NCDE_CELLS:
            out.append(run("driven", p, "realistic", "ncde", s))
    out += ["# priority 5: dkkl_inj_m subset"]
    for s in SEEDS:
        for p in INJ_CELLS:
            for lr in (1e-2, 1e-3):
                out.append(run("driven", p, "realistic", "dkkl_inj_m", s, lr))
    return out


def cpu():
    out = []
    for d in REAL:
        for m in ("ekf_gated", "ukf_gated", "ekf_gated_tuned", "ukf_gated_tuned"):
            out.append(f"runs/real/{d}/{m}/seed0.json :: real_run.py {d} {m} --seed 0")
    for s in SEEDS:
        for d in REAL:
            for m in ("ude1_ekf_gated", "ude1_ukf_gated", "ude32_ekf_gated", "node32_ekf_gated", "gru"):
                out.append(f"runs/real/{d}/{m}/seed{s}.json :: real_run.py {d} {m} --seed {s}")
    return out


if __name__ == "__main__":
    print("\n".join(gpu() if sys.argv[1] == "gpu" else cpu()))
