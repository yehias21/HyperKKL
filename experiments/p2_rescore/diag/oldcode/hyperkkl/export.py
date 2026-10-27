"""Contribution C4: export the testbed as a self-contained folder (Hugging Face layout).

    python -m hyperkkl.export --out release/hyperkkl-testbed [--systems ...] [--node C1-where]

Writes, per system: the frozen evaluation episodes of every tier for both splits
(``<system>/<split>_<tier>.npz`` with x, u, y_clean, y, families), the system config and
measured scales, plus METRICS.md, the signal-family table and a dataset card. With
``--node``, trained observers of that node are copied as checkpoints.

Nothing is uploaded. Publishing is `huggingface-cli upload <repo> <out>` by the owner.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from . import data as Dm
from . import signals as S
from .config import load, system_names

CARD = """---
license: mit
task_categories: [time-series-forecasting]
tags: [state-estimation, observers, kkl, nonlinear-systems, distribution-shift]
---
# HyperKKL testbed: state estimation under input shift

{n} simulated plants driven by exogenous inputs, with frozen evaluation episodes for
three tiers. Every tier keeps the input inside the same amplitude box (actuator limits);
what shifts is its shape or its sequence: `id` (training families and ranges, unseen
seeds), `shape_ood` (waveform families never used in training, same box and frequency
band), `sequence_ood` (the pedal-like `drive` family with a different event sequence:
large jumps, short dwells, abrupt ramps). Two splits, `dev` and
`test`, with disjoint seeds. Plants marked `role: test` are meant to be evaluated once.

Arrays per file: `x` (E, N+1, nx) true state, `u` (E, N+1, 1) input, `y_clean`, `y`
(E, N+1, ny) noise-free and measured outputs, `families` (E,). `scales.json` holds the
reference stds that define NRMSE (see METRICS.md).
"""


def export(out: Path, systems: list, node: str | None = None):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in systems:
        d = out / s
        d.mkdir(exist_ok=True)
        for split in ("dev", "test"):
            cfg = load(s, [{"eval.split": split}])
            sc = Dm.system_scales(cfg)
            for tier in S.TIERS:
                es = Dm.eval_data(cfg, tier)
                np.savez_compressed(d / f"{split}_{tier}.npz", x=es.x.numpy(), u=es.u.numpy(),
                                    y_clean=es.y_clean.numpy(), y=es.y.numpy(),
                                    families=np.array(es.families), dt=es.dt, burn=es.burn)
        (d / "system.yaml").write_text(yaml.safe_dump(cfg["system"], sort_keys=False))
        (d / "scales.json").write_text(json.dumps({k: v for k, v in sc.items() if k != "design_check"}, indent=1))
        rows.append((s, cfg["system"]["role"], len(cfg["system"]["ic_box"]), len(cfg["system"]["sensors"])))
        if node:
            src = Dm.ROOT / "experiments" / "tree" / node / "runs" / s
            if src.exists():
                shutil.copytree(src, d / "baseline_runs", dirs_exist_ok=True)
    (out / "README.md").write_text(CARD.format(n=len(systems)) + "\n| system | role | nx | ny |\n|---|---|---|---|\n"
                                   + "\n".join(f"| {a} | {b} | {c} | {e} |" for a, b, c, e in rows) + "\n")
    from . import metrics
    (out / "METRICS.md").write_text("# Metrics\n\n" + (metrics.__doc__ or ""))
    (out / "SIGNALS.json").write_text(json.dumps(
        {"ranges": S.RANGES, "train": S.TRAIN_FAMILIES, "shape_ood": S.SHAPE_OOD_FAMILIES,
         "sequence_ood": S.SEQUENCE_OOD_FAMILIES, "time_unit": "t_ref (system.t_ref seconds)",
         "tiers": S.regime_table()}, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release/hyperkkl-testbed")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--node", default=None)
    a = ap.parse_args(argv)
    print(export(Path(a.out), a.systems or system_names(), a.node))


if __name__ == "__main__":
    main()
