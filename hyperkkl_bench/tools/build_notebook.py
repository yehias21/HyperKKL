"""Write notebooks/quickstart.ipynb (plain nbformat-4 JSON; no jupyter needed to build it).
tests/test_notebook.py executes its code cells in order as a script."""
import json
from pathlib import Path

CELLS = [
    ("md", """# hyperkkl-bench quick start

Load a plant, train a small model-free GRU observer on CPU, evaluate it next to the reference gated
EKF, and read the report. This takes about 5 minutes on a laptop CPU.

Setup: `pip install -e .` in the `hyperkkl_bench` folder. Point `HYPERKKL_BENCH_DATA` at a copy of
the dataset, i.e. the folder with `data/` and `metadata/`. Inside the research repository it is
found automatically."""),
    ("code", """import numpy as np
import hyperkkl_bench as hkb
print(hkb.BENCH_VERSION, hkb.DATA_VERSION)
print(hkb.list_plants())"""),
    ("md", """## 1. A plant and its data

Every episode has the applied input `u`, held constant over each sample interval. It also has the
noisy measurements `y` and the true state `x`. Only `u` and `y` may be used at estimation time."""),
    ("code", """PLANT, LEVEL = "vdp", "realistic"      # Van der Pol; partial model, 30 % mismatch, colored Student-t noise
card = hkb.plant_card(PLANT)
print("measured:", card["sensor_names"], "| unmeasured:", [card["state_names"][j] for j in card["unmeasured"]])
print("partial model misses:", card["unmodelled_term"])
train = hkb.load(PLANT, LEVEL, "train", n=200, seed=0)        # a nested subset (data-efficiency hook)
val = hkb.load(PLANT, LEVEL, "val")
print(len(train), "train episodes; u", train.u.shape, "y", train.y.shape, "x", train.x.shape, "dt", round(train.dt, 4))"""),
    ("code", """try:
    import matplotlib.pyplot as plt
    ep = hkb.load(PLANT, LEVEL, "test_sequence_ood")[0]
    t = np.arange(len(ep.u)) * ep.dt
    fig, ax = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(t, ep.u); ax[0].set_ylabel("u (stop-and-go)")
    ax[1].plot(t, ep.y[:, 0], lw=0.5); ax[1].plot(t, ep.x[:, 0]); ax[1].set_ylabel("y, x1")
    ax[2].plot(t, ep.x[:, 1]); ax[2].set_ylabel("x2 (unmeasured)"); ax[2].set_xlabel("t (s)")
    plt.tight_layout()
except ImportError:
    print("matplotlib not installed; skipping the plot")"""),
    ("md", """## 2. Train a tiny baseline

Any object with `fit(train, val, context)`, `reset(x0_prior, P0_prior)` and `step(u, y)` is an
estimator. `evaluate` fits it on `train` and `val`, and nothing else; see PROTOCOL.md."""),
    ("code", """gru = hkb.baselines.GRUObserver(hidden=32, steps=400, window=128, batch=16, seed=0)
rep_gru = hkb.evaluate(gru, PLANT, LEVEL, train_n=200, n_episodes=16,
                       conditions=("clean", "outliers"))       # n_episodes=16: smoke run, not a reportable number
print(rep_gru.table())"""),
    ("md", """## 3. The reference: the gated EKF on the modeller's (partial, nominal) model

It needs no training. It uses only `context["model"]`, the model a modeller of this knowledge level
has."""),
    ("code", """ekf = hkb.baselines.EKF(gate=True)
rep_ekf = hkb.evaluate(ekf, PLANT, LEVEL, n_episodes=16, conditions=("clean", "outliers"), cost=True)
print(rep_ekf.table())"""),
    ("md", """On `realistic` the gated EKF does badly (NRMSE around 0.6). Its model is the *partial*
Van der Pol model: it misses the nonlinear damping, and its parameters are off by 30 %. The GRU
learned from data and does not need the model. This is the gap the benchmark measures. The
stronger, fair comparator is a grey-box filter that learns the missing term:
`hkb.baselines.UDEFilter()`. It takes longer to train; see PITFALLS P1."""),
    ("code", """for name, r in (("gru", rep_gru), ("ekf_gated", rep_ekf)):
    print(f"{name:10s}", {s: round(r.value(s), 4) for s in r.results}, "| latency us", round(r.cost["latency_us_median"], 1))
rep_gru.save("quickstart_gru_report.json")      # the full JSON: per-episode metrics, CIs, cost, versions"""),
    ("md", """## 4. Next steps

- Full numbers: drop `n_episodes`, use all 600 training episodes, and run 3 to 5 training seeds
  (PROTOCOL.md section 5).
- `hkb.tracks.run_track(make_estimator, "core")` runs all 12 plants x 4 knowledge levels.
- Select models for input-OOD robustness on `hkb.load(p, k, "val_ood")`, never on the test tiers.
- Leaderboard: `hkb.hidden.predict(est, PLANT, LEVEL, "submission/")` writes the hidden-test
  estimates. The labels are not public."""),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        c = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i}", "metadata": {},
             "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    out = Path(__file__).resolve().parents[1] / "notebooks" / "quickstart.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
