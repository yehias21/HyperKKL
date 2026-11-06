# Release checklist (maintainers)

Nothing is uploaded by this package or its tools. A release is a manual, author-approved step.

1. `pytest` green; `python reference/reproduce_reference.py --check` -> ALL MATCH; `hyperkkl-bench verify-data`.
2. Public release content: `src/`, `tests/`, `reference/`, `tools/`, `docs/`, `notebooks/`, `README.md`, `PROTOCOL.md`,
   `PITFALLS.md`, `CHANGELOG.md`, `LICENSE`, `CITATION.cff`, `data_extra/` (val_ood + label-free hidden files + MANIFEST).
3. **NEVER release `hidden_labels/`** (labels + SECRET). Keep it with the leaderboard host only. `.gitignore` excludes it.
   Note: `tools/build_extra_splits.py` is public; without SECRET it cannot regenerate the hidden episodes.
4. Data: the 5.2 GB export lives in the private HF dataset `Yahias21/HyperKKL-benchmark`. Before making it public, add
   `data_extra/` to it (same layout: `data/<plant>/<variant>/val_ood.parquet`, `hidden_*.parquet`), set the licence in
   its card, and remove `mismatch30_params` from public metadata? (author decision: it documents the real plant, but
   it is the answer key of the mismatch level; PITFALLS P9).
5. Bump `version.py` + `pyproject.toml`; add a CHANGELOG entry; tag.
6. Real-data tracks: `data_real/` (raw files fetched from the original hosts + locally prepared episodes)
   and `logs/hkkl_train/*/data.npz` (copies of real training windows) are NEVER released; both are
   gitignored. EMPS, Silverbox and NeuroBEM state no licence: do not rehost them without the providers' permission. NeuroBEM asks
   only for a citation; redistribution is not granted. Ask UZH RPG for written terms before a public release, and decide whether
   `docs/real/neurobem_episodes.png` (plots of their data) ships.
7. `HyperKKLTrain` imports the beat_greybox package copy in `archive/` (not for release): ship that code
   (or an equivalent HyperKKL training module) under the chosen licence before advertising it publicly.

## Version 2.0 additions
8. Public: `data_v2/` (30 GB: data/<family>/<plant>/<variant>/*.parquet without labels for hidden_*, metadata/<family>/*.json
   with SHA-256 per file). Upload it next to the v1 export when the authors make the data public; the licence is MIT
   (simulated data).
9. **NEVER release `hidden_labels/v2/`** (labels of hidden_op_id / hidden_op_ood / hidden_id / hidden_ic_ood and
   `SECRET_v2`). The generator of the hidden v2 episodes is experiments/bench_v2/code/build_data.py; without the secret it
   cannot regenerate them.
10. Excluded and never released: autonomous rossler at mismatch / realistic (experiments/bench_v2/data_excluded; DESIGN.md
    Addendum A of the v2 study).
