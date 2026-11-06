"""Version tags of the benchmark.

BENCH_VERSION   the package / protocol version (semantic versioning; a change of the scoring
                rule or of a frozen split bumps the MAJOR number).
DATA_VERSION    the frozen data release the scores refer to. Every report carries both tags,
                so a number can always be traced to the exact episodes and scorer.
"""

BENCH_VERSION = "2.0.0rc1"
DATA_VERSION = "hkb-data-1.0"          # core export 2026-10-27 + val_ood / hidden 2026-10-31 + deployment / scaling export 2026-11-01
DATA_V2_VERSION = "hkb-data-2.0"       # two families (autonomous / driven), operating-profile tiers; built 2026-11-03 (experiments/bench_v2)
SCORER_VERSION = "settled-nrmse-1"     # metrics.py: settle_frac 0.5, x_std per plant, clip 100
# real-data tracks (hyperkkl_bench.real): raw data fetched from the original hosts, preprocessed locally; each
# report carries its track's prep version (e.g. battery_pf-prep-1) and scorer (settled-soc-rmse-1,
# settled-velocity-nrmse-1, settled-quad-hidden-1)
REAL_VERSION = "hkb-real-1.2"          # 1.1: + neurobem; 1.2: + calce; earlier tracks unchanged

__all__ = ["BENCH_VERSION", "DATA_VERSION", "SCORER_VERSION", "REAL_VERSION"]
