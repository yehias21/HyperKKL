"""baselines2: learned, recipe-independent baselines (GRU, Neural CDE, Neural ODE / UDE + EKF/UKF,
capacity-matched NLOX, re-fitted grey-box filters) on the C3 cells.

Isolation (PREREG.md section 0): the live package ``hyperkkl`` is imported unchanged. Its data cache
is redirected: a file that exists in the live cache is READ from there (seeds 0-2: exactly the episodes
of the stored runs); a missing one (seeds 3-4 training data) is built by the unchanged code into
experiments/baselines2/cache. Nothing is written to the live cache.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import env  # bench_v2: frozen package copy first on sys.path; no live-cache redirection here
LIVE = env.LIVE
HERE = env.HERE
