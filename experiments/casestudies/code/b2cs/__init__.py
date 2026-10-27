"""b2cs: copy of experiments/baselines2/code/b2 (GRU, Neural CDE, NODE / UDE + EKF/UKF, capacity-matched
NLOX) for the deployment case studies. Changes vs b2 (see code/README.md): set-up through cs_common (package
copy, local cache); masked (sparse-label) losses in train.py; job.py replaced by code/run_cs.py."""
import cs_common  # noqa: F401  (sys.path, cache redirect, filter defaults)
