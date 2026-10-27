"""Launcher: .venv/bin/python -I experiments/real/run.py <stage> [...]  (see hyperkkl/realdata_run.py).
-I keeps the current directory and user site off sys.path; the repo root is added explicitly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperkkl.realdata_run import main  # noqa: E402
main()
