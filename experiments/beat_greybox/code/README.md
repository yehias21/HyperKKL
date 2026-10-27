Patch against the live package at copy time (06:56 CET 2026-10-26, sha1 list in live_snapshot.sha1;
train.py and cache.py were unchanged in the live package since then; soap.py, beat.py, beat_b.py are new).
The scratch copy additionally took hyperkkl/__init__.py and systems_bounded.py from the live package
at 17:53 CET (bounded test plants; unchanged).
Tests: tests/test_beat.py (17 passed, CPU, 23 min), tests/test_beat_b.py (3 passed).
Runners: run_bg.py (Stage A / D arms), run_b.py (B1), run_b2.py (B2), run_real.py (real battery),
run_gbrefit.py (grey-box refit where the tree has none), dispatch_bg.py / dispatch_b.py (GPUs 5-7),
analyze_bg.py / analyze_test.py (mechanical verdicts).
