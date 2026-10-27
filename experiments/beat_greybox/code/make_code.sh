#!/bin/bash
# Refresh experiments/beat_greybox/code from the scratch copy.
L=/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl; S=/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/beat_greybox; C=$L/experiments/beat_greybox/code
mkdir -p $C/tests
( cd $S/pkg; for f in hyperkkl/*.py; do diff -u --label a/$f --label b/$f $L/$f $f 2>/dev/null || [ -e $L/$f ] || diff -u --label /dev/null --label b/$f /dev/null $f; done ) > $C/beat_greybox.patch
cp $S/pkg/tests/test_beat.py $S/pkg/tests/test_beat_b.py $C/tests/
cp $S/run_bg.py $S/run_b.py $S/dispatch_bg.py $S/analyze_bg.py $S/make_code.sh $C/
ls -la $C
