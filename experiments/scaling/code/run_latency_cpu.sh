#!/usr/bin/env bash
# CPU latency, one pinned core (taskset), torch threads = 1, every n_x in turn.
cd /mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl
CORE=${CORE:-93}
for nx in 4 8 16 32 64 128; do
  CUDA_VISIBLE_DEVICES="" taskset -c $CORE .venv/bin/python -u experiments/scaling/code/latency.py --nx $nx --device cpu \
    > experiments/scaling/logs/latency_cpu_$nx.log 2>&1
done
