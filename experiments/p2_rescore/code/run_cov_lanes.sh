#!/bin/bash
# 10 lanes, 2 per GPU (GPUs 0-4); each lane runs its jobs serially.
P=/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/p2_rescore; cd $P/code
PY=/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python
JOBS=()
for js in "new fhn" "old two_link_arm" "new cartpole" "old cartpole" "old highway_traffic" "camp quad_tank" "camp battery"; do
  for s in 0 1 2; do JOBS+=("$js $s"); done; done
for lane in $(seq 0 9); do
  ( g=$((lane % 5)); for ((i=lane; i<${#JOBS[@]}; i+=10)); do set -- ${JOBS[$i]}
      CUDA_VISIBLE_DEVICES=$g $PY coverage.py --which $1 --system $2 --seed $3 > $P/logs/cov4_$1_$2_s$3.log 2>&1; done ) &
done
wait
