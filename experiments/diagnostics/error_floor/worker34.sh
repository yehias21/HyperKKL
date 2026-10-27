#!/bin/bash
# worker.sh GPU DEADLINE_EPOCH(0=none) QUEUEFILE   -- claims lines of QUEUEFILE via flock
GPU=$1; DL=$2; Q=$3; D=$(dirname $0); P=/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/.venv/bin/python
while true; do
  if [ "$DL" != "0" ] && [ $(date +%s) -ge $DL ]; then echo "deadline"; exit 0; fi
  LINE=$(flock $Q.lock bash -c "L=\$(head -n1 $Q); sed -i '1d' $Q; echo \$L")
  [ -z "$LINE" ] && exit 0
  echo "[$(date +%T)] gpu$GPU $LINE"
  (cd $D && CUDA_VISIBLE_DEVICES=$GPU $P $LINE cuda:0 out34 > logs/$(echo $LINE | tr ' /' '__').log 2>&1) || echo "FAIL $LINE"
done
