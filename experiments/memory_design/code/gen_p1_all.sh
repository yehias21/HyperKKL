#!/bin/bash
# CPU-only generation of the long phase-1 episodes (PREREG 2.2), 6 at a time.
cd "$(dirname "$0")"
for s in quad_tank battery vdp bioreactor; do for n in C3-partial C3-mismatch30; do for sd in 0 1 2; do
  echo "$n $s $sd"
done; done; done | xargs -P 6 -L 1 bash -c 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 ../../../.venv/bin/python run_md.py --node $0 --system $1 --seed $2 --device cpu --p1only > ../logs/p1_$0_$1_s$2.log 2>&1; echo "$(date +%H:%M:%S) p1 $0 $1 $2 rc=$?"'
