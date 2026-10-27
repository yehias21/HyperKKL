#!/bin/bash
# Re-queue jobs that failed ONLY with the transient cuFFT plan error (CUFFT_INTERNAL_ERROR,
# seen when many jobs start on one GPU at once). Max 3 retries per job; other failures stay.
cd "$(dirname "$0")"
while true; do
  for f in queue/failed/*.json; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .json); node=${n%%__*}; rest=${n#*__}; sys=${rest%%__*}; seed=${rest##*__}
    d="tree/$node/runs/$sys/seed$seed"; err="$d/error.txt"
    cat "$err" "$d/stdout.txt" 2>/dev/null | grep -q "CUFFT_INTERNAL_ERROR" || continue
    k=$(grep -c "^$n\$" retry_cufft.count 2>/dev/null); k=${k:-0}
    [ "$k" -ge 3 ] && continue
    echo "$n" >> retry_cufft.count
    mv "$f" "queue/pending/$n.json" && rm -f "$err" && echo "$(date +%H:%M) requeued $n (attempt $((k+1)))" >> retry_cufft.log
  done
  sleep 600
done
