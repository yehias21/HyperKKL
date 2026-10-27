#!/usr/bin/env bash
# GPU guard: if anyone outside the HyperKKL work shows up on GPUs 0-4, free GPUs 0-4 within minutes.
# - polls nvidia-smi every 5 s; a process is "ours" if owned by demo AND its cmdline or cwd is under TMLR/hyperkkl
# - a foreign process seen in 2 consecutive polls triggers: write the block flag (agents must not schedule
#   on 0-4 while it exists), SIGTERM our processes on 0-4, SIGKILL survivors after 60 s, then exit
#   (the exit notifies the coordinator session). Remove the flag to clear 0-4 again.
ROOT=/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl
FLAG=$ROOT/experiments/GPU_0_4_BLOCKED
LOG=$ROOT/experiments/gpu_guard.log
GUARDED="0 1 2 3 4"
declare -A BUS2IDX
while IFS=', ' read -r idx bus; do BUS2IDX[$bus]=$idx; done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)

is_ours() {  # pid
  local p=$1 u c w
  u=$(stat -c %U /proc/$p 2>/dev/null) || return 0        # vanished -> ignore
  [ "$u" = demo ] || return 1
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  [ -z "$c" ] && return 0                                  # exiting process, empty cmdline
  w=$(readlink /proc/$p/cwd 2>/dev/null)
  [[ "$c" == *TMLR/hyperkkl* || "$w" == *TMLR/hyperkkl* ]]
}
guarded_pids() {  # mode: ours|foreign
  nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader 2>/dev/null |
  while IFS=', ' read -r bus pid; do
    i=${BUS2IDX[$bus]}; [[ " $GUARDED " == *" $i "* ]] || continue
    if is_ours "$pid"; then [ "$1" = ours ] && echo "$pid"; else [ "$1" = foreign ] && echo "$i:$pid"; fi
  done
}
echo "$(date '+%F %T %Z') guard started (pid $$)" >> "$LOG"
prev=""
while true; do
  [ -e "$FLAG" ] && { echo "$(date '+%F %T') flag already present, exiting" >> "$LOG"; exit 0; }
  cur=$(guarded_pids foreign | sort | tr '\n' ' ')
  if [ -n "$cur" ] && [ -n "$prev" ]; then
    for e in $cur; do pid=${e#*:}
      echo "$(date '+%F %T') FOREIGN on GPU ${e%%:*}: pid $pid user $(stat -c %U /proc/$pid 2>/dev/null) cmd $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-120)" >> "$LOG"
    done
    date '+%F %T %Z' > "$FLAG"; echo "foreign: $cur" >> "$FLAG"
    ours=$(guarded_pids ours | sort -u)
    echo "$(date '+%F %T') freeing GPUs 0-4: SIGTERM $ours" >> "$LOG"
    [ -n "$ours" ] && kill -TERM $ours 2>/dev/null
    sleep 60
    ours=$(guarded_pids ours | sort -u)
    [ -n "$ours" ] && { echo "$(date '+%F %T') SIGKILL $ours" >> "$LOG"; kill -KILL $ours 2>/dev/null; }
    echo "$(date '+%F %T') GPUs 0-4 freed; guard exits" >> "$LOG"
    exit 3
  fi
  prev=$cur
  sleep 5
done
