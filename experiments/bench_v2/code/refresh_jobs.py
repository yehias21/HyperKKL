"""Keep logs/jobs_gpu.txt = the lines of logs/jobs_gpu_all.txt whose data cell is built (MANIFEST.json exists), in
the same (priority) order. Real-track lines are always kept. Runs until every cell is built."""
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
while True:
    lines, missing = [], 0
    for l in (HERE / "logs" / "jobs_gpu_all.txt").read_text().splitlines():
        if "::" not in l:
            lines.append(l); continue
        c = l.split("::", 1)[1].split()
        if c[0] == "v2run.py":
            if not (HERE / "data" / c[1] / c[2] / c[3] / "MANIFEST.json").exists():
                missing += 1; continue
        lines.append(l)
    tmp = HERE / "logs" / "jobs_gpu.txt.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(HERE / "logs" / "jobs_gpu.txt")
    if missing == 0:
        break
    time.sleep(60)
