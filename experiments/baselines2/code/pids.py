"""Print pids of python processes whose argv contains all given tokens (never this process)."""
import os, sys
from pathlib import Path
me = os.getpid()
for p in Path("/proc").iterdir():
    if not p.name.isdigit() or int(p.name) == me:
        continue
    try:
        a = [x.decode() for x in (p / "cmdline").read_bytes().split(b"\0") if x]
    except OSError:
        continue
    try:
        cwd = os.readlink(p / "cwd")
    except OSError:
        continue
    if cwd != os.path.dirname(os.path.abspath(__file__)):      # only OUR processes (cwd = this dir)
        continue
    if a and "python" in a[0] and all(any(t == x or x.endswith("/" + t) for x in a) for t in sys.argv[1:]):
        print(p.name)
