"""Pre-generate the slow-family training episodes on the CPU (same Job.gen, same seeds/locks)."""
import sys
from run_qt import Job, QS
node, seed = sys.argv[1], int(sys.argv[2])
j = Job(node, seed, "pregen", "cpu")
T0 = float(j.cfg["system"]["T"])
j.gen("slow_T1", QS.SLOW_FAMILIES, 300, T0, 22)
j.gen("slow_T3", QS.SLOW_FAMILIES, 600, 3 * T0, 23)
