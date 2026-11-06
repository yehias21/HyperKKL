"""Maintainer tool: write src/hyperkkl_bench/resources/real_neurobem.json from the downloaded files (hashes, file list,
flight table, manoeuvre families, FIXED split lists). Run once; `prepare` adds the alignment and the identification.

    python tools/build_real_neurobem_resources.py data_real/neurobem/raw

Split rule (fixed BEFORE any estimator was run; splits are BY FLIGHT, never within a flight):
  * test_ood: every flight of the held-out manoeuvre types cpc, satellite, vertical_osc.
  * the other types: within each type, flights in time order, index i: i % 5 == 2 -> test_id, i % 5 == 4 -> val, else train.
Aggressive flight exists on both sides: lemniscate, linear_osc, circle3d and random_points in train reach the same speed
and acceleration range as the held-out types.
"""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BASE = "https://download.ifi.uzh.ch/rpg/NeuroBEM/"
FAMILY = [("wobbly", "wobbly_circle"), ("battery test", "battery_circle"), ("linear osc", "linear_osc"),
          ("vertical", "vertical_osc"), ("3d circle", "circle3d"), ("lemniscate", "lemniscate"), ("cpc", "cpc"),
          ("random", "random_points"), ("satellite", "satellite"), ("ellipse", "ellipse"), ("circle", "circle")]
OOD = ("cpc", "satellite", "vertical_osc")


def family(d):
    d = d.lower()
    return next(v for k, v in FAMILY if k in d)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


raw = Path(sys.argv[1])
flights = [(re.search(r'"(.*?)"', l).group(1), l.split("%")[1].strip()) for l in open(raw / "Flights.txt") if l.strip()]
counts, table = {}, {}
for fl, d in sorted(flights):
    f = family(d)
    i = counts.get(f, 0)
    counts[f] = i + 1
    split = "test_ood" if f in OOD else ("test_id" if i % 5 == 2 else "val" if i % 5 == 4 else "train")
    table[fl] = {"description": d, "family": f, "split": split}
stamp = "computed by us on 2026-11-02 (the host publishes no hash)"
files = []
for name, extra in (("Readme.md", {}), ("Flights.txt", {}), ("testset.txt", {}),
                    ("processed_data.zip", {"extract": True, "extract_to": "processed"}),
                    ("raw_data.zip", {"extract": True, "extract_to": "rawlogs"})):
    p = raw / name
    e = {"path": name, "url": BASE + name, "size": p.stat().st_size, "sha256": sha(p), "sha256_source": stamp, **extra}
    if name == "raw_data.zip":       # only the Betaflight logs; the Vicon rosbags are not needed
        e["members"] = sorted(n for n in zipfile.ZipFile(p).namelist() if n.endswith(".BFL"))
    if name == "processed_data.zip":
        e["members"] = sorted(n for n in zipfile.ZipFile(p).namelist() if n.endswith(".csv"))
    files.append(e)
out = {"dataset": "neurobem",
       "source": {
           "title": "NeuroBEM quadrotor flight dataset",
           "creator": "L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, D. Scaramuzza (Robotics and Perception Group, University of Zurich)",
           "repository": "University of Zurich, https://download.ifi.uzh.ch/rpg/NeuroBEM/",
           "doi": "10.15607/RSS.2021.XVII.042",
           "landing": "https://rpg.ifi.uzh.ch/NeuroBEM.html",
           "license": "no licence text on the source (project page, Readme, download host and code archive checked on "
                      "2026-11-02; the only stated term is 'If you use this work in your research, please cite the "
                      "following paper'). Use approved by the author, who states he checked the licence personally",
           "redistribution": "NOT GRANTED by any text on the source, so all rights must be assumed reserved: do not rehost "
                             "the raw data or the prepared episodes. `fetch` downloads from the original host. Check the "
                             "terms with the UZH RPG providers before a public release",
           "citation": "L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, D. Scaramuzza, NeuroBEM: Hybrid Aerodynamic Quadrotor "
                       "Model, Robotics: Science and Systems (RSS), 2021, doi:10.15607/RSS.2021.XVII.042, arXiv:2106.08015"},
       "files": files, "flights": table, "ood_families": list(OOD)}
p = HERE / "src" / "hyperkkl_bench" / "resources" / "real_neurobem.json"
if p.exists():
    old = json.loads(p.read_text())
    if "frozen" in old:
        out["frozen"] = old["frozen"]
p.write_text(json.dumps(out, indent=1))
import collections
print(collections.Counter((v["split"]) for v in table.values()))
print(sorted(collections.Counter((v["family"], v["split"]) for v in table.values()).items()))
