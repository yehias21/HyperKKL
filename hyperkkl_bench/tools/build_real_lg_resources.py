"""Maintainer tool: the file list (Mendeley's own SHA-256) of the battery_lg track from the Mendeley public
API (read-only GET). Writes src/hyperkkl_bench/resources/real_battery_lg.json ("files" + "source"; the
"frozen" block is filled by `python -m hyperkkl_bench.real.battery_lg freeze` after identification).

    python tools/build_real_lg_resources.py [api.json]       # default: download the API record
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
API = "https://data.mendeley.com/public-api/datasets/b5mj79w5w9"
TEMPS = {"25degC": ("C20DisCh", "UDDS", "LA92", "Mixed1", "Mixed2", "Mixed3", "Mixed4", "Mixed5", "Mixed6", "Mixed7",
                    "Mixed8", "US06"),
         "0degC": ("US06", "HWFET", "Mixed8"), "10degC": ("US06", "HWFET", "Mixed8"), "40degC": ("US06", "HWFET")}   # 40 degC Mixed8 is truncated (1438 s, aborted)

d = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else json.loads(urllib.request.urlopen(
    urllib.request.Request(API, headers={"User-Agent": "hyperkkl-bench"}), timeout=60).read())
files = []
for f in d["files"]:
    m = re.match(r".*?_(\w+?)_(n?\d+degC)_LGHG2\.mat$", f["filename"])
    if not m or m.group(2) not in TEMPS or m.group(1) not in TEMPS[m.group(2)]:
        continue
    c = f["content_details"]
    files.append({"path": f"{m.group(2)}/{f['filename']}", "url": c["download_url"], "size": int(c["size"]),
                  "sha256": c["sha256_hash"]})
files.sort(key=lambda x: x["path"])
want = sum(len(v) for v in TEMPS.values())
assert len(files) == want, (len(files), want)
lic = d["data_licence"]
p = HERE / "src" / "hyperkkl_bench" / "resources" / "real_battery_lg.json"
old = {}                                   # the identification is re-frozen after any change of the file list
out = {"dataset": "battery_lg",
       "source": {"title": d["name"], "creator": "Phillip Kollmeyer, Mina Naguib, Michael Skells (McMaster University)",
                  "repository": f"Mendeley Data, version {d['version']}", "doi": d["doi"]["id"],
                  "landing": "https://data.mendeley.com/datasets/b5mj79w5w9/2", "metadata_api": API,
                  "license": lic["short_name"], "license_url": lic["url"], "license_text": lic["description"],
                  "redistribution": "allowed by CC BY 4.0 with attribution; this package nevertheless does NOT redistribute "
                                    "the raw files: they are fetched from Mendeley Data and checked against Mendeley's SHA-256",
                  "citation": "P. Kollmeyer, M. Naguib, M. Skells, LG 18650HG2 Li-ion Battery Data, Mendeley Data, v2, 2020, "
                              "doi:10.17632/b5mj79w5w9.2"},
       "files": files, "frozen": old.get("frozen", {})}
p.write_text(json.dumps(out, indent=1))
print("wrote", p, len(files), "files", sum(f["size"] for f in files) / 1e6, "MB")
