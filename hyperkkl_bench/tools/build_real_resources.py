"""Maintainer tool: freeze the battery_pf real-data track constants from the research run
(experiments/real: raw_manifest.json, results.json). READ ONLY on the research repository.

Writes src/hyperkkl_bench/resources/real_battery_pf.json:
  files        the raw files the track needs (path, original Mendeley URL, size, SHA-256 published by Mendeley)
  frozen       identified ECM parameters, thermocouple offset, Q, x_std, tuned EKF / UKF noise (validation)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REAL = HERE.parent / "experiments" / "real"
NEED = ("25degC/Drive_cycles/", "0degC/Drive_cycles/", "C20_OCV_Test_C20_25dC.mat", "Readme_file")

man = json.loads((REAL / "raw_manifest.json").read_text())
res = json.loads((REAL / "results.json").read_text())
files = [{"path": m["path"], "url": m["url"], "size": m["size"], "sha256": m["sha256_mendeley"]}
         for m in man if any(k in m["path"] for k in NEED) and "LA92_NN" not in m["path"]]
assert all(m["sha256"] == m["sha256_mendeley"] for m in man)
out = {
    "dataset": "battery_pf",
    "source": {
        "title": "Panasonic 18650PF Li-ion Battery Data",
        "creator": "Phillip Kollmeyer (University of Wisconsin-Madison)",
        "repository": "Mendeley Data, version 1 (2018-06-21)",
        "doi": "10.17632/wykht8y7tg.1",
        "landing": "https://data.mendeley.com/datasets/wykht8y7tg/1",
        "metadata_api": "https://data.mendeley.com/public-api/datasets/wykht8y7tg",
        "license": "CC BY 4.0",
        "license_url": "http://creativecommons.org/licenses/by/4.0",
        "redistribution": "allowed by CC BY 4.0 with attribution; this package nevertheless does NOT redistribute "
                          "the raw files: they are fetched from Mendeley Data and checked against Mendeley's SHA-256",
        "citation": "P. Kollmeyer, Panasonic 18650PF Li-ion Battery Data, Mendeley Data, v1, 2018, "
                    "doi:10.17632/wykht8y7tg.1",
    },
    "files": files,
    "frozen": {
        "Q_Ah": res["Q_Ah"],
        "thermocouple_offset_K": res["thermocouple_offset_K"],
        "params_25C": res["identification"]["params"],
        "ecm_fit": res["identification"]["ecm"],
        "thermal_fit": res["identification"]["thermal"],
        "ocv_fit_rms_V": res["identification"]["ocv_fit"]["rms"],
        "arrhenius_factor_0C": res["arrhenius_factor_0C"],
        "x_std": res["scales"]["x_std"],
        "x_mean": res["scales"]["x_mean"],
        "y_std": res["scales"]["y_std"],
        "filter_tuning": {k: v["choice"] for k, v in res["filter_tuning"].items()},
        "n_episodes": res["n_episodes"],
        "n_windows": res["n_windows"],
        "open_loop_model_fit": res["open_loop_model_fit"],
    },
    "provenance": "experiments/real/{raw_manifest.json,results.json} (research run of 2026-10-25/26)",
}
p = HERE / "src" / "hyperkkl_bench" / "resources" / "real_battery_pf.json"
p.write_text(json.dumps(out, indent=1))
print("wrote", p, len(files), "files")
