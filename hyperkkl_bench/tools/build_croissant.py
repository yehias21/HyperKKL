"""Write croissant.json (MLCommons Croissant 1.0 core fields) describing the Parquet layout."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [("episode", "sc:Integer", "index within the split"),
          ("family", "sc:Text", "input family"),
          ("signal", "sc:Text", "JSON parameters of the input realisation"),
          ("dt", "sc:Float", "sample time (s)"),
          ("u", "sc:Float", "applied input, list of N+1 (zero-order hold)"),
          ("y", "sc:Float", "noisy measurements, list of N+1 lists of n_y"),
          ("x", "sc:Float", "true state, list of N+1 lists of n_x (absent in hidden splits; NaN = unlabelled in cstr_exo training splits)"),
          ("x0", "sc:Float", "initial condition (absent in hidden splits)")]
doc = {
    "@context": {"@language": "en", "@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/",
                 "sc": "https://schema.org/", "dct": "http://purl.org/dc/terms/"},
    "@type": "sc:Dataset", "conformsTo": "http://mlcommons.org/croissant/1.0",
    "name": "hyperkkl-bench", "version": "2.0.0rc1",
    "description": "Simulated nonlinear plants for benchmarking state estimators in two families: AUTONOMOUS systems (no "
                   "input; limit cycles, chaos, oscillator networks) and NON-AUTONOMOUS input-driven systems under "
                   "realistic operating profiles (v2, data_v2/), with the v1 input-shape / input-sequence tiers kept as "
                   "legacy tiers and incomplete model knowledge, plus real-data tracks (quadrotor velocity and tilt, battery SOC, "
                   "positioning-axis and oscillator velocity) whose raw data are NOT distributed: they are fetched "
                   "from the original hosts (cr:FileObject entries below, with their SHA-256 and licences).",
    "license": "https://opensource.org/licenses/MIT", "url": "TBD",
    "distribution": [
        {"@type": "cr:FileSet", "@id": "core-parquet", "name": "core-parquet", "encodingFormat": "application/x-parquet",
         "includes": "data/*/*/*.parquet", "description": "HF export: data/<plant>/<variant>/<split>.parquet"},
        {"@type": "cr:FileSet", "@id": "extra-parquet", "name": "extra-parquet", "encodingFormat": "application/x-parquet",
         "includes": "data_extra/data/*/*/*.parquet",
         "description": "val_ood, label-free hidden splits, deployment and scaling plants"},
        {"@type": "cr:FileSet", "@id": "v2-parquet", "name": "v2-parquet", "encodingFormat": "application/x-parquet",
         "includes": "data_v2/data/*/*/*/*.parquet",
         "description": "v2 (hkb-data-2.0): data_v2/data/<family>/<plant>/<variant>/<split>.parquet; families autonomous / "
                        "driven; splits train, train_event, val, val_event, val_ood, phase1, test tiers, label-free hidden_*"},
        {"@type": "cr:FileSet", "@id": "v2-metadata", "name": "v2-metadata", "encodingFormat": "application/json",
         "includes": "data_v2/metadata/*/*.json",
         "description": "v2 per-cell metadata (x_std per variant, sensors, input box, actuator, operators, SHA-256 per file)"},
        {"@type": "cr:FileSet", "@id": "metadata", "name": "metadata", "encodingFormat": "application/json",
         "includes": "metadata/*.json", "description": "per-plant metadata with SHA-256 of every file"},
    ],
    "recordSet": [{"@type": "cr:RecordSet", "@id": "episodes", "name": "episodes",
                   "field": [{"@type": "cr:Field", "@id": f"episodes/{n}", "name": n, "dataType": t, "description": d,
                              "source": {"fileSet": {"@id": "core-parquet"}, "extract": {"column": n}}}
                             for n, t, d in FIELDS]},
                  {"@type": "cr:RecordSet", "@id": "episodes_v2", "name": "episodes_v2",
                   "field": [{"@type": "cr:Field", "@id": f"episodes_v2/{n}", "name": n, "dataType": t, "description": d,
                              "source": {"fileSet": {"@id": "v2-parquet"}, "extract": {"column": n}}}
                             for n, t, d in FIELDS + [("cmd", "sc:Float", "operator command (profile test tiers only; "
                                                       "used by the input_cmd_vs_applied condition)")]]}],
}
import sys
sys.path.insert(0, str(ROOT / "src"))
from hyperkkl_bench.real import list_real, spec  # noqa: E402
for d in list_real():
    sp = spec(d)
    for f in sp["files"]:
        doc["distribution"].append({
            "@type": "cr:FileObject", "@id": f"real/{d}/{f['path']}", "name": f"{d}:{Path(f['path']).name}",
            "contentUrl": f["url"], "sha256": f["sha256"], "contentSize": f"{f.get('size', '')} B",
            "encodingFormat": "application/zip" if f["path"].endswith(".zip") else ("text/plain" if f["path"].endswith((".md", ".txt")) else "application/x-matlab-data"),
            "description": f"real-data track {d}: fetched from the original host (not redistributed). Licence: "
                           f"{sp['source']['license']}. Cite: {sp['source']['citation']}"})
(ROOT / "croissant.json").write_text(json.dumps(doc, indent=1))
print("wrote croissant.json")
