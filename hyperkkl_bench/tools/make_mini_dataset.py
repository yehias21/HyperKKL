"""Write tests/fixtures/mini: a tiny CI dataset (vdp, all 3 variants, 6 episodes per split, first 401
samples) in the release layout, so the data-free CI can run the whole pipeline end to end.
NOT a benchmark: numbers on it mean nothing."""
import copy
import json
from pathlib import Path

import pyarrow.parquet as pq

from hyperkkl_bench.data import _metadata_raw
from hyperkkl_bench.paths import data_root

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mini"
N1, E = 401, 6


def main():
    src = data_root()
    meta = copy.deepcopy(_metadata_raw("vdp"))
    for f in list(meta["files"]):
        tb = pq.read_table(src / f).slice(0, E)
        cols = {}
        for name in tb.column_names:
            col = tb.column(name).to_pylist()
            if name in ("u", "y", "x"):
                col = [c[:N1] for c in col]
            cols[name] = col
        import pyarrow as pa
        out = OUT / f
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols, schema=tb.schema), out, compression="zstd")
    meta.update({"N": N1 - 1, "T": (N1 - 1) * meta["dt"], "settle_index": (N1 - 1) // 2,
                 "files": {}, "note": "CI fixture: 6 episodes per split, first 401 samples"})
    for v in meta["settings"].values():
        v["n"] = {k: E for k in v["n"]}
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    (OUT / "metadata" / "vdp.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
