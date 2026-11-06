"""Parquet writer shared by the export tools (same schema as the core export)."""
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def nested(a):
    E, T, d = a.shape
    vals = pa.array(a.reshape(-1).astype(np.float32), type=pa.float32(), from_pandas=True)
    inner = pa.ListArray.from_arrays(pa.array(np.arange(0, E * T * d + 1, d, dtype=np.int32)), vals)
    return pa.ListArray.from_arrays(pa.array(np.arange(0, E * T + 1, T, dtype=np.int32)), inner)


def flat(a):
    E, T = a.shape
    vals = pa.array(a.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.ListArray.from_arrays(pa.array(np.arange(0, E * T + 1, T, dtype=np.int32)), vals)


def write(es, path):
    E = len(es)
    x0 = np.stack([np.asarray(m.get("ic", np.full(es.x.shape[-1], np.nan)), np.float32) for m in es.meta]) \
        if es.meta else es.x[:, 0].numpy()
    tbl = pa.table({
        "episode": pa.array(np.arange(E, dtype=np.int32)),
        "family": pa.array(list(es.families), type=pa.string()),
        "x0": nested(x0[:, None, :]).flatten(),
        "signal": pa.array([json.dumps({k: v for k, v in m.items() if k != "ic"}, default=str) for m in es.meta]),
        "dt": pa.array(np.full(E, es.dt, dtype=np.float64)),
        "u": flat(es.u[..., 0].numpy()),
        "y": nested(es.y.numpy()),
        "x": nested(es.x.numpy()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd", compression_level=6, row_group_size=32)
    return E


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


