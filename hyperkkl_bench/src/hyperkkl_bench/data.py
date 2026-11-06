"""Loading frozen episodes.

    from hyperkkl_bench import load
    train = load("vdp", "realistic", "train")          # EpisodeSet
    ep = train[0]                                      # Episode(u, y, x, dt, meta)
    train.u.shape, train.y.shape, train.x.shape        # (E, N+1), (E, N+1, n_y), (E, N+1, n_x)

Sample k is at t = k dt, k = 0..N. ``u[k]`` is held constant on [t_k, t_{k+1}) (zero-order hold),
so x[k+1] depends on u[0..k]; y[k] = h(x[k]) + noise.

Splits
    train, val                      training / model selection (training families, train seed 0)
    phase1                          zero-input episodes over the operating domain (optional use)
    val_ood                         PUBLIC out-of-distribution validation: 4 input generators that
                                    are neither training nor test generators (regen_decay,
                                    gated_sine, spline, urban_cycle); for model selection under
                                    input shift. Never a test number.
    test_id, test_shape_ood,        the frozen public test tiers (labels public; report these)
    test_sequence_ood
    hidden_id, hidden_shape_ood,    fresh draws of the same tier generators with the labels
    hidden_sequence_ood             WITHHELD (x is absent); scored only by hidden.score_hidden

Knowledge levels (what a model-based method may use; see PROTOCOL.md)
    exact      data 'white'      model = true f, nominal parameters        (HF config C1)
    partial    data 'white'      model = f_known (one term missing)         (C3-partial)
    mismatch   data 'mismatch30' model = true structure, NOMINAL parameters (C3-mismatch30)
    realistic  data 'realistic'  model = f_known at nominal parameters, colored Student-t noise (C3-realistic)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import data_root, extra_root
from .registry import KNOWLEDGE, PLANTS, check_plant_name, variant_of

PUBLIC_SPLITS = ("train", "val", "phase1", "val_ood", "test_id", "test_shape_ood", "test_sequence_ood")
HIDDEN_SPLITS = ("hidden_id", "hidden_shape_ood", "hidden_sequence_ood")
TEST_TIERS = ("id", "shape_ood", "sequence_ood")
_EXPORT_NAME = {"train": "train", "val": "validation", "phase1": "phase1", "test_id": "id",
                "test_shape_ood": "shape_ood", "test_sequence_ood": "sequence_ood"}


@dataclass
class Episode:
    u: np.ndarray            # (N+1,)       applied input (zero-order hold)
    y: np.ndarray            # (N+1, n_y)   noisy measurements
    x: np.ndarray | None     # (N+1, n_x)   true state (None in hidden splits)
    dt: float
    meta: dict = field(default_factory=dict)   # family, x0, signal parameters, index


@dataclass
class EpisodeSet:
    """A split, stacked: u (E, N+1), y (E, N+1, n_y), x (E, N+1, n_x) or None."""
    u: np.ndarray
    y: np.ndarray
    x: np.ndarray | None
    dt: float
    plant: str
    knowledge: str
    split: str
    families: list
    signals: list
    x0: np.ndarray
    episode_ids: np.ndarray
    source: str = ""

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, i) -> Episode:
        return Episode(self.u[i], self.y[i], None if self.x is None else self.x[i], self.dt,
                       {"family": self.families[i], "signal": self.signals[i], "x0": self.x0[i],
                        "episode": int(self.episode_ids[i]), "plant": self.plant,
                        "split": self.split, "knowledge": self.knowledge})

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def n_steps(self) -> int:
        return self.u.shape[1]

    def subset(self, idx) -> "EpisodeSet":
        idx = np.asarray(idx, dtype=int)
        return EpisodeSet(self.u[idx], self.y[idx], None if self.x is None else self.x[idx], self.dt,
                          self.plant, self.knowledge, self.split, [self.families[i] for i in idx],
                          [self.signals[i] for i in idx], self.x0[idx], self.episode_ids[idx],
                          self.source)


def _stack(col, n_inner=None) -> np.ndarray:
    """pyarrow list<list<float>> or list<float> column -> numpy (E, L[, d]) without Python loops."""
    import pyarrow as pa
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    E = len(arr)
    flat = arr
    while pa.types.is_list(flat.type) or pa.types.is_large_list(flat.type):
        flat = flat.flatten()
    v = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    if n_inner is None:
        return v.reshape(E, -1)
    return v.reshape(E, -1, n_inner)


def _file_for(plant: str, knowledge: str, split: str, root=None, extra=None) -> Path:
    v = variant_of(knowledge)
    if PLANTS[plant].get("extra_root"):                     # deployment-track plants live in data_extra/
        if split not in _EXPORT_NAME:
            raise ValueError(f"{plant}: split {split!r} not available (deployment track: {list(_EXPORT_NAME)})")
        return extra_root(extra) / "data" / plant / v / f"{_EXPORT_NAME[split]}.parquet"
    if split in _EXPORT_NAME:
        return data_root(root) / "data" / plant / v / f"{_EXPORT_NAME[split]}.parquet"
    if split == "val_ood" or split in HIDDEN_SPLITS:
        return extra_root(extra) / "data" / plant / v / f"{split}.parquet"
    raise ValueError(f"unknown split {split!r}; public splits: {PUBLIC_SPLITS}, hidden: {HIDDEN_SPLITS}")


@lru_cache(maxsize=64)
def _metadata_raw(plant: str, root=None) -> dict:
    check_plant_name(plant)
    base = extra_root() if PLANTS[plant].get("extra_root") else data_root(root)
    return json.loads((base / "metadata" / f"{plant}.json").read_text())


REAL_PLANT_KEYS = ("mismatch30_params",)    # the answer key of the mismatch / realistic levels (PITFALLS P9)


def metadata(plant: str, root=None, include_real_plant: bool = False) -> dict:
    """metadata/<plant>.json of the dataset (x_std, sensors, unmeasured, settle_index, ...).
    The real plant's parameters are removed unless ``include_real_plant=True`` (documentation /
    data generation only; a method must never use them, PROTOCOL.md section 1)."""
    import copy
    m = copy.deepcopy(_metadata_raw(plant, root))
    if not include_real_plant:
        for k in REAL_PLANT_KEYS:
            m.pop(k, None)
    return m


def load(plant: str, knowledge: str = "exact", split: str = "train", n: int | None = None,
         seed: int = 0, root=None, extra=None, recipe: str = "standard") -> EpisodeSet:
    """Load one split of one plant at one knowledge level.

    v2 (two families): ``load("driven/battery/realistic", "test_op_ood")`` or
    ``load("autonomous/vdp/partial", "train", recipe="mixed")``: the first argument is a "family/plant/level" spec
    and the second the split (see hyperkkl_bench.families). The v1 call ``load(plant, level, split)`` is unchanged.

    n, seed   optional deterministic subset of n episodes (data-efficiency studies; the subset
              for (n, seed) is nested in the subset for any larger n with the same seed).
    """
    if "/" in plant:
        from . import families
        sp = knowledge if knowledge not in KNOWLEDGE else split
        es = families.load(plant, sp, recipe=recipe, root=root)
        if n is not None:
            if n > len(es):
                raise ValueError(f"{plant}/{sp}: asked for {n} episodes, only {len(es)} exist")
            cmd = getattr(es, "cmd", None)
            idx = np.sort(np.random.RandomState(seed).permutation(len(es))[:n])
            es2 = es.subset(idx)
            es2.cmd = None if cmd is None else cmd[idx]
            es2.family_v2 = getattr(es, "family_v2", None)
            return es2
        return es
    check_plant_name(plant)
    if knowledge not in KNOWLEDGE:
        raise ValueError(f"unknown knowledge level {knowledge!r}; choose from {list(KNOWLEDGE)}")
    if PLANTS[plant].get("generated_only"):
        raise ValueError(f"{plant}: no frozen files; see plant_card({plant!r})['how_to_get_data']")
    path = _file_for(plant, knowledge, split, root, extra)
    if not path.exists():
        hint = ("run `python tools/build_extra_splits.py` (generates val_ood / hidden on CPU)"
                if split == "val_ood" or split in HIDDEN_SPLITS else "check HYPERKKL_BENCH_DATA")
        raise FileNotFoundError(f"{path} missing; {hint}")
    import pyarrow.parquet as pq
    tb = pq.read_table(path)
    meta = metadata(plant, root)
    nx, ny = int(meta["n_x"]), int(meta["n_y"])
    names = tb.column_names
    x = _stack(tb.column("x"), nx) if "x" in names else None
    u = _stack(tb.column("u"))
    y = _stack(tb.column("y"), ny)
    dts = np.asarray(tb.column("dt").to_numpy(), float)
    if not np.allclose(dts, dts[0]):
        raise ValueError(f"{path}: mixed sample times")
    es = EpisodeSet(u=u, y=y, x=x, dt=float(dts[0]), plant=plant, knowledge=knowledge, split=split,
                    families=tb.column("family").to_pylist(), signals=tb.column("signal").to_pylist(),
                    x0=_stack(tb.column("x0")) if "x0" in names else np.full((tb.num_rows, nx), np.nan, np.float32),
                    episode_ids=np.asarray(tb.column("episode").to_numpy()),
                    source=str(path))
    if n is not None:
        if n > len(es):
            raise ValueError(f"{plant}/{split}: asked for {n} episodes, only {len(es)} exist")
        es = es.subset(np.sort(np.random.RandomState(seed).permutation(len(es))[:n]))
    return es


def episode_counts(plant: str, root=None) -> dict:
    meta = metadata(plant, root)
    return {v: s["n"] for v, s in meta["settings"].items()}
