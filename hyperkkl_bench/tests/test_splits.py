"""Split integrity: no episode appears in two splits; generators of the OOD splits are disjoint."""
import hashlib
import numpy as np
import pytest
import hyperkkl_bench as hkb
from hyperkkl_bench.plants import signals as S
from conftest import needs_data

SPLITS = ["train", "val", "val_ood", "test_id", "test_shape_ood", "test_sequence_ood",
          "hidden_id", "hidden_shape_ood", "hidden_sequence_ood"]


def _u_hashes(es):
    return {hashlib.sha1(np.ascontiguousarray(u).tobytes()).hexdigest() for u in es.u if np.ptp(u) > 0}


@needs_data
@pytest.mark.parametrize("plant", hkb.list_plants())
def test_no_overlap(plant):
    seen = {}
    for s in SPLITS:
        es = hkb.load(plant, "realistic", s)
        hs = _u_hashes(es)
        for t, h in seen.items():
            assert not (hs & h), f"{plant}: input sequences shared by {s} and {t}"
        seen[s] = hs


@needs_data
def test_generator_families_disjoint():
    fam = lambda s: set(hkb.load("vdp", "exact", s).families)
    tr, sh, sq, vo = fam("train"), fam("test_shape_ood"), fam("test_sequence_ood"), fam("val_ood")
    assert not tr & sh
    assert sq == {"drive"} and "drive" in tr          # same family, different event sequence
    assert not vo & (tr | sh | sq)
    assert set(S.SHAPE_OOD_FAMILIES) == sh


@needs_data
def test_hidden_differs_from_public_test():
    a = hkb.load("vdp", "exact", "test_id")
    b = hkb.load("vdp", "exact", "hidden_id")
    assert not (_u_hashes(a) & _u_hashes(b))
