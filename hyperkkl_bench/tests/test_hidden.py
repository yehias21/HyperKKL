import json
import numpy as np
import pytest
import hyperkkl_bench as hkb
from hyperkkl_bench import hidden
from hyperkkl_bench.paths import labels_root
from conftest import needs_data

needs_labels = pytest.mark.skipif(not (labels_root() / "data").is_dir(), reason="hidden labels are host-only")


@needs_data
@needs_labels
def test_predict_and_score_roundtrip(tmp_path):
    est = hkb.baselines.PriorMean()
    est.fit(hkb.load("vdp", "exact", "train", n=20), None, hkb.build_context("vdp", "exact", with_model=False))
    hidden.predict(est, "vdp", "exact", tmp_path)
    man = json.loads((tmp_path / "SUBMISSION.json").read_text())
    assert man["cells"] == [["vdp", "exact"]]
    sc = hidden.score_hidden(tmp_path)
    cell = sc["cells"]["vdp/exact"]
    for s in hkb.HIDDEN_SPLITS:
        v = cell[s]["summary"]["nrmse_unmeasured"]
        assert 0.3 < v < 3.0                        # the prior-mean floor is ~1 by construction


@needs_data
@needs_labels
def test_perfect_submission_scores_zero(tmp_path):
    from hyperkkl_bench.hidden import _labels
    for s in hkb.HIDDEN_SPLITS:
        X, ep = _labels("fhn", "realistic", s, None)
        p = tmp_path / "fhn" / "realistic" / f"{s}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, x_hat=X, episode=ep)
    (tmp_path / "SUBMISSION.json").write_text(json.dumps({"cells": [["fhn", "realistic"]], "estimator": "oracle"}))
    sc = hidden.score_hidden(tmp_path)
    for s in hkb.HIDDEN_SPLITS:
        assert sc["cells"]["fhn/realistic"][s]["summary"]["nrmse"] < 1e-12


@needs_data
@needs_labels
def test_wrong_shape_is_an_error(tmp_path):
    p = tmp_path / "vdp" / "exact" / "hidden_id.npz"
    p.parent.mkdir(parents=True)
    np.savez(p, x_hat=np.zeros((3, 5, 2)), episode=np.arange(3))
    (tmp_path / "SUBMISSION.json").write_text(json.dumps({"cells": [["vdp", "exact"]]}))
    sc = hidden.score_hidden(tmp_path)
    assert "error" in sc["cells"]["vdp/exact"]["hidden_id"]
    assert sc["cells"]["vdp/exact"]["hidden_shape_ood"] == {"error": "missing"}
