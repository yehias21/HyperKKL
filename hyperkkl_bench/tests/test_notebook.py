"""The quick-start notebook runs top to bottom (code cells executed as one script, in a temp dir)."""
import json
import os
from pathlib import Path
import pytest
from conftest import needs_data

NB = Path(__file__).resolve().parents[1] / "notebooks" / "quickstart.ipynb"


@pytest.mark.slow
@needs_data
def test_quickstart_runs(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    nb = json.loads(NB.read_text())
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    g = {"__name__": "__main__"}
    exec(compile(code, str(NB), "exec"), g)
    assert (tmp_path / "quickstart_gru_report.json").exists()
    for r in (g["rep_ekf"], g["rep_gru"]):
        assert 0 < r.value("test_id") < 2.0 and r.cost["latency_us_median"] > 0
