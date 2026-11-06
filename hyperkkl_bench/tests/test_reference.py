import importlib.util
from pathlib import Path
from conftest import needs_repo

HERE = Path(__file__).resolve().parents[1]


@needs_repo
def test_reference_table_regenerates_and_matches():
    spec = importlib.util.spec_from_file_location("rr", HERE / "reference" / "reproduce_reference.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    assert rr.main(["--check"]) == 0
