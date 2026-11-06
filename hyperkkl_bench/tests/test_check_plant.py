import json
import subprocess
import sys
import pytest
from hyperkkl_bench.check import check_plant
from hyperkkl_bench import metadata
from conftest import needs_data


@pytest.mark.slow
@needs_data
def test_vdp_passes():
    r = check_plant(dict(metadata("vdp")["system_config"]), fast=True, verbose=False)
    assert r["interface"]["pass"] and r["design"]["ok"] and r["observable"]["pass"] and r["bounded"]["pass"]
    assert r["pass"]


@pytest.mark.slow
@needs_data
def test_unbounded_cartpole_is_rejected():
    spec = dict(metadata("legacy_cartpole")["system_config"])
    r = check_plant(spec, fast=True, verbose=False)
    assert not r["bounded"]["pass"] and not r["pass"]


@needs_data
def test_cli_list_and_card(tmp_path):
    out = subprocess.run([sys.executable, "-m", "hyperkkl_bench", "list"], capture_output=True, text=True)
    assert out.returncode == 0 and "two_link_arm" in out.stdout and "legacy" not in out.stdout
    out = subprocess.run([sys.executable, "-m", "hyperkkl_bench", "card", "vdp", "--md"], capture_output=True, text=True)
    assert out.returncode == 0 and "repairability" in out.stdout


@pytest.mark.slow
@needs_data
def test_cli_check_plant_contributed(tmp_path):
    """A contributed plant given as module:Class (here the Duffing class re-exported) is checked."""
    spec = dict(metadata("duffing")["system_config"])
    spec["plant"] = "hyperkkl_bench.plants.systems:Duffing"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    out = subprocess.run([sys.executable, "-m", "hyperkkl_bench", "check-plant", str(p), "--fast",
                          "--out", str(tmp_path / "r.json")], capture_output=True, text=True)
    r = json.loads((tmp_path / "r.json").read_text())
    assert out.returncode == 0, out.stdout + out.stderr
    assert r["pass"]
