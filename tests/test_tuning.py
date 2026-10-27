"""C1 per-variant tuning (experiments/tuning/PROTOCOL.md): architecture knobs, per-method
phase-2 overrides, the validation-NRMSE selection score, and bit-identity of the default.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_tuning.py -q

Every cache write goes to a pytest tmp dir. CPU only; ~10 min idle, ~50 min on a loaded host.
"""

from __future__ import annotations

import importlib.util
import tarfile

import numpy as np
import pytest
import torch

import hyperkkl.data as Dm
from hyperkkl import conditioning as Cd
from hyperkkl import config as Cf
from hyperkkl import evaluate as Ev
from hyperkkl import job as J
from hyperkkl import train as T
from hyperkkl.cache import MODEL_MODULES, module_hash
from hyperkkl.metrics import episode_metrics
from hyperkkl.nets import BaseMap, n_params
from hyperkkl.observers import run

DEV = "cpu"
KINDS = ("latent", "concat", "film", "lora", "hyper")
KNOBS = {"hyper": {"gen_layers": 2}, "film": {"film_depth": 2}, "concat": {"concat_depth": 3},
         "latent": {"latent_depth": 3}}
DEFAULTS = {"hyper": {"gen_layers": 4}, "film": {"film_depth": 1}, "concat": {"concat_depth": 2},
            "latent": {"latent_depth": 2}}
# the C1-where code (conditioning.py of that snapshot = the code before the tuning knobs)
SNAPSHOT = Cf.ROOT / "experiments" / "code" / "a1e5d75cce66.tar.gz"


def _base(nz=5, nx=2, width=128, depth=3, seed=1):
    torch.manual_seed(seed)
    return BaseMap(nz, nx, width, depth, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(nx), torch.ones(nx)))


def _randomise(m, seed=3, s=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(0, s, generator=g)


@pytest.fixture(scope="module")
def snap_cond(tmp_path_factory):
    """The pre-tuning conditioning module, imported against the current (unchanged) nets."""
    if not SNAPSHOT.exists():
        pytest.skip(f"no code snapshot {SNAPSHOT}")
    d = tmp_path_factory.mktemp("snap")
    with tarfile.open(SNAPSHOT) as tf:
        tf.extract("hyperkkl/conditioning.py", d)
    spec = importlib.util.spec_from_file_location("hyperkkl._snap_conditioning", d / "hyperkkl" / "conditioning.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hyperkkl"
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# default architecture is bit-identical to the pre-tuning code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nz,nx", [(5, 2), (7, 2), (9, 4), (11, 5)])
@pytest.mark.parametrize("kind", KINDS)
def test_default_bit_identical_to_snapshot(snap_cond, kind, nz, nx):
    base = _base(nz, nx)
    z_std = torch.ones(nz)
    wo = snap_cond.match_width(kind, base, 16, 140000, 4, z_std)
    assert Cd.match_width(kind, base, 16, 140000, 4, z_std) == wo
    torch.manual_seed(7)
    mo = snap_cond.build_conditioner(kind, base, 16, wo[0], 4, z_std)
    torch.manual_seed(7)
    mn = Cd.build_conditioner(kind, base, 16, wo[0], 4, z_std)
    so, sn = mo.state_dict(), mn.state_dict()
    assert list(so) == list(sn) and all(torch.equal(so[k], sn[k]) for k in so)     # same init, same RNG use
    _randomise(mo)
    mn.load_state_dict(mo.state_dict())
    z, c = torch.randn(33, nz), torch.randn(33, 16)
    yo, yn = mo(z, c), mn(z, c)
    assert torch.equal(yo, yn)
    yo.pow(2).sum().backward()
    yn.pow(2).sum().backward()
    for a, b in zip(mo.parameters(), mn.parameters()):
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("kind", ["latent", "concat", "film", "hyper"])
def test_explicit_default_knob_equals_absent(kind):
    base, z_std = _base(), torch.ones(5)
    w, n = Cd.match_width(kind, base, 16, 140000, 4, z_std)
    assert Cd.match_width(kind, base, 16, 140000, 4, z_std, **DEFAULTS[kind]) == (w, n)
    torch.manual_seed(0)
    a = Cd.build_conditioner(kind, base, 16, w, 4, z_std)
    torch.manual_seed(0)
    b = Cd.build_conditioner(kind, base, 16, w, 4, z_std, **DEFAULTS[kind])
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    _randomise(a)
    b.load_state_dict(a.state_dict())
    z, c = torch.randn(9, 5), torch.randn(9, 16)
    assert torch.equal(a(z, c), b(z, c))


def test_arch_options_and_method_kind():
    assert Cd.arch_options(Cf.load("vdp")["phase2"]) == {}             # base.yaml sets no knob
    assert Cd.arch_options({"gen_layers": 2, "film_depth": None, "lr": 1.0}) == {"gen_layers": 2}
    for name, k in [("hyper", "hyper"), ("latent@current", "latent"), ("hyper[lr=0.001,gen_layers=2]", "hyper"),
                    ("latent[latent_depth=3]@gru:dim=8", "latent"), ("autonomous", "autonomous")]:
        assert Cd.method_kind(name) == k


# ---------------------------------------------------------------------------
# the knobs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["latent", "concat", "film", "lora", "hyper"])
def test_knob_collapses_at_zero_context_and_matches_budget(kind):
    base, z_std = _base(), torch.ones(5)
    rank = 16 if kind == "lora" else 4
    arch = KNOBS.get(kind, {})
    w, n = Cd.match_width(kind, base, 16, 140000, rank, z_std, **arch)
    assert abs(n - 140000) / 140000 < 0.04, (kind, w, n)
    m = Cd.build_conditioner(kind, base, 16, w, rank, z_std, **arch)
    assert n_params(m) == n
    _randomise(m, s=0.1)
    z = torch.randn(11, 5)
    if kind == "latent":
        assert m(z, torch.zeros(11, 16)).abs().max() == 0.0
        assert m(z, torch.randn(11, 16)).abs().max() > 0
    else:
        assert torch.allclose(m(z, torch.zeros(11, 16)), base(z), atol=1e-6)
        assert not torch.allclose(m(z, torch.randn(11, 16)), base(z), atol=1e-4)


def test_knob_structure():
    base, z_std = _base(), torch.ones(5)
    dims = [(l.in_features, l.out_features) for l in base.layers]          # 5-128-128-128-2
    w_all, _ = Cd.match_width("hyper", base, 16, 140000, 4, z_std)
    w2, _ = Cd.match_width("hyper", base, 16, 140000, 4, z_std, gen_layers=2)
    assert (w_all, w2) == (4, 8)                                            # trunk width k doubles
    h = Cd.build_conditioner("hyper", base, 16, w2, 4, z_std, gen_layers=2)
    assert h.gen == [2, 3] and [hh.out_features for hh in h.H] == [a * o + o for a, o in dims[2:]]
    # the first two layers are not conditioned: perturbing their (non-existent) heads is impossible
    # and the hidden features after layer 1 do not depend on c
    _randomise(h, s=0.1)
    z = torch.randn(7, 5)
    feats = []
    for c in (torch.zeros(7, 16), torch.randn(7, 16)):
        hh = base.inp.norm(z)
        for layer in base.layers[:2]:
            hh = base.act(layer(hh))
        feats.append(hh)
    assert torch.equal(*feats)
    with pytest.raises(ValueError):
        Cd.build_conditioner("hyper", base, 16, 4, 4, z_std, gen_layers=5)
    assert len(Cd.build_conditioner("film", base, 16, 10, 4, z_std, film_depth=2).G.layers) == 3
    assert len(Cd.build_conditioner("concat", base, 16, 10, 4, z_std, concat_depth=3).R.layers) == 4
    assert len(Cd.build_conditioner("latent", base, 16, 10, 4, z_std, latent_depth=3).M.layers) == 4
    # knobs of other kinds are ignored (one phase-2 section serves every method of a job)
    a = Cd.build_conditioner("lora", base, 16, 10, 4, z_std)
    b = Cd.build_conditioner("lora", base, 16, 10, 4, z_std, gen_layers=2, film_depth=2, concat_depth=3, latent_depth=3)
    assert [p.shape for p in a.parameters()] == [p.shape for p in b.parameters()]


# ---------------------------------------------------------------------------
# per-method phase-2 overrides: kind[key=value,...]
# ---------------------------------------------------------------------------

def test_method_cfg_brackets():
    cfg = Cf.load("vdp")
    p2 = cfg["phase2"]
    k, mc = J.method_cfg(cfg, "hyper[lr=1e-3,gen_layers=2]")
    assert k == "hyper" and mc["phase2"]["lr"] == 1e-3 and isinstance(mc["phase2"]["lr"], float)
    assert mc["phase2"]["gen_layers"] == 2 and cfg["phase2"] == p2 and "gen_layers" not in cfg["phase2"]
    assert {kk: v for kk, v in mc["phase2"].items() if kk not in ("lr", "gen_layers")} == \
        {kk: v for kk, v in p2.items() if kk != "lr"}
    k, mc = J.method_cfg(cfg, "lora[rank=16]@gru:dim=8")
    assert k == "lora" and mc["phase2"]["rank"] == 16 and mc["phase2"]["context"] == {**p2["context"], "kind": "gru", "dim": 8}
    k, mc = J.method_cfg(cfg, "autonomous[lr=0.1]")
    assert k == "none" and mc["phase2"]["lr"] == 0.1
    for bad in ("hyper[lr=1e-3", "hyper[foo=1]", "hyper[context=1]", "hyper[lr]", "hyper[=3]"):
        with pytest.raises(ValueError):
            J.method_cfg(cfg, bad)
    # without brackets: the old parser's results, the config object itself when nothing changes
    assert J.method_cfg(cfg, "lora")[1] is cfg
    assert J.method_cfg(cfg, "lora@window:window=20,dim=8")[1]["phase2"]["context"] == \
        {**p2["context"], "kind": "window", "window": 20, "dim": 8}


def _key(cfg, m, seed=0):
    kind, mc = J.method_cfg(cfg, m)
    return f"{kind}_{Cf.digest(mc, 'system', 'data', 'phase1', 'phase2', extra=[seed, kind, module_hash(MODEL_MODULES)])}"


def test_bracket_cache_keys():
    cfg = Cf.load("vdp")
    assert cfg["phase2"]["lr"] == 2e-3
    # restating a default is the same model (the selection's default config is C1-where's)
    assert _key(cfg, "hyper[lr=0.002]") == _key(cfg, "hyper") == _key(cfg, "hyper[lr=2e-3]")
    assert _key(cfg, "lora[rank=4]") == _key(cfg, "lora")
    # every grid point is its own model
    grid = [f"{k}[lr={lr}{x}]" for k, xs in (("hyper", ("", ",gen_layers=2")), ("lora", ("", ",rank=16")),
                                              ("film", ("", ",film_depth=2")), ("concat", ("", ",concat_depth=3")),
                                              ("latent", ("", ",latent_depth=3")))
            for x in xs for lr in ("5e-4", "1e-3", "2e-3", "4e-3")]
    assert len(grid) == 40 and len({_key(cfg, m) for m in grid}) == 40
    # a node-level set and a per-method bracket give the same artefact (selection -> C1-tuned)
    cfg2 = Cf.load("vdp", {"phase2.lr": 1e-3, "phase2.gen_layers": 2})
    assert _key(cfg2, "hyper") == _key(cfg, "hyper[lr=0.001,gen_layers=2]")


def test_build_observers_recognises_bracketed_latent():
    cfg = Cf.load("vdp", {"baselines": []})
    des = Dm.make_design(cfg["system"])
    nz = des.A.shape[0]
    base = {"T0": _base(2, nz, 8, 1), "T0inv": _base(nz, 2, 8, 1)}
    inj, dec, ctx = object(), torch.nn.Identity(), object()
    obs = Ev.build_observers(cfg, des, {"x_std": [1.0, 1.0], "y_std": [1.0]}, base,
                             {"latent[lr=0.001,latent_depth=3]": {"cond": inj, "ctx": ctx},
                              "hyper[gen_layers=2]": {"cond": dec, "ctx": ctx}}, None, "cpu")
    lat = obs["latent[lr=0.001,latent_depth=3]"]
    assert lat.inj is inj and lat.dec.kind == "none" and lat.dec.base is base["T0inv"]
    assert obs["hyper[gen_layers=2]"].dec is dec and obs["hyper[gen_layers=2]"].inj is None


# ---------------------------------------------------------------------------
# small real run: phase 2 with knobs, defaults vs snapshot builders, validation NRMSE
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vdp(tmp_path_factory):
    old = Dm.CACHE
    Dm.CACHE = tmp_path_factory.mktemp("cache")
    cfg = Cf.load("vdp", [{"data.n_p1": 48, "data.n_train": 48, "data.n_val": 16, "data.n_coll": 4096,
                           "phase1.steps_enc": 200, "phase1.steps_dec": 200, "phase2.steps": 20,
                           "phase2.val_every": 10, "phase2.episodes_per_batch": 8, "phase2.times_per_episode": 32,
                           "phase2.rollout_K": 16, "phase2.budget": 20000, "cuda_graphs": False}])
    sc = Dm.system_scales(cfg, DEV)
    des = Dm.make_design(cfg["system"])
    data = Dm.training_data(cfg, 0, DEV)
    base = T.phase1(cfg, data, sc, des, DEV, 0, log=lambda m: None)
    yield {"cfg": cfg, "scales": sc, "des": des, "data": data, "base": base}
    Dm.CACHE = old


def _p2(v, m, **kw):
    kind, mc = J.method_cfg(v["cfg"], m)
    return T.phase2(kind, mc, v["data"], v["scales"], v["des"], v["base"], DEV, 0, log=lambda s: None, **kw)


@pytest.mark.parametrize("kind", KINDS)
def test_phase2_default_equals_snapshot_builders(vdp, snap_cond, kind, monkeypatch):
    new = _p2(vdp, kind)
    assert new["arch"] == {} and new["lr"] == 2e-3
    seen = []

    def mw(k, base, dc, budget, rank, z_std, **arch):
        seen.append(arch)
        return snap_cond.match_width(k, base, dc, budget, rank, z_std)

    def bc(k, base, dc, w, rank, z_std, **arch):
        seen.append(arch)
        return snap_cond.build_conditioner(k, base, dc, w, rank, z_std)

    monkeypatch.setattr(T, "match_width", mw)
    monkeypatch.setattr(T, "build_conditioner", bc)
    old = _p2(vdp, kind)
    assert seen == [{}, {}]                                  # absent keys reach the builders as nothing
    assert new["width"] == old["width"] and new["n_params"] == old["n_params"]
    assert np.array_equal(np.asarray(new["hist"]), np.asarray(old["hist"]))
    so, sn = old["cond"].state_dict(), new["cond"].state_dict()
    assert all(torch.equal(so[k], sn[k]) for k in so)


@pytest.mark.parametrize("m", ["hyper[lr=4e-3,gen_layers=2]", "lora[lr=5e-4,rank=16]", "film[film_depth=2]",
                               "concat[lr=1e-3,concat_depth=3]", "latent[lr=1e-3,latent_depth=3]"])
def test_phase2_with_knobs(vdp, m):
    r = _p2(vdp, m)
    kind, mc = J.method_cfg(vdp["cfg"], m)
    assert r["lr"] == mc["phase2"]["lr"] and r["arch"] == Cd.arch_options(mc["phase2"])
    assert abs(r["n_params"] - 20000) / 20000 < 0.15 and all(np.isfinite(h).all() for h in np.asarray(r["hist"]))
    assert len(r["hist"]) == 2


def test_validation_nrmse_is_the_deployed_clean_error(vdp):
    trained = {m: _p2(vdp, m) for m in ("hyper[gen_layers=2]", "latent[latent_depth=3]")}
    for r in trained.values():
        r["cond"].eval()
    out = J.validation_nrmse(vdp["cfg"], vdp["des"], vdp["scales"], vdp["base"], trained, vdp["data"]["val"],
                             DEV, lambda s: None)
    es = vdp["data"]["val"]
    x0 = Ev.initial_guess(vdp["cfg"], len(es), "center", 0, DEV)
    settle = int(0.5 * (es.x.shape[1] - 1))
    obs = Ev.build_observers(Cf.deep_merge(vdp["cfg"], {"baselines": []}), vdp["des"], vdp["scales"], vdp["base"],
                             trained, None, DEV)
    for m in trained:
        with torch.no_grad():
            xh = run(obs[m], es.y, es.u, x0)
        ref = float(np.mean(episode_metrics(es.x, xh, vdp["scales"]["x_std"], settle)["nrmse"]))
        assert out[m]["val_n"] == len(es) and np.isfinite(out[m]["val_nrmse"])
        assert abs(out[m]["val_nrmse"] - ref) <= 1e-6 * max(1.0, ref), (m, out[m], ref)


# ---------------------------------------------------------------------------
# the mechanical selection (analysis/select_tuned.py)
# ---------------------------------------------------------------------------

def _table(kind, score, dev=("a", "b")):
    from analysis import select_tuned as ST
    rows = []
    for c in ST.configs(kind):
        per = {s: {"val_nrmse": score(c, s), "val_loss": score(c, s) ** 2, "width": 1, "n_params": 1} for s in dev}
        rows.append({**c, "per_system": per, "complete": True})
    return {kind: rows}


def test_selection_rule():
    from analysis import select_tuned as ST
    assert len(ST.select_nodes()) == 10 and all(len(d["methods"]) == 4 for d in ST.select_nodes().values())
    assert sum(c["default"] for k in ST.DESIGN for c in ST.configs(k)) == 5
    assert ST.method_string("hyper", 0.001, 2) == "hyper[lr=0.001,gen_layers=2]" and \
        ST.method_string("lora", 0.002) == "lora[lr=0.002]"
    for k in ST.DESIGN:                                  # every method string parses, to its own phase-2 config
        for c in ST.configs(k):
            kind, mc = J.method_cfg(Cf.load("vdp"), c["method"])
            assert kind == k and mc["phase2"]["lr"] == c["lr"]
    dev = ("a", "b")
    # clear winner
    s = ST.select(_table("hyper", lambda c, s: 0.1 if c["method"] == "hyper[lr=0.001,gen_layers=2]" else 0.2), dev)
    assert s["hyper"]["selected"] == "hyper[lr=0.001,gen_layers=2]" and not s["hyper"]["tie_to_default"]
    # within 0.5% of the default: the default is kept
    s = ST.select(_table("lora", lambda c, s: 0.2 * (0.996 if c["method"] == "lora[lr=0.001]" else 1.0)), dev)
    assert s["lora"]["selected"] == "lora[lr=0.002]" and s["lora"]["tie_to_default"] and s["lora"]["argmin"] == "lora[lr=0.001]"
    # non-finite counts as +inf (a diverged config is never chosen)
    s = ST.select(_table("latent", lambda c, s: float("nan") if c["lr"] == 0.004 else 0.2 + c["lr"]), dev)
    assert s["latent"]["selected"] == "latent[lr=0.0005]"
    # a missing (config, system) result refuses the whole variant
    t = _table("film", lambda c, s: 0.2)
    del t["film"][3]["per_system"]["b"]
    t["film"][3]["complete"] = False
    assert ST.select(t, dev)["film"]["status"] == "incomplete"


# ---------------------------------------------------------------------------
# the shared context-scale knob phase2.ctx_rms (PROTOCOL.md section 10)
# ---------------------------------------------------------------------------

def test_context_scale_wrapper():
    base, z_std = _base(), torch.ones(5)
    for kind in KINDS:
        m = Cd.build_conditioner(kind, base, 16, 8, 4, z_std)
        _randomise(m, s=0.1)
        w = Cd.ContextScale(m, 2.5)
        assert n_params(w) == n_params(m) and w.kind == getattr(m, "kind", None)
        z, c = torch.randn(7, 5), torch.randn(7, 16)
        assert torch.equal(w(z, c), m(z, 2.5 * c))
        assert torch.equal(w(z, torch.zeros(7, 16)), m(z, torch.zeros(7, 16)))      # zero-input collapse kept


def test_ctx_rms_key_and_cache():
    cfg = Cf.load("vdp")
    k, mc = J.method_cfg(cfg, "concat[lr=0.008,concat_depth=3,ctx_rms=1]")
    assert mc["phase2"]["ctx_rms"] == 1 and _key(cfg, "concat[ctx_rms=1]") != _key(cfg, "concat")
    assert _key(cfg, "concat[ctx_rms=0.3]") != _key(cfg, "concat[ctx_rms=1]")


@pytest.mark.parametrize("m", ["lora[ctx_rms=1]", "latent[ctx_rms=0.3]"])
def test_phase2_ctx_rms_calibrates_initial_scale(vdp, m):
    r = _p2(vdp, m)
    target = float(m.split("ctx_rms=")[1].rstrip("]"))
    assert isinstance(r["cond"], Cd.ContextScale) and r["ctx_rms"] == target and r["ctx_scale"] > 0
    # re-create the freshly initialised encoder (same seed, same construction order) and check the RMS
    torch.manual_seed(0)
    p, des = vdp["cfg"]["phase2"], vdp["des"]
    from hyperkkl.nets import build_context
    ctx = build_context(p["context"]["kind"], p["context"], 1, des.dt, np.linalg.eigvals(des.A),
                        float(vdp["cfg"]["system"].get("u_scale", 1.0)))
    tr = vdp["data"]["train"]
    with torch.no_grad():
        cc = ctx.seq(tr.u)[:, tr.burn:]
    assert abs(float((r["ctx_scale"] * cc).pow(2).mean().sqrt()) - target) < 1e-3 * target
    assert "ctx_rms" not in _p2(vdp, m.split("[")[0])                 # absent key: no wrapper, nothing recorded
