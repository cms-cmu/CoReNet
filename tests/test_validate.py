"""corenet-validate: valid configurations / files pass, broken ones fail with a specific message."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from synthetic import write_synthetic

from corenet import validate

repo = Path(__file__).resolve().parent.parent
HH4B, BBWW = repo / "examples/hh4b/hh4b.yaml", repo / "examples/bbww/bbww.yaml"


def _write(tmp_path, cfg, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def _errors(report):
    return "\n".join(report.errors)


# ----------------------------------------------------------------------------------- valid cases
@pytest.mark.parametrize("topology", ["hh4b", "bbww"])
def test_valid_synthetic(tmp_path, topology):
    cfg = HH4B if topology == "hh4b" else BBWW
    data = write_synthetic(tmp_path / "d.h5", topology, n=400)
    r = validate.validate(cfg, data)
    assert not r.errors, r.errors
    msgs = "\n".join(m for _, _, m in r.items)
    assert (
        "network built: " + ("1,508" if topology == "hh4b" else "3,832") + " parameters"
        in msgs
    )
    assert "loader + forward pass" in msgs
    assert "selected by `btag` highest first (reads btag, pt for the selection)" in msgs
    assert "selected by `rest` (reads pt for the selection)" in msgs
    assert "['btag'] read for the selection only (not network inputs)" in msgs
    assert "features ['pt', 'eta', 'phi', 'mass'] (the network inputs)" in msgs
    assert ("leaf `b`: tensor (400, 5, 4)" if topology == "hh4b" else "leaf `b`: tensor (400, 5, 2)") in msgs
    assert validate.main(["--config", str(cfg), "--input", str(data), "--json", str(tmp_path / "r.json")]) == 0


@pytest.mark.parametrize(
    "cfg,data",
    [
        (HH4B, "hh4b_small.h5"),
        (BBWW, "bbww_small.h5"),
        (repo / "examples/hh4b/hh4b_hyperparameters.yaml", "hh4b_small.h5"),
        (repo / "examples/bbww/bbww_hyperparameters.yaml", "bbww_small.h5"),
    ],
)
def test_valid_examples(cfg, data):
    path = repo / "examples/data" / data
    if not path.exists():
        pytest.skip(f"{path} not found")
    r = validate.validate(cfg, path)
    assert not r.errors, r.errors


# ----------------------------------------------------------------------------------- broken configurations
def _bbww():
    return yaml.safe_load(BBWW.read_text())


BROKEN_CONFIGS = [
    (lambda c: c.pop("outputs"), "missing required section 'outputs'"),
    (
        lambda c: c.__setitem__("attentions", []),
        "unknown top-level sections ['attentions']",
    ),
    (
        lambda c: c["objects"]["jets"].__setitem__("maximum", 12),
        "objects.jets: unknown keys ['maximum']",
    ),
    (lambda c: c["objects"]["jets"].__setitem__("max", "12"), "max must be an integer"),
    (lambda c: c["objects"].__setitem__("b", "jet btag 2"), "unknown collection 'jet'"),
    (lambda c: c["objects"].__setitem__("b", "jets btag 2 top"), "expected '<collection> <column> <n> [lowest]', got 'jets btag 2 top'"),
    (lambda c: c["objects"].__setitem__("q", "jets rest 2 lowest"), "'rest' takes no 'lowest'"),
    (lambda c: c["hypotheses"]["tt"].__setitem__("t_had", ["bb", "W_had"]), "child 'bb' is neither"),
    (lambda c: c["hypotheses"]["HH"]["H_WW"].__setitem__("W_lep", "lepton + met"), "the '+' form was removed; write [lepton, met]"),
    (lambda c: c.__setitem__("attention", [{"W_lep": "W_hadronic"}]), "'W_hadronic' is not a node"),
    (lambda c: c.__setitem__("relations", [["lepton", "neutrino"]]), "'neutrino' is neither a node nor a leaf"),
    (lambda c: c.__setitem__("relations", [["t_had", "met"]]), "t_had has 2 candidates"),
    (lambda c: c.__setitem__("combine", ["tt"]), "combine must list the main hypothesis 'HH'"),
    (lambda c: c.__setitem__("combine", ["HH", "HH"]), "combine: duplicate entries ['HH']"),
    (lambda c: c["outputs"].__setitem__("ttbar", "ttbar"), "outputs.ttbar: 'ttbar' is not a declared hypothesis"),
    (lambda c: c["training"].__setitem__("features", "8"), "training.features must be an integer"),
    (lambda c: c["training"].__setitem__("features", 7), "must be divisible by training.attention_heads"),
    (lambda c: c["training"].update(features=9, attention_heads=3), "must be even"),
    (
        lambda c: c["training"].__setitem__("phase_symetric", True),
        "training: unknown keys ['phase_symetric']",
    ),
    (
        lambda c: c["training"].__setitem__("reflect_eta", "no"),
        "training.reflect_eta must be true or false",
    ),
    (
        lambda c: c["training"].__setitem__("head_weights", {"tt": 1.0}),
        "head_weights: 'tt' is not a secondary output",
    ),
    (
        lambda c: c.__setitem__("trainer", {"epoch": 3}),
        "trainer: unknown keys ['epoch']",
    ),
    (
        lambda c: c.__setitem__("trainer", {"lr": "1e-2"}),
        "trainer.lr must be a number > 0",
    ),
    (lambda c: c.__setitem__("event", ["HT", "HT"]), "event: duplicate entries ['HT']"),
]


@pytest.mark.parametrize(
    "edit,message", BROKEN_CONFIGS, ids=[m for _, m in BROKEN_CONFIGS]
)
def test_broken_config(tmp_path, edit, message):
    cfg = _bbww()
    edit(cfg)
    r = validate.validate(_write(tmp_path, cfg))
    assert r.errors and message in _errors(r), _errors(r)


def test_missing_selector_column(tmp_path):
    """A column named in a candidate line must exist in the file (FAIL), whether or not it is a feature."""
    data = write_synthetic(tmp_path / "d.h5", "bbww", n=400)
    cfg = _bbww()
    cfg["objects"]["b"] = "jets dnn_score 2 lowest"
    r = validate.validate(_write(tmp_path, cfg), data)
    msgs = "\n".join(m for _, _, m in r.items)
    assert "selected by `dnn_score` lowest first (reads dnn_score, pt for the selection)" in msgs
    assert "expects dataset INPUTS/Jets/dnn_score (read for the selection by `b`, not a feature): not found" in _errors(r)
    cfg["objects"]["b"] = "jets mass 2 lowest"                   # a feature column: no selection-only line
    r = validate.validate(_write(tmp_path, cfg), data)
    assert not r.errors, r.errors
    assert "read for the selection only" not in "\n".join(m for _, _, m in r.items)


def test_btag_feature_is_an_input(tmp_path):
    """btag listed under features: accepted, reported as a network input, one more channel."""
    cfg = _bbww()
    cfg["objects"]["jets"]["features"] = ["pt", "eta", "phi", "mass", "btag"]
    data = write_synthetic(tmp_path / "d.h5", "bbww", n=400)
    r = validate.validate(_write(tmp_path, cfg), data)
    assert not r.errors, r.errors
    msgs = "\n".join(m for _, _, m in r.items)
    assert "features ['pt', 'eta', 'phi', 'mass', 'btag'] (the network inputs)" in msgs
    assert "read for the selection only" not in msgs
    assert "leaf `b`: tensor (400, 6, 2)" in msgs and "network built: 3,832 parameters" not in msgs


def test_yaml_syntax_error(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("objects: {jets: [\n")
    r = validate.validate(p)
    assert "cannot read" in _errors(r)
    assert validate.main(["--config", str(p)]) == 1


# ----------------------------------------------------------------------------------- broken files
def _edit(path, fn):
    with h5py.File(path, "a") as f:
        fn(f)
    return path


def _replace(f, name, data):
    del f[name]
    f.create_dataset(name, data=data)


BROKEN_FILES = [
    (lambda f: f.move("INPUTS/Lepton", "INPUTS/Electron"), "collection `lepton` (used by `lepton`) expects a group INPUTS/lepton or INPUTS/Lepton or INPUTS/Leptons: not found"),
    (lambda f: f.__delitem__("INPUTS/Lepton/isE"), "expects dataset INPUTS/Lepton/isE: not found"),
    (lambda f: _replace(f, "INPUTS/Jets/pt", f["INPUTS/Jets/pt"][:, :10]), "INPUTS/Jets/pt: 10 slots, the configuration asks for max 12"),
    (lambda f: _replace(f, "INPUTS/Met/pt", f["INPUTS/Met/pt"][:-5]), "INPUTS/Met/pt has 395 entries, expected 400"),
    (lambda f: _replace(f, "INPUTS/Lepton/pt", np.stack([f["INPUTS/Lepton/pt"][...]] * 2, 1)), "expected (events,) for a single-object collection"),
    (lambda f: _replace(f, "INPUTS/Jets/btag", np.array(["x"] * 400, dtype="S1")), "INPUTS/Jets/btag: dtype |S1 is not numeric"),
    (lambda f: f.__delitem__("INPUTS/Jets/btag"), "expects dataset INPUTS/Jets/btag (read for the selection by `b`, not a feature): not found"),
    (lambda f: _replace(f, "INPUTS/Jets/btag", f["INPUTS/Jets/btag"][:, :10]), "INPUTS/Jets/btag: 10 slots, the configuration asks for max 12"),
    (lambda f: _replace(f, "INPUTS/Jets/MASK", f["INPUTS/Jets/MASK"][:, :11]), "INPUTS/Jets/MASK: shape (400, 11), expected (400, 12)"),
    (lambda f: _replace(f, "EVENT/label", np.full(400, 3)), "EVENT/label: labels [3] outside 0..2"),
    (lambda f: _replace(f, "EVENT/label", np.full(400, 1.0)), "EVENT/label: shape (400,) dtype float64, expected integer"),
    (lambda f: f.__delitem__("EVENT/label"), "no labels: expected EVENT/label or CLASSIFICATIONS/EVENT/signal"),
    (lambda f: f.__delitem__("EVENT/fold"), "EVENT/fold not found: corenet-train needs a fold index"),
    (lambda f: _replace(f, "EVENT/weight", np.full(400, np.nan, dtype=np.float32)), "EVENT/weight: 400 non-finite weights"),
    (lambda f: _replace(f, "INPUTS/Lepton/pt", np.full(400, np.inf, dtype=np.float32)), "leaf `lepton`: non-finite features in 400 present objects"),
]


@pytest.mark.parametrize(
    "edit,message", BROKEN_FILES, ids=[m[:60] for _, m in BROKEN_FILES]
)
def test_broken_file(tmp_path, edit, message):
    data = _edit(write_synthetic(tmp_path / "d.h5", "bbww", n=400), edit)
    r = validate.validate(BBWW, data)
    assert r.errors and message in _errors(r), _errors(r)
    assert validate.main(["--config", str(BBWW), "--input", str(data)]) == 1


def test_missing_event_scalar_and_warnings(tmp_path):
    cfg = _bbww()
    cfg["event"] = ["nSelJets", "HT", "MET_significance"]
    data = _edit(
        write_synthetic(tmp_path / "d.h5", "bbww", n=400),
        lambda f: (f.__delitem__("EVENT/weight"), f.__delitem__("INPUTS/Jets/MASK")),
    )
    r = validate.validate(_write(tmp_path, cfg), data)
    assert (
        "event scalar `MET_significance` expects dataset EVENT/MET_significance: not found"
        in _errors(r)
    )
    w = "\n".join(r.warnings)
    assert "EVENT/weight not found" in w and "INPUTS/Jets/MASK not found" in w
