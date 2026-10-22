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
    assert (
        "leaf `b`: tensor (400, 5, 4)"
        if topology == "hh4b"
        else "leaf `b`: tensor (400, 5, 2)"
    ) in msgs
    assert (
        validate.main(
            [
                "--config",
                str(cfg),
                "--input",
                str(data),
                "--json",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "cfg,data",
    [
        (HH4B, "hh4b_small.h5"),
        (BBWW, "bbww_small.h5"),
        (repo / "examples/tth/tth.yaml", "tth_small.h5"),
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
    (
        lambda c: c["objects"].__setitem__("b", "jets btag 2 top"),
        "expected '<collection> <column> <n> [lowest]', got 'jets btag 2 top'",
    ),
    (
        lambda c: c["objects"].__setitem__("q", "jets rest 2 lowest"),
        "'rest' takes no 'lowest'",
    ),
    (
        lambda c: c["hypotheses"]["tt"].__setitem__("t_had", ["bb", "W_had"]),
        "child 'bb' is neither",
    ),
    (
        lambda c: c["hypotheses"]["HH"]["H_WW"].__setitem__("W_lep", "lepton + met"),
        "the '+' form was removed; write [lepton, met]",
    ),
    (
        lambda c: c.__setitem__("attention", [{"W_lep": "W_hadronic"}]),
        "'W_hadronic' is not a node",
    ),
    (
        lambda c: c.__setitem__("relations", [["lepton", "neutrino"]]),
        "'neutrino' is neither a node nor a leaf",
    ),
    (
        lambda c: c.__setitem__("relations", [["t_had", "met"]]),
        "t_had has 2 candidates",
    ),
    (
        lambda c: c.__setitem__("combine", ["tt"]),
        "combine must list the main hypothesis 'HH'",
    ),
    (
        lambda c: c.__setitem__("combine", ["HH", "HH"]),
        "combine: duplicate entries ['HH']",
    ),
    (
        lambda c: c["outputs"].__setitem__("ttbar", "ttbar"),
        "outputs.ttbar: 'ttbar' is not a declared hypothesis",
    ),
    (
        lambda c: c["training"].__setitem__("features", "8"),
        "training.features must be an integer",
    ),
    (
        lambda c: c["training"].__setitem__("features", 7),
        "must be divisible by training.attention_heads",
    ),
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
    assert (
        "selected by `dnn_score` lowest first (reads dnn_score, pt for the selection)"
        in msgs
    )
    assert (
        "expects dataset INPUTS/Jets/dnn_score (read for the selection by `b`, not a feature): not found"
        in _errors(r)
    )
    cfg["objects"]["b"] = (
        "jets mass 2 lowest"  # a feature column: no selection-only line
    )
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
    assert (
        "leaf `b`: tensor (400, 6, 2)" in msgs
        and "network built: 3,832 parameters" not in msgs
    )


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
    (
        lambda f: f.move("INPUTS/Lepton", "INPUTS/Electron"),
        "collection `lepton` (used by `lepton`) expects a group INPUTS/lepton or INPUTS/Lepton or INPUTS/Leptons: not found",
    ),
    (
        lambda f: f.__delitem__("INPUTS/Lepton/isE"),
        "expects dataset INPUTS/Lepton/isE: not found",
    ),
    (
        lambda f: _replace(f, "INPUTS/Jets/pt", f["INPUTS/Jets/pt"][:, :10]),
        "INPUTS/Jets/pt: 10 slots, the configuration asks for max 12",
    ),
    (
        lambda f: _replace(f, "INPUTS/Met/pt", f["INPUTS/Met/pt"][:-5]),
        "INPUTS/Met/pt has 395 entries, expected 400",
    ),
    (
        lambda f: _replace(
            f, "INPUTS/Lepton/pt", np.stack([f["INPUTS/Lepton/pt"][...]] * 2, 1)
        ),
        "expected (events,) for a single-object collection",
    ),
    (
        lambda f: _replace(f, "INPUTS/Jets/btag", np.array(["x"] * 400, dtype="S1")),
        "INPUTS/Jets/btag: dtype |S1 is not numeric",
    ),
    (
        lambda f: f.__delitem__("INPUTS/Jets/btag"),
        "expects dataset INPUTS/Jets/btag (read for the selection by `b`, not a feature): not found",
    ),
    (
        lambda f: _replace(f, "INPUTS/Jets/btag", f["INPUTS/Jets/btag"][:, :10]),
        "INPUTS/Jets/btag: 10 slots, the configuration asks for max 12",
    ),
    (
        lambda f: _replace(f, "INPUTS/Jets/MASK", f["INPUTS/Jets/MASK"][:, :11]),
        "INPUTS/Jets/MASK: shape (400, 11), expected (400, 12)",
    ),
    (
        lambda f: _replace(f, "EVENT/label", np.full(400, 3)),
        "EVENT/label: labels [3] outside 0..2",
    ),
    (
        lambda f: _replace(f, "EVENT/label", np.full(400, 1.0)),
        "EVENT/label: shape (400,) dtype float64, expected integer",
    ),
    (
        lambda f: f.__delitem__("EVENT/label"),
        "no labels: expected EVENT/label or CLASSIFICATIONS/EVENT/signal",
    ),
    (
        lambda f: f.__delitem__("EVENT/fold"),
        "EVENT/fold not found: corenet-train needs a fold index",
    ),
    (
        lambda f: _replace(
            f,
            "EVENT/fold",
            np.where(f["EVENT/fold"][...] == 2, 7, f["EVENT/fold"][...]),
        ),
        "EVENT/fold: fold ids [0, 1, 3, 4, 7] are not 0..K-1 (the ids of K folds must be exactly 0..K-1; "
        "--exclude-folds next = fold (k + 1) % K): renumber the column or pass --n-folds K",
    ),
    (
        lambda f: _replace(f, "EVENT/weight", np.full(400, np.nan, dtype=np.float32)),
        "EVENT/weight: 400 non-finite weights",
    ),
    (
        lambda f: _replace(
            f, "INPUTS/Lepton/pt", np.full(400, np.inf, dtype=np.float32)
        ),
        "leaf `lepton`: non-finite features in 400 present objects",
    ),
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


def test_fold_report(tmp_path):
    """The fold source and the events per fold; --n-folds reports the squares(event) % K split instead."""
    from corenet.data.folds import fold_of

    data = write_synthetic(tmp_path / "d.h5", "bbww", n=400)
    r = validate.validate(BBWW, data)
    msgs = "\n".join(m for _, _, m in r.items)
    assert not r.errors and "folds from EVENT/fold: events per fold {0: " in msgs
    assert "--n-folds" not in msgs
    # the synthetic folds are random: the rule disagrees with the column for most events
    r = validate.validate(BBWW, data, n_folds=5)
    msgs = "\n".join(m for _, _, m in r.items)
    assert not r.errors, r.errors
    assert (
        "folds from squares(event) % 5, seed 0 (--n-folds): events per fold {0: "
        in msgs
    )
    assert "EVENT/fold disagrees with squares(event) % 5, seed 0 for " in "\n".join(
        r.warnings
    )
    assert "loader + forward pass" in msgs
    # a column equal to the rule (seed 3)
    with h5py.File(data, "a") as f:
        _replace(f, "EVENT/fold", fold_of(f["EVENT/event"][...], 5, 3))
    r = validate.validate(BBWW, data, n_folds=5, fold_seed=3)
    assert not r.errors and "EVENT/fold equals squares(event) % 5, seed 3" in "\n".join(
        m for _, _, m in r.items
    )
    assert not r.warnings, r.warnings
    assert "disagrees" in "\n".join(validate.validate(BBWW, data, n_folds=5).warnings)
    # the shipped file: its column is the rule with 5 folds and seed 0
    shipped = repo / "examples/data/hh4b_small.h5"
    if shipped.exists():
        r = validate.validate(HH4B, shipped, n_folds=5)
        assert not r.errors and not r.warnings, (r.errors, r.warnings)
        assert "EVENT/fold equals squares(event) % 5, seed 0" in "\n".join(
            m for _, _, m in r.items
        )
    # no column: FAIL by default (see BROKEN_FILES), valid with --n-folds
    with h5py.File(data, "a") as f:
        del f["EVENT/fold"]
    r = validate.validate(BBWW, data)
    assert "EVENT/fold not found" in _errors(r) and "pass --n-folds K" in _errors(r)
    assert (
        validate.main(["--config", str(BBWW), "--input", str(data), "--n-folds", "3"])
        == 0
    )
    r = validate.validate(BBWW, data, n_folds=3)
    msgs = "\n".join(m for _, _, m in r.items)
    with h5py.File(data, "r") as f:
        counts = np.bincount(fold_of(f["EVENT/event"][...], 3), minlength=3).tolist()
    expected = {i: c for i, c in enumerate(counts)}
    assert not r.errors and (
        f"folds from squares(event) % 3, seed 0 (--n-folds): events per fold {expected}"
        in msgs
    )
    # negative event numbers cannot be folded
    with h5py.File(data, "a") as f:
        _replace(f, "EVENT/event", f["EVENT/event"][...] - 10)
    r = validate.validate(BBWW, data, n_folds=3)
    assert (
        "folds = squares(event) % 3, seed 0 (--n-folds): 10 event numbers are negative"
        in _errors(r)
    )
    with h5py.File(data, "a") as f:
        del f["EVENT/event"]
    r = validate.validate(BBWW, data, n_folds=3)
    assert (
        "folds = squares(event) % 3, seed 0 (--n-folds): EVENT/event not found"
        in _errors(r)
    )


def test_fold_ids_not_contiguous_with_n_folds(tmp_path):
    """A column whose ids are not 0..K-1 is a FAIL by itself and a WARN with --n-folds (the column is ignored)."""
    data = write_synthetic(tmp_path / "d.h5", "bbww", n=400)
    with h5py.File(data, "a") as f:
        _replace(f, "EVENT/fold", f["EVENT/fold"][...] + 1)
    r = validate.validate(BBWW, data)
    assert "EVENT/fold: fold ids [1, 2, 3, 4, 5] are not 0..K-1" in _errors(r)
    r = validate.validate(BBWW, data, n_folds=5)
    assert not r.errors
    assert "fold ids [1, 2, 3, 4, 5] are not 0..K-1" in "\n".join(r.warnings)
    assert "--n-folds 5 ignores the column" in "\n".join(r.warnings)
