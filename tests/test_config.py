"""Validation rules of corenet.models.config (docs/EventInfo.md)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from corenet.models.config import ConfigError, load_config, parse_config
from corenet.models.network import CoReNet

TTH = Path(__file__).resolve().parent.parent / "examples/tth/tth.yaml"

HH4B = {
    "objects": {
        "jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12},
        "b": "jets btag 4",
        "other": "jets rest 8",
    },
    "event": ["nSelJets", "HT"],
    "hypotheses": {"HH": {"H1": ["b", "b"], "H2": ["b", "b"], "context": "other"}},
    "outputs": {"signal": "HH"},
    "combine": ["HH"],
}
BBWW = {
    "objects": {
        "jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12},
        "b": "jets btag 2",
        "q": "jets rest 2",
        "lepton": {"features": ["pt", "eta", "phi", "mass", "isE", "isM"]},
        "met": {"features": ["pt", "phi"]},
    },
    "event": ["nSelJets", "HT"],
    "hypotheses": {
        "HH": {
            "H_bb": ["b", "b"],
            "H_WW": {"W_had": ["q", "q"], "W_lep": ["lepton", "met"]},
        },
        "tt": {"t_had": ["b", "W_had"], "t_lep": ["b", "W_lep"]},
    },
    "attention": [{"W_lep": "W_had"}, {"t_had": "t_lep"}],
    "relations": [["lepton", "met"], ["H_bb", "met"], ["lepton", "W_had"]],
    "outputs": {"signal": "HH", "ttbar": "tt"},
    "combine": ["HH", "tt"],
}


def _fails(cfg, fragment):
    try:
        parse_config(cfg)
    except ConfigError as e:
        assert fragment in str(e), f"expected error containing {fragment!r}, got {e}"
        return
    raise AssertionError(f"expected ConfigError containing {fragment!r}")


def test_hh4b_parses():
    s = parse_config(HH4B)
    assert (
        s.roles["b"].selector == "btag"
        and s.roles["b"].n == 4
        and s.roles["other"].selector == "rest"
    )
    assert set(s.nodes) == {"H1", "H2", "HH"} and s.nodes["HH"].children == ["H1", "H2"]
    assert s.hypotheses["HH"].context == "other" and s.classes == ["rest", "signal"]
    assert (
        s.training.features == 8
        and s.training.attention_heads == 2
        and s.training.phase_symmetric
    )
    assert s.leaves()["b"].n == 4 and s.nodes_in("HH") == {"H1", "H2", "HH"}


def test_selector_columns_are_not_features():
    """A btag selector needs no btag feature: the selector reads its column from the file on its own."""
    s = parse_config(HH4B)
    assert s.collections["jets"].features == ["pt", "eta", "phi", "mass"]
    assert s.selector_columns("jets") == ["btag", "pt"]
    assert s.columns_to_read("jets") == ["pt", "eta", "phi", "mass", "btag"]
    # btag listed as a feature is accepted and stays a feature (an ordinary extra input)
    c = copy.deepcopy(HH4B)
    c["objects"]["jets"]["features"] = ["pt", "eta", "phi", "mass", "btag"]
    s = parse_config(c)
    assert s.collections["jets"].features == ["pt", "eta", "phi", "mass", "btag"]
    assert s.columns_to_read("jets") == ["pt", "eta", "phi", "mass", "btag"]
    # a pt selector reads pt, which a max > 1 collection always has as a feature
    c = copy.deepcopy(HH4B)
    c["objects"]["b"] = "jets pt 4"
    assert parse_config(c).columns_to_read("jets") == ["pt", "eta", "phi", "mass"]


def test_bbww_parses():
    s = parse_config(BBWW)
    assert s.nodes["W_lep"].children == [
        "lepton",
        "met",
    ]  # two single-object leaves, like any [A, B]
    assert s.nodes["t_had"].children == ["b", "W_had"]  # W_had reused from HH by name
    assert (
        set(s.leaves()) == {"b", "q", "lepton", "met"} and s.leaves()["lepton"].n == 1
    )
    assert [a.query + ">" + a.keys for a in s.attention] == [
        "W_lep>W_had",
        "t_had>t_lep",
    ]
    assert [(r.a, r.b) for r in s.relations] == [
        ("lepton", "met"),
        ("H_bb", "met"),
        ("lepton", "W_had"),
    ]
    assert s.classes == ["rest", "signal", "ttbar"] and s.combine == ["HH", "tt"]
    assert s.nodes_in("tt") == {"t_had", "t_lep", "W_had", "W_lep", "tt"}


def test_role_grammar():
    """<collection> <column> <n> [lowest]: any column of the file, highest first unless `lowest`; `rest` reserved."""
    c = copy.deepcopy(HH4B)
    c["objects"]["b"] = "jets mass 4"
    s = parse_config(c)
    r = s.roles["b"]
    assert (r.selector, r.n, r.ascending, r.columns) == (
        "mass",
        4,
        False,
        ("mass", "pt"),
    )
    assert s.columns_to_read("jets") == [
        "pt",
        "eta",
        "phi",
        "mass",
    ]  # mass is a feature already
    c = copy.deepcopy(HH4B)
    c["objects"]["b"] = "jets mass 4 lowest"
    r = parse_config(c).roles["b"]
    assert (r.selector, r.ascending, r.columns) == ("mass", True, ("mass", "pt"))
    c = copy.deepcopy(HH4B)
    c["objects"]["b"] = "jets  dnn_score   4   lowest "
    r = parse_config(c).roles["b"]
    assert (r.selector, r.ascending) == ("dnn_score", True) and parse_config(
        c
    ).columns_to_read("jets")[-1] == "dnn_score"
    c = copy.deepcopy(HH4B)
    c["objects"]["b"] = "jets pt 4"
    r = parse_config(c).roles["b"]
    assert (r.selector, r.ascending, r.columns) == ("pt", False, ("pt",))
    assert parse_config(HH4B).roles["other"].columns == ("pt",) and parse_config(
        HH4B
    ).roles["b"].columns == ("btag", "pt")
    for bad, msg in (
        ("jets btag 4 highest", "expected '<collection> <column> <n> [lowest]'"),
        ("jets btag 4 Lowest", "expected '<collection> <column> <n> [lowest]'"),
        ("jets btag 4 lowest extra", "expected '<collection> <column> <n> [lowest]'"),
        ("jets rest 8 lowest", "'rest' takes no 'lowest'"),
        ("jets btag", "expected '<collection> <column> <n> [lowest]'"),
        ("jets btag four", "expected '<collection> <column> <n> [lowest]'"),
    ):
        c = copy.deepcopy(HH4B)
        c["objects"]["b"] = bad
        _fails(c, msg)


def test_validation_rules():
    c = copy.deepcopy(HH4B)
    c["combine"] = []
    _fails(c, "combine must list the main hypothesis")
    c = copy.deepcopy(HH4B)
    c["objects"]["b"] = "jets btag 13"
    _fails(c, "within 1..max")
    c = copy.deepcopy(HH4B)
    c["objects"]["b"] = "jets btag 4 highest"
    _fails(c, "expected '<collection> <column> <n> [lowest]'")
    c = copy.deepcopy(HH4B)
    c["objects"]["jets"]["features"] = ["eta", "phi", "mass"]
    _fails(c, "need pt, eta, phi features")
    c = copy.deepcopy(HH4B)
    c["hypotheses"]["HH"]["context"] = "nothing"
    _fails(c, "is not a declared role")
    c = copy.deepcopy(HH4B)
    c["hypotheses"]["HH"]["H1"] = ["b", "c"]
    _fails(c, "neither a role")
    c = copy.deepcopy(HH4B)
    c["hypotheses"]["ZZ"] = {"Z1": ["b", "b"], "Z2": ["b", "b"]}
    _fails(c, "used by no output")
    c = copy.deepcopy(HH4B)
    c["outputs"] = {"signal": "XX"}
    _fails(c, "is not a declared hypothesis")
    c = copy.deepcopy(HH4B)
    c["outputs"] = {"signal": None}
    _fails(c, "main score) needs a hypothesis")
    c = copy.deepcopy(HH4B)
    c["training"] = {"features": 8, "attention_heads": 3}
    _fails(c, "divisible")
    c = copy.deepcopy(HH4B)
    c["training"] = {"foo": 1}
    _fails(c, "unknown keys")
    c = copy.deepcopy(BBWW)
    c["hypotheses"]["HH"]["H_WW"]["W_lep"] = "lepton + met"
    _fails(c, "the '+' form was removed; write [lepton, met]")
    c = copy.deepcopy(BBWW)
    c["hypotheses"]["HH"]["H_WW"]["W_lep"] = "lepton met"
    _fails(c, "cannot interpret")
    c = copy.deepcopy(BBWW)
    c["hypotheses"]["tt"]["W_had"] = ["q", "b"]
    _fails(c, "defined twice with different children")
    c = copy.deepcopy(BBWW)
    c["attention"] = [{"W_lep": "nowhere"}]
    _fails(c, "is not a node")
    c = copy.deepcopy(BBWW)
    c["objects"]["q"] = "jets rest 11"
    _fails(c, "slots > max")
    c = copy.deepcopy(HH4B)
    del c["combine"]
    _fails(c, "missing required section")
    c = copy.deepcopy(HH4B)
    c["relation"] = []
    _fails(c, "unknown top-level sections")
    c = copy.deepcopy(HH4B)
    c["trainer"] = {"epochs": 3}
    parse_config(c)  # read by corenet-train only


def test_relations_and_single_object_children():
    s = parse_config(HH4B)
    assert s.relations == []  # optional; absent for 4b
    c = copy.deepcopy(BBWW)
    c["relations"] = [["lepton", "nothing"]]
    _fails(c, "neither a node nor a leaf")
    c = copy.deepcopy(BBWW)
    c["relations"] = [["met", "met"]]
    _fails(c, "two different names")
    c = copy.deepcopy(BBWW)
    c["relations"] = ["lepton met"]
    _fails(c, "pairs '[A, B]'")
    # a role taken from a multi-object lepton collection is a leaf like any other: [l, met] parses as [lepton, met]
    c = copy.deepcopy(BBWW)
    c["objects"]["lepton"] = {
        "features": ["pt", "eta", "phi", "mass", "isE", "isM"],
        "max": 2,
    }
    c["objects"]["l"] = "lepton pt 1"
    c["hypotheses"]["HH"]["H_WW"]["W_lep"] = ["l", "met"]
    c["relations"] = [["l", "met"]]
    s = parse_config(c)
    assert (
        s.nodes["W_lep"].children == ["l", "met"]
        and s.leaves()["l"].n == 1
        and s.leaves()["met"].n == 1
    )
    c = copy.deepcopy(BBWW)
    c["hypotheses"]["HH"]["H_WW"]["W_lep"] = "l + met"
    _fails(c, "the '+' form was removed; write [l, met]")


def test_min_jets():
    """min_jets (optional, default 4) sets the nSelJets offset; rejected unless an integer >= 2."""
    assert parse_config(copy.deepcopy(HH4B)).min_jets == 4
    c = copy.deepcopy(HH4B)
    c["min_jets"] = 2
    assert parse_config(c).min_jets == 2
    for bad in (1, 0, "4", True, 4.0):
        c = copy.deepcopy(HH4B)
        c["min_jets"] = bad
        with pytest.raises(ConfigError):
            parse_config(c)


def test_tth_example_parses():
    """examples/tth/tth.yaml: min_jets 6, hypotheses ttH and WW, 12 ttH views, 3,543 parameters."""
    s = load_config(TTH)
    assert s.min_jets == 6
    assert set(s.hypotheses) == {"ttH", "WW"} and s.combine == ["ttH", "WW"]
    model = CoReNet(s, device="cpu")
    assert model.tables["ttH"].nodes["ttH"].n == 12
    n_par = sum(p.numel() for p in model.parameters())
    assert n_par == 3543, n_par


def test_labels():
    """``labels:`` (optional): the classes in label order; every output must be a label; main_class is
    the class of the first output; without it the classes are rest + outputs and main_class is 1."""
    s = parse_config(copy.deepcopy(BBWW))
    assert s.classes == ["rest", "signal", "ttbar"] and s.main_class == 1
    c = copy.deepcopy(BBWW)
    c["labels"] = ["ttbar", "multijet", "signal"]
    s = parse_config(c)
    assert s.classes == ["ttbar", "multijet", "signal"] and s.main_class == 2
    model = CoReNet(s, device="cpu")
    assert model.nC == 3 and set(model.aux_heads) == {"ttbar"}
    c = copy.deepcopy(HH4B)
    c["labels"] = ["d4", "d3", "t4", "t3"]
    c["outputs"] = {"d4": "HH"}
    s = parse_config(c)
    assert s.classes == ["d4", "d3", "t4", "t3"] and s.main_class == 0
    model = CoReNet(s, device="cpu")
    assert model.nC == 4 and len(model.aux_heads) == 0


def test_labels_rejected():
    c = copy.deepcopy(BBWW)
    c["labels"] = ["multijet", "signal"]  # ttbar is an output but not a label
    _fails(c, "outputs.ttbar: the output name must be one of the labels")
    c = copy.deepcopy(HH4B)
    c["labels"] = ["signal"]
    _fails(c, "labels: at least two classes are needed")
    c = copy.deepcopy(HH4B)
    c["labels"] = ["signal", "rest", "signal"]
    _fails(c, "labels: duplicate entries ['signal']")
    c = copy.deepcopy(HH4B)
    c["labels"] = ["signal", 3]
    _fails(c, "labels: expected a list of names")
