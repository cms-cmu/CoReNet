#!/usr/bin/env python3
"""Validation rules of corenet.models.config (docs/EventInfo.md)."""
from __future__ import annotations

import copy

from corenet.models.config import ConfigError, parse_config

HH4B = {
    "objects": {"jets": {"features": ["pt", "eta", "phi", "mass", "btag"], "max": 12}, "b": "jets btag 4", "other": "jets rest 8"},
    "event": ["nSelJets", "HT"],
    "hypotheses": {"HH": {"H1": ["b", "b"], "H2": ["b", "b"], "context": "other"}},
    "outputs": {"signal": "HH"},
    "combine": ["HH"],
}
BBWW = {
    "objects": {"jets": {"features": ["pt", "eta", "phi", "mass", "btag"], "max": 12}, "b": "jets btag 2", "q": "jets rest 2",
                "lepton": {"features": ["pt", "eta", "phi", "mass", "isE", "isM"]}, "met": {"features": ["pt", "phi"]}},
    "event": ["nSelJets", "HT"],
    "hypotheses": {"HH": {"H_bb": ["b", "b"], "H_WW": {"W_had": ["q", "q"], "W_lep": ["lepton", "met"]}},
                   "tt": {"t_had": ["b", "W_had"], "t_lep": ["b", "W_lep"]}},
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
    assert s.roles["b"].selector == "btag" and s.roles["b"].n == 4 and s.roles["other"].selector == "rest"
    assert set(s.nodes) == {"H1", "H2", "HH"} and s.nodes["HH"].children == ["H1", "H2"]
    assert s.hypotheses["HH"].context == "other" and s.classes == ["rest", "signal"]
    assert s.training.features == 8 and s.training.attention_heads == 2 and s.training.phase_symmetric
    assert s.leaves()["b"].n == 4 and s.nodes_in("HH") == {"H1", "H2", "HH"}


def test_bbww_parses():
    s = parse_config(BBWW)
    assert s.nodes["W_lep"].children == ["lepton", "met"]        # two single-object leaves, like any [A, B]
    assert s.nodes["t_had"].children == ["b", "W_had"]           # W_had reused from HH by name
    assert set(s.leaves()) == {"b", "q", "lepton", "met"} and s.leaves()["lepton"].n == 1
    assert [a.query + ">" + a.keys for a in s.attention] == ["W_lep>W_had", "t_had>t_lep"]
    assert [(r.a, r.b) for r in s.relations] == [("lepton", "met"), ("H_bb", "met"), ("lepton", "W_had")]
    assert s.classes == ["rest", "signal", "ttbar"] and s.combine == ["HH", "tt"]
    assert s.nodes_in("tt") == {"t_had", "t_lep", "W_had", "W_lep", "tt"}


def test_validation_rules():
    c = copy.deepcopy(HH4B); c["combine"] = []; _fails(c, "combine must list the main hypothesis")
    c = copy.deepcopy(HH4B); c["objects"]["b"] = "jets btag 13"; _fails(c, "within 1..max")
    c = copy.deepcopy(HH4B); c["objects"]["b"] = "jets weird 4"; _fails(c, "selector must be one of")
    c = copy.deepcopy(HH4B); c["objects"]["jets"]["features"] = ["pt", "eta", "phi", "mass"]; _fails(c, "needs a 'btag' feature")
    c = copy.deepcopy(HH4B); c["hypotheses"]["HH"]["context"] = "nothing"; _fails(c, "is not a declared role")
    c = copy.deepcopy(HH4B); c["hypotheses"]["HH"]["H1"] = ["b", "c"]; _fails(c, "neither a role")
    c = copy.deepcopy(HH4B); c["hypotheses"]["ZZ"] = {"Z1": ["b", "b"], "Z2": ["b", "b"]}; _fails(c, "used by no output")
    c = copy.deepcopy(HH4B); c["outputs"] = {"signal": "XX"}; _fails(c, "is not a declared hypothesis")
    c = copy.deepcopy(HH4B); c["outputs"] = {"signal": None}; _fails(c, "main score) needs a hypothesis")
    c = copy.deepcopy(HH4B); c["training"] = {"features": 8, "attention_heads": 3}; _fails(c, "divisible")
    c = copy.deepcopy(HH4B); c["training"] = {"foo": 1}; _fails(c, "unknown keys")
    c = copy.deepcopy(BBWW); c["hypotheses"]["HH"]["H_WW"]["W_lep"] = "lepton + met"; _fails(c, "the '+' form was removed; write [lepton, met]")
    c = copy.deepcopy(BBWW); c["hypotheses"]["HH"]["H_WW"]["W_lep"] = "lepton met"; _fails(c, "cannot interpret")
    c = copy.deepcopy(BBWW); c["hypotheses"]["tt"]["W_had"] = ["q", "b"]; _fails(c, "defined twice with different children")
    c = copy.deepcopy(BBWW); c["attention"] = [{"W_lep": "nowhere"}]; _fails(c, "is not a node")
    c = copy.deepcopy(BBWW); c["objects"]["q"] = "jets rest 11"; _fails(c, "slots > max")
    c = copy.deepcopy(HH4B); del c["combine"]; _fails(c, "missing required section")
    c = copy.deepcopy(HH4B); c["relation"] = []; _fails(c, "unknown top-level sections")
    c = copy.deepcopy(HH4B); c["trainer"] = {"epochs": 3}; parse_config(c)       # read by corenet-train only


def test_relations_and_single_object_children():
    s = parse_config(HH4B)
    assert s.relations == []                                                    # optional; absent for 4b
    c = copy.deepcopy(BBWW); c["relations"] = [["lepton", "nothing"]]; _fails(c, "neither a node nor a leaf")
    c = copy.deepcopy(BBWW); c["relations"] = [["met", "met"]]; _fails(c, "two different names")
    c = copy.deepcopy(BBWW); c["relations"] = ["lepton met"]; _fails(c, "pairs '[A, B]'")
    # a role taken from a multi-object lepton collection is a leaf like any other: [l, met] parses as [lepton, met]
    c = copy.deepcopy(BBWW)
    c["objects"]["lepton"] = {"features": ["pt", "eta", "phi", "mass", "isE", "isM"], "max": 2}
    c["objects"]["l"] = "lepton pt 1"
    c["hypotheses"]["HH"]["H_WW"]["W_lep"] = ["l", "met"]
    c["relations"] = [["l", "met"]]
    s = parse_config(c)
    assert s.nodes["W_lep"].children == ["l", "met"] and s.leaves()["l"].n == 1 and s.leaves()["met"].n == 1
    c = copy.deepcopy(BBWW); c["hypotheses"]["HH"]["H_WW"]["W_lep"] = "l + met"; _fails(c, "the '+' form was removed; write [l, met]")
