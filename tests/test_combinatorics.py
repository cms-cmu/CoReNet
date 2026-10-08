"""Candidate enumeration (corenet.models.combinatorics).

The HH -> 4b tree gives the dijet order (0,1),(2,3),(0,2),(1,3),(0,3),(1,2) and the views
{01|23},{02|13},{03|12}; the bbWW tt tree gives the 2 exclusive (t_had, t_lep) assignments.
"""
from __future__ import annotations

import torch

from corenet.models.combinatorics import Node, ReinforceLayer, ResNetBlock, Role, build_hypothesis_table

PAIRS = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]


def slots(assignment, role):
    return tuple(i for r, i in assignment if r == role)


# ------------------------------------------------------------------ candidate tables
def test_hh4b_table():
    roles = {"b": Role("b", 4)}
    nodes = {"H1": Node("H1", ["b", "b"]), "H2": Node("H2", ["b", "b"]), "HH": Node("HH", ["H1", "H2"])}
    T = build_hypothesis_table("HH", nodes, roles)
    assert T.aliases == {"H1": "H1", "H2": "H1"}, T.aliases          # interchangeable siblings share one table
    H = T["H1"]
    assert [slots(c, "b") for c in H.candidates] == PAIRS, [slots(c, "b") for c in H.candidates]
    assert H.child_index.tolist() == [list(p) for p in PAIRS], H.child_index.tolist()
    assert H.symmetric_children          # [b, b]: interchangeable leaves (combinations); sym/antisym inputs are a builder choice
    top = T["HH"]
    assert top.n == 3 and top.children == ["H1", "H1"] and top.symmetric_children
    assert top.child_index.tolist() == [[0, 1], [2, 3], [4, 5]]     # views pair dijets (0,1),(2,3),(4,5)
    assert len(T.views) == 3 and T.view_index.tolist() == [0, 1, 2]


def test_bbww_tables():
    roles = {"b": Role("b", 2), "q": Role("q", 2), "lepton": Role("lepton", 1), "met": Role("met", 1)}
    nodes = {
        "H_bb": Node("H_bb", ["b", "b"]), "W_had": Node("W_had", ["q", "q"]),
        "W_lep": Node("W_lep", ["lepton", "met"]),
        "H_WW": Node("H_WW", ["W_had", "W_lep"]), "HH": Node("HH", ["H_bb", "H_WW"]),
        "t_had": Node("t_had", ["b", "W_had"]), "t_lep": Node("t_lep", ["b", "W_lep"]), "tt": Node("tt", ["t_had", "t_lep"]),
    }
    HH = build_hypothesis_table("HH", nodes, roles)
    assert all(HH[n].n == 1 for n in ("H_bb", "W_had", "W_lep", "H_WW", "HH")) and len(HH.views) == 1
    assert HH["H_bb"].symmetric_children and not HH["H_WW"].symmetric_children
    tt = build_hypothesis_table("tt", nodes, roles)
    assert tt["t_had"].n == 2 and tt["t_lep"].n == 2 and tt["tt"].n == 2   # 2x2 minus same-b -> 2 hypotheses
    for had, lep in tt.views:
        assert slots(had, "b") != slots(lep, "b")
    assert tt["tt"].child_index.shape == (2, 2)


def test_reinforce_shapes_and_symmetry():
    torch.manual_seed(0)
    d, bs = 8, 64
    layer = ReinforceLayer(d, [[0, 1], [2, 3], [4, 5]], phase_symmetric=True, symmetric_pair=True, device="cpu")
    layer.conv.setGhostBatches(0)
    layer.eval()
    x, xx = torch.randn(bs, d, 6), torch.randn(bs, d, 3)
    out = layer(x, xx)
    assert out.shape == (bs, d, 3)
    swapped = x[:, :, [1, 0, 3, 2, 5, 4]]                 # exchange the two children of every parent
    assert torch.allclose(layer(swapped, xx), out, atol=1e-6), "symmetric_pair must be exchange invariant"
    block = ResNetBlock(d, [[0, 1]], phase_symmetric=True, device="cpu", nLayers=2, xx0Update=True)
    block.setGhostBatches(0)
    block.eval()
    y, y0 = block(torch.randn(bs, d, 2), torch.randn(bs, d, 1), torch.randn(bs, d, 2), torch.randn(bs, d, 1))
    assert y.shape == y0.shape == (bs, d, 1)
