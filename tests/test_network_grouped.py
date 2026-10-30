"""Grouped leaf embedding: all child slots of one leaf role in one node are embedded in a single call
(candidate-major, slot-minor), so in training mode they share the ghost-batch statistics. In eval mode
(running statistics, 0 ghost batches) the embedded pixel of a slot must not depend on which other slots
are in the call: the grouped call, split back per slot, equals the per-slot call exactly. This is why
stored checkpoints evaluate as before the grouping."""

from __future__ import annotations

from pathlib import Path

import torch
from synthetic import write_synthetic

from corenet.data.h5 import H5Dataset
from corenet.models.config import load_config
from corenet.models.network import CoReNet

repo = Path(__file__).resolve().parent.parent


def _model_and_inputs(tmp_path: Path, topology: str):
    data = write_synthetic(tmp_path / f"{topology}.h5", topology, n=400)
    torch.manual_seed(0)
    model = CoReNet(load_config(repo / f"examples/{topology}/{topology}.yaml"), "cpu")
    ds = H5Dataset(data, model.spec)
    model.train()
    model.updateMeanStd(ds.inputs)
    model.initMeanStd()
    model.eval()
    model.setGhostBatches(0)
    return model, ds.inputs


def _check_grouping_invariance(model: CoReNet, inputs) -> int:
    checked = 0
    with torch.no_grad():
        prep = model._prepare(inputs)
        n = prep["n"]
        a = model.event_embed(prep["event"]) if model.event_embed is not None else None
        for h, T in model.tables.items():
            hp = prep["hyp"][h]
            for node in model.levels[h]:
                nt = T.nodes[node]
                for role, js in model._leaf_groups(nt).items():
                    feats, pres = model._grouped_slots(hp, node, js)
                    assert feats.shape[2] == nt.n * len(js)
                    grouped = model.leaf_embed[role](feats, add=a)
                    grouped = grouped.view(n, model.d, nt.n, len(js))
                    for i, j in enumerate(js):
                        alone = model.leaf_embed[role](hp["slot_feats"][node][j], add=a)
                        assert torch.equal(grouped[:, :, :, i], alone), (
                            h,
                            node,
                            role,
                            j,
                        )
                        # the presence is laid out the same way
                        assert torch.equal(
                            pres.view(n, nt.n, len(js))[:, :, i],
                            hp["slot_present"][node][j],
                        )
                        checked += 1
    return checked


def test_grouped_embedding_eval_invariance_hh4b(tmp_path):
    model, inputs = _model_and_inputs(tmp_path, "hh4b")
    nt = model.tables["HH"].nodes[model.levels["HH"][0]]
    assert model._leaf_groups(nt) == {"b": [0, 1]}  # a dijet: both slots in one call
    assert _check_grouping_invariance(model, inputs) >= 2


def test_grouped_embedding_eval_invariance_bbww(tmp_path):
    model, inputs = _model_and_inputs(tmp_path, "bbww")
    groups = {
        node: model._leaf_groups(T.nodes[node])
        for h, T in model.tables.items()
        for node in model.levels[h]
    }
    assert groups["W_lep"] == {
        "lepton": [0],
        "met": [1],
    }  # different roles: separate calls
    assert _check_grouping_invariance(model, inputs) > 0
