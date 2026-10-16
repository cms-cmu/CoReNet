"""corenet.data.H5Dataset on the example files (examples/data/*_small.h5): tensor shapes, feature
order, presence flags and padding, the role selectors, the computed event scalars and the labels."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from corenet.data import H5Dataset
from corenet.models.config import load_config, parse_config

repo = Path(__file__).resolve().parent.parent
DATA = repo / "examples" / "data"
N_CHECK = 2000  # events checked one by one against a direct reading of the file


def _need(path):
    if not path.exists():
        pytest.skip(f"{path} not found")
    return path


def _check_flags(x):
    for k, v in x.items():
        if k == "event":
            continue
        assert set(np.unique(v[:, -1].numpy())) <= {-1.0, 1.0}, (
            f"{k}: presence flag is not +-1"
        )
        pad = v[:, -1] < 0
        assert (v.transpose(1, 2)[pad] == -1).all(), (
            f"{k}: padded slots are not -1 everywhere"
        )


def test_bbww_small():
    path = _need(DATA / "bbww_small.h5")
    spec = load_config(repo / "examples/bbww/bbww.yaml")
    ds = H5Dataset(path, spec)
    x, n = ds.inputs, len(ds)
    assert {k: tuple(v.shape) for k, v in x.items()} == {
        "b": (n, 6, 2),
        "q": (n, 6, 2),
        "lepton": (n, 7, 1),
        "met": (n, 3, 1),
        "event": (n, 2),
    }
    assert all(v.dtype == torch.float32 for v in x.values())
    _check_flags(x)
    assert (x["lepton"][:, -1] == 1).all() and (x["met"][:, -1] == 1).all(), (
        "max-1 collections are always present"
    )
    with h5py.File(path, "r") as f:
        mask = f["INPUTS/Jets/MASK"][...]
        jet = {
            k: f[f"INPUTS/Jets/{k}"][...] for k in ("pt", "eta", "phi", "mass", "btag")
        }
        lep = {
            k: f[f"INPUTS/Lepton/{k}"][...]
            for k in ("pt", "eta", "phi", "mass", "isE", "isM")
        }
        label, weight, fold = (
            f["EVENT/label"][...],
            f["EVENT/weight"][...],
            f["EVENT/fold"][...],
        )
    assert (x["event"][:, 0].numpy() == mask.sum(1)).all()  # nSelJets
    assert np.allclose(
        x["event"][:, 1].numpy(), (jet["pt"] * mask).sum(1), rtol=1e-5
    )  # HT
    for i, k in enumerate(("pt", "eta", "phi", "mass", "isE", "isM")):
        assert np.array_equal(
            x["lepton"][:, i, 0].numpy(), lep[k].astype(np.float32)
        ), k
    for i in range(N_CHECK):
        present = np.flatnonzero(mask[i])
        b_rank = sorted(present, key=lambda j: (-jet["btag"][i, j], -jet["pt"][i, j]))[
            :2
        ]
        rest = sorted(set(present) - set(b_rank), key=lambda j: -jet["pt"][i, j])[:2]
        assert np.array_equal(
            x["b"][i, 0, : len(b_rank)].numpy(), jet["pt"][i, b_rank]
        ), i
        assert np.array_equal(
            x["b"][i, 4, : len(b_rank)].numpy(), jet["btag"][i, b_rank]
        ), i
        assert np.array_equal(x["q"][i, 0, : len(rest)].numpy(), jet["pt"][i, rest]), i
    assert (ds.labels.numpy() == label).all() and set(np.unique(label)) == {1, 2}
    assert np.array_equal(ds.weights.numpy(), weight) and np.array_equal(
        ds.folds.numpy(), fold
    )
    tr, va = ds.fold_indices(0)
    assert len(tr) + len(va) == n and (ds.folds.numpy()[va] == 0).all()


def test_hh4b_small():
    path = _need(DATA / "hh4b_small.h5")
    spec = load_config(repo / "examples/hh4b/hh4b.yaml")
    with h5py.File(path, "r") as f:
        mask = f["INPUTS/Source/MASK"][...]
        jet = {
            k: f[f"INPUTS/Source/{k}"][...]
            for k in ("pt", "eta", "phi", "mass", "btag")
        }
        label = f["CLASSIFICATIONS/EVENT/signal"][...]
    for rest_order in ("pt", "selector"):
        ds = H5Dataset(path, spec, rest_order=rest_order)
        x, n = ds.inputs, len(ds)
        assert (
            x["b"].shape == (n, 6, 4)
            and x["other"].shape == (n, 6, 8)
            and x["event"].shape == (n, 2)
        )
        _check_flags(x)
        for i in range(N_CHECK):
            present = np.flatnonzero(mask[i])
            order = sorted(
                present, key=lambda j: (-jet["btag"][i, j], -jet["pt"][i, j])
            )
            b = order[:4]
            rest = (
                order[4:]
                if rest_order == "selector"
                else sorted(order[4:], key=lambda j: -jet["pt"][i, j])
            )
            assert np.array_equal(x["b"][i, 0, : len(b)].numpy(), jet["pt"][i, b]), i
            assert np.array_equal(
                x["other"][i, 0, : len(rest)].numpy(), jet["pt"][i, rest[:8]]
            ), i
            assert int((x["other"][i, -1] > 0).sum()) == min(8, len(rest))
        assert np.array_equal(
            ds.labels.numpy(), label
        )  # no EVENT/label: CLASSIFICATIONS/EVENT/signal


def test_errors(tmp_path):
    path = _need(DATA / "bbww_small.h5")
    cfg = {
        "objects": {
            "jets": {"features": ["pt", "eta", "phi", "mass", "btag"], "max": 12},
            "b": "jets btag 2",
            "photon": {"features": ["pt", "eta", "phi"]},
        },
        "hypotheses": {"HH": {"H_bb": ["b", "b"], "X": ["b", "photon"]}},
        "outputs": {"signal": "HH"},
        "combine": ["HH"],
    }
    with pytest.raises(KeyError, match="photon"):
        H5Dataset(path, parse_config(cfg))
    cfg = {
        "objects": {
            "jets": {"features": ["pt", "eta", "phi", "mass", "btag"], "max": 12},
            "b": "jets btag 4",
        },
        "hypotheses": {"HH": {"H1": ["b", "b"], "H2": ["b", "b"]}},
        "outputs": {"signal": "HH"},
        "combine": ["HH"],
    }
    with pytest.raises(
        ValueError, match="labels"
    ):  # bbWW labels 1, 2 with a single output
        H5Dataset(path, parse_config(cfg))
