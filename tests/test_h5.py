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
        "b": (n, 5, 2), "q": (n, 5, 2), "lepton": (n, 7, 1), "met": (n, 3, 1), "event": (n, 2)}
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
        assert np.array_equal(x["b"][i, 0, :len(b_rank)].numpy(), jet["pt"][i, b_rank]), i    # btag-ranked, btag not a channel
        assert np.array_equal(x["b"][i, 3, :len(b_rank)].numpy(), jet["mass"][i, b_rank]), i
        assert np.array_equal(x["q"][i, 0, :len(rest)].numpy(), jet["pt"][i, rest]), i
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
        assert x["b"].shape == (n, 5, 4) and x["other"].shape == (n, 5, 8) and x["event"].shape == (n, 2)
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


def _hh4b_cfg(features):
    return {"objects": {"jets": {"features": features, "max": 12}, "b": "jets btag 4", "other": "jets rest 8"},
            "event": ["nSelJets", "HT"], "hypotheses": {"HH": {"H1": ["b", "b"], "H2": ["b", "b"], "context": "other"}},
            "outputs": {"signal": "HH"}, "combine": ["HH"]}


def test_selector_column_read_without_feature():
    """(a) `b: jets btag 4` with btag absent from features: the loader reads INPUTS/Source/btag on its own,
    the selection is the btag ranking, and the tensor holds the features only."""
    path = _need(DATA / "hh4b_small.h5")
    spec = parse_config(_hh4b_cfg(["pt", "eta", "phi", "mass"]))
    ds = H5Dataset(path, spec)
    x, n = ds.inputs, len(ds)
    assert x["b"].shape == (n, 5, 4) and x["other"].shape == (n, 5, 8)
    with h5py.File(path, "r") as f:
        mask = f["INPUTS/Source/MASK"][...]
        jet = {k: f[f"INPUTS/Source/{k}"][...] for k in ("pt", "btag")}
    for i in range(N_CHECK):
        present = np.flatnonzero(mask[i])
        b = sorted(present, key=lambda j: (-jet["btag"][i, j], -jet["pt"][i, j]))[:4]
        assert np.array_equal(x["b"][i, 0, :len(b)].numpy(), jet["pt"][i, b]), i
    # the same selection with btag as a pt-only selector would differ: the file is really read for btag
    cfg_pt = _hh4b_cfg(["pt", "eta", "phi", "mass"]); cfg_pt["objects"]["b"] = "jets pt 4"
    assert not torch.equal(H5Dataset(path, parse_config(cfg_pt)).inputs["b"], x["b"])


def test_btag_feature_is_an_extra_channel():
    """(c) btag listed under features enters the tensor as an ordinary extra channel (and the network as an input)."""
    from corenet.models.network import CoReNet
    path = _need(DATA / "hh4b_small.h5")
    spec4, spec5 = parse_config(_hh4b_cfg(["pt", "eta", "phi", "mass"])), parse_config(_hh4b_cfg(["pt", "eta", "phi", "mass", "btag"]))
    x4, x5 = H5Dataset(path, spec4).inputs, H5Dataset(path, spec5).inputs
    n = len(x4["event"])
    assert x5["b"].shape == (n, 6, 4) and x5["other"].shape == (n, 6, 8)
    assert torch.equal(x5["b"][:, :4], x4["b"][:, :4]) and torch.equal(x5["b"][:, -1], x4["b"][:, -1])   # same jets
    assert set(np.unique(x5["b"][:, 4][x5["b"][:, -1] > 0].numpy())) <= {0.0, 1.0}                       # the btag column
    with h5py.File(path, "r") as f:
        mask, pt, btag = (f[f"INPUTS/Source/{k}"][...] for k in ("MASK", "pt", "btag"))
    for i in range(N_CHECK):
        present = np.flatnonzero(mask[i])
        b = sorted(present, key=lambda j: (-btag[i, j], -pt[i, j]))[:4]
        assert np.array_equal(x5["b"][i, 4, :len(b)].numpy(), btag[i, b]), i
    m4, m5 = CoReNet(spec4, device="cpu"), CoReNet(spec5, device="cpu")
    assert m4._extras("b") == [] and m5._extras("b") == ["btag"] and m5._extras("other") == ["btag"]
    n4, n5 = (sum(p.numel() for p in m.parameters()) for m in (m4, m5))
    assert n4 == 1508 and n5 > n4, (n4, n5)
    batch = {k: v[:256] for k, v in x5.items()}
    m5.train(); m5.updateMeanStd(batch); m5.initMeanStd(); m5.eval(); m5.setGhostBatches(0)
    with torch.no_grad():
        out = m5(batch)
    assert out["logits"].shape == (256, 2) and torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["views"]["HH"]).all()


def _brute_force(mask, cols, pt, column, n, ascending, taken=None):
    """Per-event python sort: available members by the column (lowest first when ascending), ties by highest pt."""
    out = []
    for i in range(len(mask)):
        avail = [j for j in np.flatnonzero(mask[i]) if taken is None or j not in taken[i]]
        key = (lambda j: (cols[i, j], -pt[i, j])) if ascending else (lambda j: (-cols[i, j], -pt[i, j]))
        out.append(sorted(avail, key=key)[:n])
    return out


def test_generic_selectors_synthetic(tmp_path):
    """Selection by any column, highest or lowest first, against a brute-force per-event sort."""
    from synthetic import write_synthetic
    path = write_synthetic(tmp_path / "s.h5", "hh4b", n=600)
    with h5py.File(path, "r") as f:
        mask = f["INPUTS/Source/MASK"][...]
        jet = {k: f[f"INPUTS/Source/{k}"][...] for k in ("pt", "eta", "phi", "mass", "btag")}
    for line, column, n, asc in (("jets mass 3", "mass", 3, False), ("jets mass 2 lowest", "mass", 2, True),
                                 ("jets btag 4", "btag", 4, False), ("jets pt 4 lowest", "pt", 4, True)):
        cfg = _hh4b_cfg(["pt", "eta", "phi", "mass"])
        cfg["objects"] = {"jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12}, "b": line, "other": "jets rest 8"}
        ds = H5Dataset(path, parse_config(cfg))
        x = ds.inputs
        assert x["b"].shape == (600, 5, n) and x["other"].shape == (600, 5, 8)
        want = _brute_force(mask, jet[column], jet["pt"], column, n, asc)
        for i in range(600):
            b = want[i]
            assert np.array_equal(x["b"][i, 0, :len(b)].numpy(), jet["pt"][i, b]), (line, i)
            assert np.array_equal(x["b"][i, 1, :len(b)].numpy(), jet["eta"][i, b]), (line, i)
            assert (x["b"][i, -1, len(b):] == -1).all()
            rest = sorted(set(np.flatnonzero(mask[i])) - set(b), key=lambda j: -jet["pt"][i, j])[:8]
            assert np.array_equal(x["other"][i, 0, :len(rest)].numpy(), jet["pt"][i, rest]), (line, i)
    # rest_order="selector" continues the previous line's order, lowest first included
    cfg = _hh4b_cfg(["pt", "eta", "phi", "mass"])
    cfg["objects"] = {"jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12}, "b": "jets mass 2 lowest", "other": "jets rest 8"}
    x = H5Dataset(path, parse_config(cfg), rest_order="selector").inputs
    for i in range(600):
        order = _brute_force(mask, jet["mass"], jet["pt"], "mass", 12, True)[i]
        rest = order[2:10]
        assert np.array_equal(x["other"][i, 0, :len(rest)].numpy(), jet["pt"][i, rest]), i


@pytest.mark.parametrize("name,group", [("hh4b_small.h5", "Source"), ("bbww_small.h5", "Jets")])
def test_lexicographic_order_matches_old_key(name, group):
    """The lexicographic (column, pt) sort gives the same btag selection as the former weighted key
    -(btag * 1e6 + pt) on every event of the example files."""
    path = _need(DATA / name)
    with h5py.File(path, "r") as f:
        mask = f[f"INPUTS/{group}/MASK"][...]
        pt, btag = (f[f"INPUTS/{group}/{k}"][...].astype(np.float32) for k in ("pt", "btag"))
    old = np.argsort(np.where(mask, -(btag * 1e6 + pt), np.inf), axis=1, kind="stable")
    from corenet.data.h5 import _take
    new, _ = _take((-btag, -pt), mask, mask.shape[1])
    n_avail = mask.sum(1)
    for i in range(len(mask)):
        assert np.array_equal(old[i, :n_avail[i]], new[i, :n_avail[i]]), i


def test_missing_selector_column(tmp_path):
    """The selector column must exist in the file even when it is not a feature."""
    src = _need(DATA / "hh4b_small.h5")
    dst = tmp_path / "nobtag.h5"
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        for key in ("INPUTS/Source/MASK", "INPUTS/Source/pt", "INPUTS/Source/eta", "INPUTS/Source/phi", "INPUTS/Source/mass",
                    "CLASSIFICATIONS/EVENT/signal", "EVENT/weight", "EVENT/fold"):
            g.create_dataset(key, data=f[key][:500])
    with pytest.raises(KeyError, match=r"missing columns \['btag'\].*selector columns \['btag'\]"):
        H5Dataset(dst, parse_config(_hh4b_cfg(["pt", "eta", "phi", "mass"])))
    cfg = _hh4b_cfg(["pt", "eta", "phi", "mass"]); cfg["objects"]["b"] = "jets pt 4"
    assert H5Dataset(dst, parse_config(cfg)).inputs["b"].shape == (500, 5, 4)      # pt selector: no btag needed


def test_errors(tmp_path):
    path = _need(DATA / "bbww_small.h5")
    cfg = {"objects": {"jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12}, "b": "jets btag 2",
                       "photon": {"features": ["pt", "eta", "phi"]}},
           "hypotheses": {"HH": {"H_bb": ["b", "b"], "X": ["b", "photon"]}}, "outputs": {"signal": "HH"}, "combine": ["HH"]}
    with pytest.raises(KeyError, match="photon"):
        H5Dataset(path, parse_config(cfg))
    cfg = {"objects": {"jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12}, "b": "jets btag 4"},
           "hypotheses": {"HH": {"H1": ["b", "b"], "H2": ["b", "b"]}}, "outputs": {"signal": "HH"}, "combine": ["HH"]}
    with pytest.raises(ValueError, match="labels"):                # bbWW labels 1, 2 with a single output
        H5Dataset(path, parse_config(cfg))
