"""Checkpoint format: new checkpoints carry ``format``; checkpoints written before every feature became a
network input (btag listed under features but skipped) still load, from their stored configuration."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from corenet import evaluate
from corenet.checkpoint import CHECKPOINT_FORMAT, legacy_config, load_checkpoint, save_checkpoint
from corenet.data import H5Dataset
from corenet.models.config import parse_config, read_config_dict
from corenet.models.network import CoReNet
from synthetic import write_synthetic

repo = Path(__file__).resolve().parent.parent


def _legacy_checkpoint(tmp_path):
    """A checkpoint as corenet-train wrote it before ``format``: configuration with btag under features, state
    dict of the network that ignored it (= today's network without the btag feature)."""
    config = repo / "examples/hh4b/hh4b.yaml"
    cfg = read_config_dict(config)
    old_cfg = copy.deepcopy(cfg)
    old_cfg["objects"]["jets"]["features"] = ["pt", "eta", "phi", "mass", "btag"]
    torch.manual_seed(0)
    model = CoReNet(parse_config(cfg), device="cpu")
    path = tmp_path / "legacy.pkl"
    torch.save({"model": model.state_dict(), "config": old_cfg, "config_path": str(config), "classes": model.spec.classes,
                "metadata": {"fold": 0, "rest_order": "pt"}}, path)
    return path, old_cfg, model


def test_legacy_config_translation():
    cfg = {"objects": {"jets": {"features": ["pt", "eta", "phi", "mass", "btag"], "max": 12}, "b": "jets btag 4",
                       "lepton": {"features": ["pt", "eta", "phi", "mass", "isE"]}}}
    new, edited = legacy_config(cfg)
    assert edited == ["jets"] and new["objects"]["jets"]["features"] == ["pt", "eta", "phi", "mass"]
    assert new["objects"]["lepton"]["features"] == ["pt", "eta", "phi", "mass", "isE"]
    assert cfg["objects"]["jets"]["features"][-1] == "btag"                       # the input is not modified
    assert legacy_config({"objects": {"jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12}}})[1] == []


def _same_state(a, b) -> bool:
    """Same keys, every tensor equal."""
    sa, sb = a.state_dict(), b.state_dict()
    return list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_new_checkpoint_has_format(tmp_path):
    cfg = read_config_dict(repo / "examples/hh4b/hh4b.yaml")
    model = CoReNet(parse_config(cfg), device="cpu")
    path = save_checkpoint(tmp_path / "m.pkl", model, cfg, repo / "examples/hh4b/hh4b.yaml", {"fold": 0})
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["format"] == CHECKPOINT_FORMAT == 2
    reloaded, spec, _ = load_checkpoint(path)
    assert spec.collections["jets"].features == ["pt", "eta", "phi", "mass"]
    model.setGhostBatches(0)          # load_checkpoint sets the nGhostBatches buffers to 0 (eval mode)
    assert _same_state(model, reloaded)


def test_legacy_checkpoint_loads(tmp_path):
    path, old_cfg, model = _legacy_checkpoint(tmp_path)
    reloaded, spec, ck = load_checkpoint(path)
    assert "format" not in ck
    assert spec.collections["jets"].features == ["pt", "eta", "phi", "mass"]      # btag dropped: the trained network
    model.setGhostBatches(0)
    assert _same_state(model, reloaded)
    # corenet-evaluate on it: the loader builds (N, 5, n) tensors from the translated configuration
    data = write_synthetic(tmp_path / "d.h5", "hh4b", n=300)
    out = tmp_path / "eval.h5"
    metrics = evaluate.main(["--checkpoint", str(path), "--input", str(data), "--output", str(out), "--fold", "-1"])
    assert metrics["n_events"] == 300 and out.exists()
    # the same legacy checkpoint with an explicit, untranslated configuration: a clear error
    import yaml
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump(old_cfg, sort_keys=False))
    with pytest.raises(RuntimeError, match="predates format 2"):
        load_checkpoint(path, config=old)
    # with btag as a feature the network is a different (larger) one: loading its own checkpoint works
    big = CoReNet(parse_config(old_cfg), device="cpu")
    big_ck = save_checkpoint(tmp_path / "big.pkl", big, old_cfg, old, {"fold": 0})
    _, spec_big, ck_big = load_checkpoint(big_ck)
    assert ck_big["format"] == 2 and spec_big.collections["jets"].features[-1] == "btag"
    assert H5Dataset(data, spec_big).inputs["b"].shape == (300, 6, 4)


@pytest.mark.parametrize("run", ["runs/repro_quickstart", "runs/repro_hh4b_full"])
def test_shipped_legacy_runs(run):
    """Checkpoints trained in this repository before the format marker: they load and evaluate."""
    path, data = repo / run / "model_fold0.pkl", repo / "examples/data/hh4b_small.h5"
    if not (path.exists() and data.exists()):
        pytest.skip(f"{path} or {data} not found")
    model, spec, ck = load_checkpoint(path)
    assert "format" not in ck and spec.collections["jets"].features == ["pt", "eta", "phi", "mass"]
    ds = H5Dataset(data, spec, rest_order=ck.get("metadata", {}).get("rest_order", "pt"))
    with torch.no_grad():
        out = model({k: v[:256] for k, v in ds.inputs.items()})
    assert out["logits"].shape == (256, 2) and torch.isfinite(out["logits"]).all()
