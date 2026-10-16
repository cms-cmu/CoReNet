"""corenet-export: TorchScript and ONNX exports of both example networks agree with PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from synthetic import write_synthetic

from corenet import export
from corenet.checkpoint import load_checkpoint, save_checkpoint
from corenet.data import H5Dataset
from corenet.models.config import parse_config, read_config_dict
from corenet.models.network import CoReNet

repo = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("topology", ["hh4b", "bbww"])
def test_export(tmp_path, topology):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = repo / f"examples/{topology}/{topology}.yaml"
    data = write_synthetic(tmp_path / f"{topology}.h5", topology, n=600)
    cfg = read_config_dict(config)
    torch.manual_seed(0)
    model = CoReNet(parse_config(cfg), device="cpu")
    ds = H5Dataset(data, model.spec)
    model.train()
    model.updateMeanStd(ds.inputs)
    model.initMeanStd()
    ckpt = save_checkpoint(
        tmp_path / "model.pkl", model, cfg, config, {"fold": 0, "rest_order": "pt"}
    )
    reloaded, _, _ = load_checkpoint(ckpt)  # eval mode, 0 ghost batches
    model.setGhostBatches(0)
    for (k, a), b in zip(model.state_dict().items(), reloaded.state_dict().values()):
        assert torch.equal(a, b), k

    meta = export.main(
        [
            "--checkpoint",
            str(ckpt),
            "--input",
            str(data),
            "--output-dir",
            str(tmp_path / "export"),
            "--n-events",
            "256",
        ]
    )
    assert set(meta["files"]) == {"torchscript", "onnx"}
    assert max(meta["max_abs_diff"].values()) <= 1e-4, meta["max_abs_diff"]
    assert (
        json.loads((tmp_path / "export" / "model.json").read_text())["outputs"][0]
        == "class_probabilities"
    )
    # the TorchScript file needs neither corenet nor the configuration
    ts = torch.jit.load(str(tmp_path / "export" / "model.pt"))
    names = [*model.leaf_order, "event"]
    x = tuple(ds.inputs[k][:7] for k in names)
    with torch.no_grad():
        probs = ts(*x)[0]
    assert probs.shape == (7, len(model.spec.classes)) and np.allclose(
        probs.sum(1).numpy(), 1, atol=1e-5
    )
