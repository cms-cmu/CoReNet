"""Training-recipe options that default to off, on small synthetic files, on the CPU:

1. finetune_reset_optimizer: a new optimizer and learning-rate schedule at the freeze epoch.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
from synthetic import write_synthetic

from corenet import train
from corenet.models.config import ConfigError, load_config
from corenet.models.network import CoReNet

repo = Path(__file__).resolve().parent.parent
HH4B = repo / "examples/hh4b/hh4b.yaml"


def _train(data: Path, out_dir: Path, *extra: str, epochs: int = 3) -> dict:
    """corenet-train on the synthetic hh4b file: fold 0 held out, fold 1 validation, freeze at epoch 2."""
    train.main(
        [
            "--config",
            str(HH4B),
            "--input",
            str(data),
            "--output-dir",
            str(out_dir),
            "--fold",
            "0",
            "--exclude-folds",
            "1",
            "--epochs",
            str(epochs),
            "--batch-size-init",
            "256",
            "--freeze-epoch",
            "2",
            "--seed",
            "7",
            "--device",
            "cpu",
            *extra,
        ]
    )
    return json.loads((out_dir / "history_fold0.json").read_text())


def _state(out_dir: Path) -> dict:
    return torch.load(
        out_dir / "model_fold0.pkl", map_location="cpu", weights_only=False
    )["model"]


def test_finetune_reset_optimizer_training() -> None:
    """Off and on with the same seed: both run, the history records the option, the weights differ."""
    with tempfile.TemporaryDirectory(prefix="corenet_reset_") as tmp:
        tmp = Path(tmp)
        data = write_synthetic(tmp / "hh4b_all.h5", "hh4b")
        off = _train(data, tmp / "off")
        on = _train(data, tmp / "on", "--finetune-reset-optimizer")
        assert off["recipe"]["finetune_reset_optimizer"] is False
        assert on["recipe"]["finetune_reset_optimizer"] is True
        assert len(off["history"]) == len(on["history"]) == 3
        assert off["history"][:2] == on["history"][:2]  # identical before the freeze
        key = "head_main.conv.module.weight"
        assert not torch.equal(_state(tmp / "off")[key], _state(tmp / "on")[key])


def test_reset_optimizer_helper() -> None:
    """The fresh optimizer has no state, the requested lr and exactly the trainable parameters."""
    model = CoReNet(load_config(HH4B), device="cpu")
    train.freeze_heads(model)
    opt, sched = train._reset_optimizer(model, 1e-3, [1, 2], 0.5)
    assert len(opt.state) == 0
    assert opt.param_groups[0]["lr"] == 1e-3
    trainable = [p for p in model.parameters() if p.requires_grad]
    params = opt.param_groups[0]["params"]
    assert len(params) == len(trainable)
    assert all(a is b for a, b in zip(params, trainable))
    assert sorted(sched.milestones) == [1, 2] and sched.gamma == 0.5


def test_finetune_reset_optimizer_config() -> None:
    cfg = yaml.safe_load(HH4B.read_text())
    cfg["trainer"] = {"finetune_reset_optimizer": True}
    args = train.build_parser().parse_args(["--config", "x", "--input", "y"])
    assert train.resolve_recipe(args, cfg).finetune_reset_optimizer is True
    args = train.build_parser().parse_args(
        ["--config", "x", "--input", "y", "--no-finetune-reset-optimizer"]
    )
    assert train.resolve_recipe(args, cfg).finetune_reset_optimizer is False
    cfg["trainer"] = {"finetune_reset_optimizer": "yes"}
    with pytest.raises(
        ConfigError, match="finetune_reset_optimizer must be true or false"
    ):
        train.resolve_recipe(
            train.build_parser().parse_args(["--config", "x", "--input", "y"]), cfg
        )
