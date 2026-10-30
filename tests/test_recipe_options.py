"""Training-recipe options that default to off, on small synthetic files, on the CPU:

1. finetune_reset_optimizer: a new optimizer and learning-rate schedule at the freeze epoch;
2. masked_classes: classes removed from the softmax for flagged events (loss, validation loss,
   loader and validator checks); the evaluation keeps every class.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from synthetic import write_synthetic

from corenet import evaluate, train, validate
from corenet.data.h5 import H5Dataset
from corenet.evaluate import Batches
from corenet.models.config import ConfigError, load_config, parse_config
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


# ----------------------------------------------------------------------------------- masked_classes
def _abc_file(tmp: Path, n_violations: int = 0) -> Path:
    """Synthetic file with labels a = 0, b = 1 (the boosted signal), c = 2 and a flag EVENT/SR that
    avoids class b, apart from ``n_violations`` flagged events of class b."""
    data = write_synthetic(
        tmp / "abc.h5",
        "hh4b",
        seed=3,
        n=600,
        flag="SR",
        masked_label=1,
        n_violations=n_violations,
    )
    with h5py.File(data, "r+") as f:
        label = f["EVENT/label"][...]
        rng = np.random.default_rng(5)
        f["EVENT/label"][...] = np.where(
            label == 0, np.where(rng.uniform(size=len(label)) < 0.5, 0, 2), 1
        )
    return data


def _abc_config(tmp: Path, masked_classes=None) -> Path:
    cfg = yaml.safe_load(HH4B.read_text())
    cfg["labels"] = ["a", "b", "c"]
    cfg["outputs"] = {"b": "HH"}
    cfg.pop("trainer", None)
    if masked_classes is not None:
        cfg["trainer"] = {"masked_classes": masked_classes}
    path = tmp / "abc.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_masked_cross_entropy_equals_column_deletion() -> None:
    """-inf logits = deleting the column for the flagged rows; zero gradient on the removed logits."""
    gen = torch.Generator().manual_seed(0)
    n, n_classes, excluded = 64, 4, 2
    logits = torch.randn(n, n_classes, generator=gen, dtype=torch.float64)
    logits.requires_grad_(True)
    flagged = torch.rand(n, generator=gen) < 0.4
    label = torch.randint(0, n_classes, (n,), generator=gen)
    kept = torch.tensor([0, 1, 3])
    label[flagged] = kept[torch.randint(0, 3, (int(flagged.sum()),), generator=gen)]
    w = torch.rand(n, generator=gen, dtype=torch.float64)
    mask = torch.zeros(n, n_classes, dtype=torch.bool)
    mask[flagged, excluded] = True

    masked = train.masked_cross_entropy(logits, label, mask)
    explicit = torch.empty(n, dtype=torch.float64)
    explicit[~flagged] = F.cross_entropy(
        logits[~flagged], label[~flagged], reduction="none"
    )
    shifted = label[flagged] - (label[flagged] > excluded).long()
    explicit[flagged] = F.cross_entropy(
        logits[flagged][:, kept], shifted, reduction="none"
    )
    assert torch.allclose(masked, explicit, atol=1e-6, rtol=0)
    loss = (masked * w).sum() / w.sum()
    assert torch.allclose(loss, (explicit * w).sum() / w.sum(), atol=1e-6, rtol=0)
    loss.backward()
    zero = torch.zeros(int(flagged.sum()), dtype=torch.float64)
    assert torch.equal(logits.grad[flagged, excluded], zero)
    assert (logits.grad[~flagged, excluded] != 0).all()
    plain = train.masked_cross_entropy(logits, label, None)
    assert torch.equal(plain, F.cross_entropy(logits, label, reduction="none"))


def test_masked_classes_training(tmp_path: Path) -> None:
    """1 epoch with masked_classes {SR: [b]}: the history records it; the evaluation is untouched."""
    data = _abc_file(tmp_path)
    config = _abc_config(tmp_path, {"SR": ["b"]})
    out_dir = tmp_path / "models"
    train.main(
        [
            "--config",
            str(config),
            "--input",
            str(data),
            "--fold",
            "0",
            "--exclude-folds",
            "1",
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch-size-init",
            "128",
            "--freeze-epoch",
            "1",
            "--device",
            "cpu",
        ]
    )
    hist = json.loads((out_dir / "history_fold0.json").read_text())
    assert hist["recipe"]["masked_classes"] == {"SR": ["b"]}
    assert np.isfinite(hist["history"][0]["train_loss"])
    assert np.isfinite(hist["history"][0]["val_loss"])
    with h5py.File(out_dir / "predictions_fold0.h5", "r") as f:
        probs = f["class_probabilities"][...]
    assert probs.shape[1] == 3
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5)


def test_masked_classes_blinding_error(tmp_path: Path) -> None:
    """A flagged event of an excluded class: the loader raises, the validator reports NOT VALID."""
    data = _abc_file(tmp_path, n_violations=3)
    config = _abc_config(tmp_path, {"SR": ["b"]})
    spec = load_config(config)
    message = "3 events with EVENT/SR true carry the excluded class `b`"
    with pytest.raises(ValueError, match=message):
        H5Dataset(data, spec, masked_classes={"SR": ["b"]})
    report = validate.validate(config, data)
    assert any(message in m for m in report.errors), report.errors
    buf = io.StringIO()
    report.print(buf)
    assert "NOT VALID" in buf.getvalue()
    assert validate.main(["--config", str(config), "--input", str(data)]) == 1
    # without the violations the same configuration is valid and the flags are loaded
    (tmp_path / "clean").mkdir()
    clean = _abc_file(tmp_path / "clean")
    report = validate.validate(config, clean)
    assert not report.errors, report.errors
    ds = H5Dataset(clean, spec, masked_classes={"SR": ["b"]})
    assert ds.flags["SR"].dtype == torch.bool and len(ds.flags["SR"]) == len(ds)
    assert set(ds[:5]["flags"]) == {"SR"}
    assert H5Dataset(clean, spec).flags == {}


def test_flag_column_rules(tmp_path: Path) -> None:
    """Missing flag column: KeyError; integer 0/1 is accepted; other values: ValueError."""
    data = _abc_file(tmp_path)
    spec = load_config(_abc_config(tmp_path))
    with pytest.raises(KeyError, match="EVENT/CR"):
        H5Dataset(data, spec, flags=["CR"])
    with h5py.File(data, "r+") as f:
        sr = f["EVENT/SR"][...]
        f.create_dataset("EVENT/SR01", data=sr.astype(np.int64))
        f.create_dataset("EVENT/SR2", data=sr.astype(np.int64) * 2)
    ds = H5Dataset(data, spec, flags=["SR01"])
    assert torch.equal(ds.flags["SR01"], torch.from_numpy(sr.astype(bool)))
    with pytest.raises(ValueError, match="boolean or integer 0/1"):
        H5Dataset(data, spec, flags=["SR2"])


def test_masked_validation_loss(tmp_path: Path) -> None:
    """validation_metrics with a class mask equals the masked formula on the full probabilities."""
    data = _abc_file(tmp_path)
    spec = load_config(_abc_config(tmp_path))
    ds = H5Dataset(data, spec, masked_classes={"SR": ["b"]})
    model = CoReNet(spec, device="cpu")
    batches = Batches(ds, "cpu")
    with torch.no_grad():
        model.train()
        model.updateMeanStd(batches.slice(np.arange(len(ds))))
        model.initMeanStd()
    val_idx = np.arange(200)
    mask = batches.class_mask(torch.as_tensor(val_idx), {"SR": ["b"]}, 3).numpy()
    assert mask[:, 1].sum() == int(ds.flags["SR"][:200].sum()) > 0
    labels, weights = ds.labels.numpy(), ds.weights
    out = train.validation_metrics(model, batches, val_idx, labels, weights, mask)
    probs = evaluate.predict(model, batches, val_idx)[0].astype(np.float64)
    label, w = labels[val_idx], weights.numpy()[val_idx].astype(np.float64)
    p = np.empty(len(val_idx))
    for i, row in enumerate(probs):
        keep = ~mask[i]
        p[i] = row[label[i]] / row[keep].sum()
    expected = -(w * np.log(p)).sum() / w.sum()
    assert out["val_loss"] == pytest.approx(expected, rel=1e-5)
    unmasked = train.validation_metrics(model, batches, val_idx, labels, weights)
    assert unmasked["val_loss"] != pytest.approx(expected, rel=1e-5)
    assert out["val_auc"] == unmasked["val_auc"]


def test_masked_classes_config() -> None:
    """The trainer key and the --masked-classes flag; malformed values are rejected."""
    cfg = yaml.safe_load(HH4B.read_text())
    cfg["labels"] = ["a", "b", "c"]
    cfg["outputs"] = {"b": "HH"}
    spec = parse_config(cfg)
    cfg["trainer"] = {"masked_classes": {"SR": ["b"]}}
    args = train.build_parser().parse_args(["--config", "x", "--input", "y"])
    assert train.resolve_recipe(args, cfg, spec).masked_classes == {"SR": ["b"]}
    args = train.build_parser().parse_args(
        ["--config", "x", "--input", "y", "--masked-classes", "SR=b,c", "CR=a"]
    )
    assert train.resolve_recipe(args, cfg, spec).masked_classes == {
        "SR": ["b", "c"],
        "CR": ["a"],
    }
    args = train.build_parser().parse_args(
        ["--config", "x", "--input", "y", "--masked-classes"]
    )
    assert train.resolve_recipe(args, cfg, spec).masked_classes == {}
    with pytest.raises(ConfigError, match="expected FLAG=CLASS"):
        train.resolve_recipe(
            train.build_parser().parse_args(
                ["--config", "x", "--input", "y", "--masked-classes", "SR"]
            ),
            cfg,
            spec,
        )
    assert "--masked-classes" in train.build_parser().format_usage()
    for bad, message in [
        ({"SR": ["d"]}, "unknown classes"),
        ({"SR": []}, "masked_classes must be a mapping"),
        ({"SR": ["a", "b", "c"]}, "masks every class"),
        (["SR"], "masked_classes must be a mapping"),
        ({"SR": "b"}, "masked_classes must be a mapping"),
        ({"SR": ["b", "b"]}, "masked_classes must be a mapping"),
    ]:
        cfg["trainer"] = {"masked_classes": bad}
        with pytest.raises(ConfigError, match=message):
            train.resolve_recipe(
                train.build_parser().parse_args(["--config", "x", "--input", "y"]),
                cfg,
                spec,
            )
