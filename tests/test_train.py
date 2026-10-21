"""corenet-train and corenet-evaluate on small synthetic files, on the CPU, for both example
configurations (HH -> 4b: 2 classes; HH -> bbWW: 3 classes and an auxiliary ttbar head):

  1. the training loss is finite and decreases, the history has a validation loss / AUC,
  2. the prediction file has the documented layout, normalised probabilities, the held-out events,
  3. the freeze leaves exactly the output-head parameters trainable, parameter counts 1,508 / 3,832,
  4. the checkpoint is self-contained and corenet-evaluate reproduces the training-time predictions,
  5. the `trainer:` section of the configuration sets the recipe and the command line overrides it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml
from synthetic import write_synthetic

from corenet import evaluate, train
from corenet.models.config import ConfigError, load_config
from corenet.models.network import CoReNet

repo = Path(__file__).resolve().parent.parent
HH4B = repo / "examples/hh4b/hh4b.yaml"
BBWW = repo / "examples/bbww/bbww.yaml"


def run_one(
    tmp: Path,
    topology: str,
    config: Path,
    expected_params: int,
    expected_classes: list[str],
) -> None:
    data = write_synthetic(tmp / f"{topology}_all.h5", topology)
    out_dir = tmp / f"models_{topology}"
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
            "2",
            "--batch-size-init",
            "128",
            "--freeze-epoch",
            "1",
            "--device",
            "cpu",
            "--seed",
            "7",
        ]
    )

    hist = json.loads((out_dir / "history_fold0.json").read_text())
    assert hist["classes"] == expected_classes, hist["classes"]
    assert (
        hist["recipe"]["epochs"] == 2
        and hist["recipe"]["lr"] == 1e-2
        and hist["recipe"]["seed"] == 7
    )
    losses = [e["train_loss"] for e in hist["history"]]
    assert len(losses) == 2 and all(np.isfinite(losses)), losses
    assert losses[1] < losses[0], (
        f"{topology}: training loss did not decrease: {losses}"
    )
    for e in hist["history"]:
        assert np.isfinite(e["val_loss"]), e
        assert 0.0 <= e["val_auc"] <= 1.0, e

    with h5py.File(data, "r") as f:
        fold, ev, label = (
            f["EVENT/fold"][...],
            f["EVENT/event"][...],
            f["EVENT/label"][...],
        )
    test = fold == 0
    pred = out_dir / "predictions_fold0.h5"
    with h5py.File(pred, "r") as f:
        for key in (
            "class_probabilities",
            "EVENT/event",
            "EVENT/label",
            "EVENT/weight",
            "EVENT/fold",
            "EVENT/index",
        ):
            assert key in f, f"{topology}: {key} missing from the prediction file"
        probs = f["class_probabilities"][...]
        assert probs.shape == (int(test.sum()), len(expected_classes)), probs.shape
        assert np.allclose(probs.sum(1), 1.0, atol=1e-5) and np.isfinite(probs).all()
        assert (f["EVENT/event"][...] == ev[test]).all() and (
            f["EVENT/label"][...] == label[test]
        ).all()
        assert int(f.attrs["n_parameters"]) == expected_params, f.attrs["n_parameters"]
        assert json.loads(f.attrs["classes"]) == expected_classes
        for name in [o.name for o in load_config(config).outputs[1:]]:
            p = f[f"heads/{name}"][...]
            assert p.shape == (int(test.sum()),) and ((0 <= p) & (p <= 1)).all()
        views = {h: f[f"views/{h}"][...] for h in f["views"]}
        for v in views.values():
            assert np.allclose(v.sum(1), 1.0, atol=1e-5) and np.isfinite(v).all()
        if topology == "hh4b":
            assert set(views) == {"HH"} and views["HH"].shape[1] == 3
        else:
            assert (
                set(views) == {"tt"} and views["tt"].shape[1] == 2
            )  # HH has a single candidate
        train_probs = probs

    # corenet-evaluate on the same events, from the self-contained checkpoint only
    out = tmp / f"eval_{topology}.h5"
    metrics = evaluate.main(
        [
            "--checkpoint",
            str(out_dir / "model_fold0.pkl"),
            "--input",
            str(data),
            "--output",
            str(out),
        ]
    )
    with h5py.File(out, "r") as f:
        # equal up to float rounding: after the freeze the parameters' requires_grad flags differ from the
        # reloaded model's, which changes the CPU kernels PyTorch selects (observed max |diff| ~3e-7)
        assert np.allclose(
            f["class_probabilities"][...], train_probs, atol=1e-6, rtol=0
        ), "evaluate differs from training"
        assert f["roc/fpr"].shape == f["roc/tpr"].shape and f["roc/tpr"][-1] == 1.0
    assert 0.5 < metrics["auc"]["main"] <= 1.0, metrics["auc"]
    assert json.loads(Path(str(out) + ".json").read_text())["auc"] == metrics["auc"]
    if topology == "bbww":
        assert metrics["auc"]["heads"]["ttbar"] is not None
    all_events = evaluate.main(
        [
            "--checkpoint",
            str(out_dir / "model_fold0.pkl"),
            "--input",
            str(data),
            "--output",
            str(tmp / f"eval_all_{topology}.h5"),
            "--fold",
            "-1",
        ]
    )
    assert all_events["n_events"] == len(fold)
    print(
        f"  {topology}: loss {losses[0]:.5f} -> {losses[1]:.5f}, test AUC {metrics['auc']['main']:.4f}"
    )


def check_freeze(config: Path, expected_params: int) -> None:
    model = CoReNet(load_config(config), device="cpu")
    assert evaluate.count_parameters(model) == expected_params, (
        evaluate.count_parameters(model)
    )
    n_trainable = train.freeze_heads(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "nothing left trainable"
    assert all(n.startswith(train.HEAD_PREFIXES) for n in trainable), sorted(trainable)
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert not any(n.startswith(train.HEAD_PREFIXES) for n in frozen), sorted(frozen)
    assert 0 < n_trainable < expected_params


def test_batch_schedule() -> None:
    assert [train.batch_schedule(e, 1024) for e in (0, 1, 3, 6, 10)] == [
        (1024, 64),
        (2048, 16),
        (4096, 4),
        (8192, 1),
        (16384, 0),
    ]
    assert [train.batch_schedule(e, 1024)[0] for e in range(12)] == [
        1024,
        2048,
        2048,
        4096,
        4096,
        4096,
        8192,
        8192,
        8192,
        8192,
        16384,
        16384,
    ]
    bs, n_gb = train.batch_schedule(
        0, 64
    )  # tiny batches: never a one-event ghost batch
    assert bs // max(n_gb, 1) >= train.MIN_GHOST_BATCH_SIZE, (bs, n_gb)


def test_freeze() -> None:
    check_freeze(HH4B, 1508)
    check_freeze(BBWW, 3832)


def test_recipe_from_config() -> None:
    cfg = yaml.safe_load(HH4B.read_text())
    cfg["trainer"] = {
        "epochs": 3,
        "lr": 0.005,
        "lr_milestones": [1],
        "class_norm": False,
    }
    args = train.build_parser().parse_args(
        ["--config", "x", "--input", "y", "--lr", "0.002"]
    )
    train.resolve_recipe(args, cfg)
    assert (
        args.epochs == 3
        and args.lr == 0.002
        and args.lr_milestones == [1]
        and args.class_norm is False
    )
    assert (
        args.freeze_epoch == 15
        and args.batch_size_init == 1024
        and args.lr_gamma == 0.25
        and args.seed == 42
    )
    args = train.build_parser().parse_args(["--config", "x", "--input", "y"])
    train.resolve_recipe(args, {})
    assert {k: getattr(args, k) for k in train.RECIPE_DEFAULTS} == train.RECIPE_DEFAULTS
    cfg["trainer"] = {"epoch": 3}
    try:
        train.resolve_recipe(
            train.build_parser().parse_args(["--config", "x", "--input", "y"]), cfg
        )
    except ConfigError as e:
        assert "unknown keys" in str(e)
    else:
        raise AssertionError("expected a ConfigError for an unknown trainer key")


def test_train_and_evaluate() -> None:
    with tempfile.TemporaryDirectory(prefix="corenet_train_") as tmp:
        run_one(Path(tmp), "hh4b", HH4B, 1508, ["rest", "signal"])
        run_one(Path(tmp), "bbww", BBWW, 3832, ["rest", "signal", "ttbar"])


def test_early_stopping() -> None:
    """--early-stop-patience: stop after N epochs without a better validation AUC and keep the best epoch."""
    import torch
    with tempfile.TemporaryDirectory(prefix="corenet_es_") as tmp:
        tmp = Path(tmp)
        data = write_synthetic(tmp / "hh4b_all.h5", "hh4b")
        out_dir = tmp / "m"
        train.main(["--config", str(HH4B), "--input", str(data), "--output-dir", str(out_dir), "--fold", "0",
                    "--exclude-folds", "1", "--epochs", "12", "--batch-size-init", "128", "--early-stop-patience", "1",
                    "--device", "cpu"])
        h = json.loads((out_dir / "history_fold0.json").read_text())
        aucs = [e["val_auc"] for e in h["history"]]
        best = max(range(len(aucs)), key=lambda i: aucs[i])
        assert h["best_epoch"] == best and h["early_stop_patience"] == 1
        if h["stopped_after_epoch"] is not None:                    # stopped: one non-improving epoch after the best
            assert h["stopped_after_epoch"] == best + 1 and len(aucs) == best + 2
        else:
            assert len(aucs) == 12
        ck = torch.load(out_dir / "model_fold0.pkl", map_location="cpu", weights_only=False)
        assert ck["metadata"]["epoch"] == best and ck["metadata"]["early_stop_patience"] == 1
        # without a validation fold the option is refused
        try:
            train.main(["--config", str(HH4B), "--input", str(data), "--output-dir", str(tmp / "n"), "--fold", "0",
                        "--epochs", "2", "--early-stop-patience", "1", "--device", "cpu"])
        except SystemExit:
            pass
        else:
            raise AssertionError("expected an error without --exclude-folds")
