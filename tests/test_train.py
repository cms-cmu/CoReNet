"""corenet-train and corenet-evaluate on small synthetic files, on the CPU, for both example
configurations (HH -> 4b: 2 classes; HH -> bbWW: 3 classes and an auxiliary ttbar head):

  1. the training loss is finite and decreases, the history has a validation loss / AUC,
  2. the prediction file has the documented layout, normalised probabilities, the held-out events,
  3. the freeze leaves exactly the output-head parameters trainable, parameter counts 1,508 / 3,832,
  4. the checkpoint is self-contained and corenet-evaluate reproduces the training-time predictions,
  5. the `trainer:` section of the configuration sets the recipe and the command line overrides it,
  6. the fine-tuning stage (finetune_layers / finetune_lr / finetune_batch_size) and its defaults,
  7. the fold rule (--n-folds, --exclude-folds next) and its record in checkpoint and predictions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
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
                "12",
                "--batch-size-init",
                "128",
                "--early-stop-patience",
                "1",
                "--device",
                "cpu",
            ]
        )
        h = json.loads((out_dir / "history_fold0.json").read_text())
        aucs = [e["val_auc"] for e in h["history"]]
        best = max(range(len(aucs)), key=lambda i: aucs[i])
        assert h["best_epoch"] == best and h["early_stop_patience"] == 1
        if (
            h["stopped_after_epoch"] is not None
        ):  # stopped: one non-improving epoch after the best
            assert h["stopped_after_epoch"] == best + 1 and len(aucs) == best + 2
        else:
            assert len(aucs) == 12
        ck = torch.load(
            out_dir / "model_fold0.pkl", map_location="cpu", weights_only=False
        )
        assert (
            ck["metadata"]["epoch"] == best
            and ck["metadata"]["early_stop_patience"] == 1
        )
        # without a validation fold the option is refused
        try:
            train.main(
                [
                    "--config",
                    str(HH4B),
                    "--input",
                    str(data),
                    "--output-dir",
                    str(tmp / "n"),
                    "--fold",
                    "0",
                    "--epochs",
                    "2",
                    "--early-stop-patience",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("expected an error without --exclude-folds")


def _train(data: Path, out_dir: Path, *extra: str, epochs: int = 2) -> dict:
    """corenet-train on the synthetic hh4b file (fold 0 held out, fold 1 validation, freeze at epoch 1)."""
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
            "128",
            "--freeze-epoch",
            "1",
            "--device",
            "cpu",
            *extra,
        ]
    )
    return json.loads((out_dir / "history_fold0.json").read_text())


def _metadata(out_dir: Path, fold: int = 0) -> dict:
    import torch

    return torch.load(
        out_dir / f"model_fold{fold}.pkl", map_location="cpu", weights_only=False
    )["metadata"]


def test_finetune_layers() -> None:
    """The default fine-tuning stage equals the explicit head prefixes bit for bit; `none` freezes nothing."""
    with tempfile.TemporaryDirectory(prefix="corenet_ft_") as tmp:
        tmp = Path(tmp)
        data = write_synthetic(tmp / "hh4b_all.h5", "hh4b")
        default = _train(data, tmp / "default")
        explicit = _train(
            data, tmp / "explicit", "--finetune-layers", *train.HEAD_PREFIXES
        )
        assert default["history"] == explicit["history"]
        assert default["recipe"] == explicit["recipe"]
        assert default["recipe"]["finetune_layers"] == list(train.HEAD_PREFIXES)
        assert (
            default["recipe"]["finetune_lr"] is None
            and default["recipe"]["finetune_batch_size"] is None
        )
        fresh = CoReNet(load_config(HH4B), device="cpu")
        n_all = train.count_trainable(
            fresh
        )  # some parameters are non-trainable by construction
        n_heads = train.freeze_heads(fresh)
        assert 0 < n_heads < n_all < evaluate.count_parameters(fresh)
        assert default["finetune_trainable_parameters"] == n_heads
        meta = _metadata(tmp / "default")
        assert meta["finetune_trainable_parameters"] == n_heads
        assert meta["finetune_layers"] == list(train.HEAD_PREFIXES)
        assert meta["finetune_lr"] is None and meta["finetune_batch_size"] is None
        # `none`: nothing is frozen, every trainable parameter of a fresh model stays trainable
        none = _train(data, tmp / "none", "--finetune-layers", "none")
        assert none["recipe"]["finetune_layers"] == []
        assert none["finetune_trainable_parameters"] == n_all
        assert _metadata(tmp / "none")["finetune_trainable_parameters"] == n_all
        assert (
            none["history"][0] == default["history"][0]
        )  # identical before the freeze
        assert none["history"][1]["train_loss"] != default["history"][1]["train_loss"]
        with pytest.raises(ConfigError, match="'none' cannot be combined"):
            _train(data, tmp / "bad", "--finetune-layers", "none", "head_main.")


def test_finetune_lr_and_batch_size() -> None:
    """--finetune-lr / --finetune-batch-size apply from the freeze epoch; the milestones continue from the lr."""
    with tempfile.TemporaryDirectory(prefix="corenet_ft_") as tmp:
        tmp = Path(tmp)
        data = write_synthetic(tmp / "hh4b_all.h5", "hh4b")
        h = _train(
            data,
            tmp / "m",
            "--finetune-lr",
            "1e-4",
            "--finetune-batch-size",
            "256",
            "--lr-milestones",
            "3",
            epochs=3,
        )
        e0, e1, e2 = h["history"]
        assert (
            e0["batch_size"] == 128 and e0["ghost_batches"] == 16
        )  # schedule: 128 // 8
        assert e0["lr"] == 1e-2
        assert (
            e1["batch_size"] == 256 and e1["ghost_batches"] == 16
        )  # schedule: min(16, 256 // 8)
        assert e1["lr"] == 1e-4
        assert e2["batch_size"] == 256 and e2["ghost_batches"] == 16
        assert e2["lr"] == pytest.approx(
            1e-4 * 0.25
        )  # milestone 3 multiplies the fine-tuning lr
        assert h["recipe"]["finetune_lr"] == 1e-4
        assert h["recipe"]["finetune_batch_size"] == 256
        meta = _metadata(tmp / "m")
        assert meta["finetune_lr"] == 1e-4 and meta["finetune_batch_size"] == 256
        assert (
            meta["finetune_trainable_parameters"] == h["finetune_trainable_parameters"]
        )
    assert train.finetune_batch(10, 1024, None) == (16384, 0)
    assert train.finetune_batch(10, 1024, 256) == (256, 0)  # 0 ghost batches stay 0
    assert train.finetune_batch(1, 1024, 256) == (256, 16)
    assert train.finetune_batch(1, 1024, 32) == (32, 4)
    assert train.finetune_batch(0, 1024, 4) == (4, 1)


def test_finetune_recipe_from_config() -> None:
    """The three fine-tuning keys in the trainer: section, and their validation."""
    cfg = yaml.safe_load(HH4B.read_text())
    cfg["trainer"] = {
        "finetune_layers": ["head_main.", "resnet."],
        "finetune_lr": 0.0001,
        "finetune_batch_size": 256,
    }
    args = train.build_parser().parse_args(["--config", "x", "--input", "y"])
    train.resolve_recipe(args, cfg)
    assert args.finetune_layers == ["head_main.", "resnet."]
    assert args.finetune_lr == 0.0001 and args.finetune_batch_size == 256
    args = train.build_parser().parse_args(
        ["--config", "x", "--input", "y", "--finetune-layers", "none"]
    )
    train.resolve_recipe(args, cfg)  # the command line wins
    assert args.finetune_layers == []
    cfg["trainer"] = {"finetune_layers": "none"}
    args = train.build_parser().parse_args(["--config", "x", "--input", "y"])
    assert train.resolve_recipe(args, cfg).finetune_layers == []
    for bad, message in [
        ({"finetune_layers": ["none", "head_main."]}, "cannot be combined"),
        ({"finetune_layers": [1]}, "finetune_layers must be a list"),
        ({"finetune_lr": 0.0}, "finetune_lr must be a number > 0"),
        ({"finetune_lr": "1e-4"}, "finetune_lr must be a number > 0"),
        ({"finetune_batch_size": 0}, "finetune_batch_size must be an integer >= 1"),
    ]:
        cfg["trainer"] = bad
        with pytest.raises(ConfigError, match=message):
            train.resolve_recipe(
                train.build_parser().parse_args(["--config", "x", "--input", "y"]), cfg
            )


def test_n_folds_rule() -> None:
    """--n-folds K: folds = squares(event) % K; --exclude-folds next; the rule and its seed are stored and
    reused by corenet-evaluate."""
    from corenet.data.folds import fold_of
    from corenet.data.h5 import FOLD_RULE_COLUMN, FOLD_RULE_EVENT

    assert train.resolve_exclude_folds(["next"], 4, 5) == [0]
    assert train.resolve_exclude_folds([2, "next"], 0, 3) == [2, 1]
    with tempfile.TemporaryDirectory(prefix="corenet_nf_") as tmp:
        tmp = Path(tmp)
        data = write_synthetic(
            tmp / "hh4b_all.h5", "hh4b"
        )  # events 0..N-1, 5 random folds
        out_dir = tmp / "m"
        train.main(
            [
                "--config",
                str(HH4B),
                "--input",
                str(data),
                "--output-dir",
                str(out_dir),
                "--n-folds",
                "3",
                "--fold",
                "0",
                "--exclude-folds",
                "next",
                "--epochs",
                "1",
                "--batch-size-init",
                "128",
                "--device",
                "cpu",
            ]
        )
        h = json.loads((out_dir / "history_fold0.json").read_text())
        assert h["exclude_folds"] == [1] and h["n_folds"] == 3
        assert h["fold_rule"] == FOLD_RULE_EVENT == "squares(event) % n_folds"
        assert h["fold_seed"] == 0
        meta = _metadata(out_dir)
        assert meta["n_folds"] == 3 and meta["fold_rule"] == FOLD_RULE_EVENT
        assert meta["fold_seed"] == 0 and meta["exclude_folds"] == [1]
        with h5py.File(data, "r") as f:
            event = f["EVENT/event"][...]
        folds3 = fold_of(event, 3)
        n_test, n_val = int((folds3 == 0).sum()), int((folds3 == 1).sum())
        assert h["n_val"] == n_val and h["n_train"] == len(event) - n_test - n_val
        with h5py.File(out_dir / "predictions_fold0.h5", "r") as f:
            assert f.attrs["n_folds"] == 3 and f.attrs["fold_rule"] == FOLD_RULE_EVENT
            assert f.attrs["fold_seed"] == 0
            assert (fold_of(f["EVENT/event"][...], 3) == 0).all() and len(
                f["EVENT/event"]
            ) == n_test
            assert (f["EVENT/fold"][...] == 0).all()
        # corenet-evaluate applies the checkpoint's rule, also to a file without the column
        with h5py.File(data, "a") as f:
            del f["EVENT/fold"]
        metrics = evaluate.main(
            [
                "--checkpoint",
                str(out_dir / "model_fold0.pkl"),
                "--input",
                str(data),
                "--output",
                str(tmp / "eval.h5"),
            ]
        )
        assert metrics["n_events"] == n_test
        with h5py.File(tmp / "eval.h5", "r") as f:
            assert f.attrs["n_folds"] == 3 and f.attrs["fold_rule"] == FOLD_RULE_EVENT
            assert f.attrs["fold_seed"] == 0
        assert evaluate.main(
            [
                "--checkpoint",
                str(out_dir / "model_fold0.pkl"),
                "--input",
                str(data),
                "--output",
                str(tmp / "eval5.h5"),
                "--n-folds",
                "5",
            ]
        )["n_events"] == int((fold_of(event, 5) == 0).sum())
        # a checkpoint trained on the column keeps the column rule and needs it in the file
        with_column = write_synthetic(tmp / "col.h5", "hh4b")
        train.main(
            [
                "--config",
                str(HH4B),
                "--input",
                str(with_column),
                "--output-dir",
                str(tmp / "c"),
                "--fold",
                "4",
                "--exclude-folds",
                "next",
                "--epochs",
                "1",
                "--batch-size-init",
                "128",
                "--device",
                "cpu",
            ]
        )
        meta = _metadata(tmp / "c", fold=4)
        assert (
            meta["n_folds"] == 5
            and meta["fold_rule"] == FOLD_RULE_COLUMN == "EVENT/fold"
        )
        assert meta["exclude_folds"] == [0]
        with pytest.raises(KeyError, match="no EVENT/fold dataset"):
            evaluate.main(
                [
                    "--checkpoint",
                    str(tmp / "c/model_fold4.pkl"),
                    "--input",
                    str(data),
                    "--output",
                    str(tmp / "eval_c.h5"),
                ]
            )
