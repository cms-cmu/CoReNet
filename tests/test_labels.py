"""``labels:``: a network with N classes named in label order, trained and evaluated on the CPU on a small
synthetic file with labels 0, 1, 2 (``labels: [a, b, c]``, ``outputs: {b: HH}``): the checkpoint and the
prediction file carry the labels as classes, and the main AUC is that of class 1 = ``b``."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from synthetic import write_synthetic

from corenet import evaluate, train

repo = Path(__file__).resolve().parent.parent
HH4B = repo / "examples/hh4b/hh4b.yaml"


def test_labels_train_and_evaluate(tmp_path: Path) -> None:
    data = write_synthetic(tmp_path / "abc.h5", "hh4b", seed=3, n=600)
    with h5py.File(data, "r+") as f:
        label = f["EVENT/label"][...]
        rng = np.random.default_rng(5)
        label = np.where(
            label == 0, np.where(rng.uniform(size=len(label)) < 0.5, 0, 2), 1
        )  # a = 0, b = 1 (the boosted signal), c = 2
        f["EVENT/label"][...] = label
    assert set(np.unique(label)) == {0, 1, 2}

    cfg = yaml.safe_load(HH4B.read_text())
    cfg["labels"] = ["a", "b", "c"]
    cfg["outputs"] = {"b": "HH"}
    cfg.pop("trainer", None)
    config = tmp_path / "abc.yaml"
    config.write_text(yaml.safe_dump(cfg))

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
    ckpt = torch.load(
        out_dir / "model_fold0.pkl", map_location="cpu", weights_only=False
    )
    assert ckpt["classes"] == ["a", "b", "c"]
    with h5py.File(out_dir / "predictions_fold0.h5", "r") as f:
        assert f["class_probabilities"].shape[1] == 3
        assert json.loads(f.attrs["classes"]) == ["a", "b", "c"]
        assert "heads" not in f or len(f["heads"]) == 0

    out = tmp_path / "eval.h5"
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
    from sklearn.metrics import roc_auc_score

    with h5py.File(out, "r") as f:
        probs = f["class_probabilities"][...]
        lab = f["EVENT/label"][...]
        w = np.clip(f["EVENT/weight"][...], 0, None)
    assert probs.shape[1] == 3
    expected = roc_auc_score(lab == 1, probs[:, 1], sample_weight=w)
    assert metrics["auc"]["main"] == expected == metrics["auc"]["one_vs_rest"]["b"]
    saved = json.loads(Path(str(out) + ".json").read_text())
    assert saved["auc"]["main"] == expected
