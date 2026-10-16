"""
corenet-evaluate: apply a trained CoReNet to an HDF5 file and compute its performance.

    corenet-evaluate --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \\
        --output runs/hh4b/eval_fold0.h5

Events: by default the fold held out when the checkpoint was trained (``--fold -1`` = every event of
the file). The output file (layout in docs/Dataset.md) holds the class probabilities, the view
(candidate) probabilities of every multi-candidate hypothesis, the auxiliary head probabilities,
the event labels / weights and, for the main output, the ROC curve; the AUCs are printed and
written as JSON next to it (``<output>.json``).

Metrics use the event weights of the file, clipped at >= 0 (``--unweighted`` ignores them):
  * main AUC: class 1 (the first output) against every other class, scored by P(class 1);
  * one-vs-rest AUC of every class c present, scored by P(c);
  * the AUC of every auxiliary head against ``label == its class``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from .data.h5 import H5Dataset
from .models.config import NetworkSpec
from .models.network import CoReNet


def count_parameters(model: torch.nn.Module) -> int:
    """All parameters, also the frozen ones."""
    return int(sum(p.numel() for p in model.parameters()))


class Batches:
    """Whole-array tensors on the device, sliced into batches (the loader keeps everything in memory)."""

    def __init__(self, ds: H5Dataset, device):
        self.inputs = {k: v.to(device) for k, v in ds.inputs.items()}
        self.labels = ds.labels.to(device)
        self.device = device

    def slice(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        sel = torch.as_tensor(idx, device=self.device)
        return {k: v[sel] for k, v in self.inputs.items()}

    def iter(
        self,
        idx: np.ndarray,
        bs: int,
        shuffle: bool = False,
        drop_last: bool = False,
        generator=None,
    ):
        order = (
            idx[torch.randperm(len(idx), generator=generator).numpy()]
            if shuffle
            else idx
        )
        for start in range(0, len(order), bs):
            chunk = order[start : start + bs]
            if drop_last and len(chunk) < bs:
                return
            yield chunk, self.slice(chunk)


@torch.no_grad()
def predict(model: CoReNet, data: Batches, idx: np.ndarray, batch_size: int = 16384):
    """(class probabilities (n, nC), {hypothesis: view probabilities (n, V)}, {aux head: probability (n,)}).

    Puts the model in eval mode with 0 ghost batches (running statistics)."""
    model.eval()
    model.setGhostBatches(0)
    probs, views, heads = [], {}, {}
    for _, batch in data.iter(idx, batch_size):
        out = model(batch)
        probs.append(F.softmax(out["logits"], dim=1).cpu().numpy())
        for h, v in out["views"].items():
            views.setdefault(h, []).append(F.softmax(v, dim=1).cpu().numpy())
        for name, x in out["heads"].items():
            heads.setdefault(name, []).append(torch.sigmoid(x).cpu().numpy())
    cat = np.concatenate
    return (
        cat(probs),
        {h: cat(v) for h, v in views.items()},
        {k: cat(v) for k, v in heads.items()},
    )


def write_predictions(
    path,
    ds: H5Dataset,
    idx: np.ndarray,
    probs,
    views,
    heads,
    spec: NetworkSpec,
    attrs: dict | None = None,
    roc: dict | None = None,
) -> Path:
    """Prediction file; layout documented in docs/Dataset.md."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("class_probabilities", data=probs.astype(np.float32))
        for h, v in views.items():
            f.create_dataset(f"views/{h}", data=v.astype(np.float32))
        for name, p in heads.items():
            f.create_dataset(f"heads/{name}", data=p.astype(np.float32))
        f.create_dataset("EVENT/index", data=np.asarray(idx, dtype=np.int64))
        f.create_dataset("EVENT/event", data=ds.events.numpy()[idx])
        f.create_dataset("EVENT/label", data=ds.labels.numpy()[idx])
        f.create_dataset("EVENT/weight", data=ds.weights.numpy()[idx])
        f.create_dataset("EVENT/fold", data=ds.folds.numpy()[idx])
        if roc:
            for k, v in roc.items():
                f.create_dataset(f"roc/{k}", data=np.asarray(v))
        f.attrs["classes"] = json.dumps(spec.classes)
        f.attrs["main_hypothesis"] = spec.main.hypothesis
        f.attrs["input"] = str(ds.path)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v if isinstance(v, (int, float, str)) else json.dumps(v)
    return path


def compute_metrics(
    probs, heads, spec: NetworkSpec, label: np.ndarray, weight: np.ndarray | None
):
    """AUCs (see the module docstring) and the ROC curve of the main output."""
    from sklearn.metrics import roc_auc_score, roc_curve

    w = None if weight is None else np.clip(weight, 0, None)
    out = {"n_events": int(len(label)), "weighted": weight is not None, "auc": {}}

    def auc(y, s):
        if y.all() or not y.any():
            return None
        return float(roc_auc_score(y, s, sample_weight=w))

    y_main = label == 1
    out["auc"]["main"] = auc(y_main, probs[:, 1])
    out["auc"]["one_vs_rest"] = {
        c: auc(label == i, probs[:, i])
        for i, c in enumerate(spec.classes)
        if (label == i).any()
    }
    classes = spec.classes
    out["auc"]["heads"] = {
        name: auc(label == classes.index(name), p) for name, p in heads.items()
    }
    roc = None
    if y_main.any() and not y_main.all():
        fpr, tpr, thr = roc_curve(y_main, probs[:, 1], sample_weight=w)
        roc = {
            "fpr": fpr.astype(np.float32),
            "tpr": tpr.astype(np.float32),
            "threshold": thr.astype(np.float32),
        }
    return out, roc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corenet-evaluate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True, help="model_fold{k}.pkl written by corenet-train"
    )
    p.add_argument("--input", required=True, help="HDF5 file to evaluate")
    p.add_argument(
        "--output",
        required=True,
        help="prediction file to write (.h5); metrics go to <output>.json",
    )
    p.add_argument(
        "--config",
        default=None,
        help="configuration file (default: the one stored in the checkpoint)",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="evaluate the events of this fold; -1 = every event (default: the checkpoint's held-out fold)",
    )
    p.add_argument(
        "--rest-order",
        choices=["pt", "selector"],
        default=None,
        help="ordering of the 'rest' selector (default: as in training)",
    )
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument(
        "--unweighted",
        action="store_true",
        help="ignore the event weights in the metrics",
    )
    p.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    from .checkpoint import load_checkpoint

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    device = torch.device(
        args.device
        if (not args.device.startswith("cuda") or torch.cuda.is_available())
        else "cpu"
    )
    model, spec, ckpt = load_checkpoint(args.checkpoint, args.config, device)
    meta = ckpt.get("metadata", {})
    rest_order = args.rest_order or meta.get("rest_order", "pt")
    ds = H5Dataset(args.input, spec, rest_order=rest_order)
    fold = args.fold if args.fold is not None else meta.get("fold", -1)
    idx = (
        np.arange(len(ds))
        if fold is None or fold < 0
        else np.flatnonzero(ds.folds.numpy() == fold)
    )
    if not len(idx):
        raise SystemExit(f"{args.input}: no events in fold {fold}")
    logging.info(
        f"{args.checkpoint}: {count_parameters(model):,} parameters; evaluating {len(idx):,} events "
        f"({'all folds' if fold is None or fold < 0 else f'fold {fold}'}) of {args.input}"
    )

    probs, views, heads = predict(model, Batches(ds, device), idx, args.batch_size)
    label = ds.labels.numpy()[idx]
    weight = None if args.unweighted else ds.weights.numpy()[idx]
    metrics, roc = compute_metrics(probs, heads, spec, label, weight)
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "input": str(args.input),
            "fold": fold,
            "classes": spec.classes,
        }
    )
    out = write_predictions(
        args.output,
        ds,
        idx,
        probs,
        views,
        heads,
        spec,
        roc=roc,
        attrs={
            "checkpoint": str(args.checkpoint),
            "fold": -1 if fold is None else int(fold),
            "n_parameters": count_parameters(model),
            "auc_main": metrics["auc"]["main"] or float("nan"),
        },
    )
    json_path = out.with_suffix(out.suffix + ".json")
    json_path.write_text(json.dumps(metrics, indent=1))
    logging.info(f"wrote {out} and {json_path}")
    print(json.dumps(metrics["auc"], indent=1))
    return metrics


def cli():
    """Console entry point (exit status 0; ``main`` returns the result for programmatic use)."""
    main()


if __name__ == "__main__":
    cli()
