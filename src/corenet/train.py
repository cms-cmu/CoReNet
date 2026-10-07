"""
corenet-train: train CoReNet from a configuration file and an HDF5 file.

The network, the classes and the auxiliary heads all come from ``--config``; the tensors come from
``corenet.data.H5Dataset``. Training recipe (defaults; every item can be changed on the command line
or in the optional ``trainer:`` section of the configuration, see docs/training.md):

  * Adam, learning rate ``lr`` (1e-2), MultiStepLR with ``lr_milestones`` (epochs 15..24) and
    ``lr_gamma`` (0.25): the learning rate is multiplied by 0.25 after each of those epochs;
  * ``epochs`` (20) epochs; batch size ``batch_size_init`` (1024) at epoch 0, doubled at epochs
    1, 3, 6 and 10, with (64, 16, 4, 1, 0) ghost batches;
  * the input normalisation of every embedder is frozen from the training set before epoch 0;
  * at epoch ``freeze_epoch`` (15) every parameter except the output heads is frozen;
  * per-class weight normalisation (``class_norm``): the event weights of every class sum to 1 over
    the training set; weights are clamped at >= 0 inside the loss.

Loss = weighted cross entropy over ``spec.classes`` (0 = rest, 1 = first output, 2 = second, ...) on
the main logits, plus, for every further output ``o``, ``head_weights.get(o, 1.0)`` times a weighted
binary cross entropy of its head against ``label == class index of o``. The view (candidate
selection) scores are not supervised.

For each held-out fold k this writes, in ``--output-dir``:
    model_fold{k}.pkl          checkpoint: state dict, configuration, classes and metadata
    history_fold{k}.json       per-epoch training loss, and validation loss/AUC on --exclude-folds
    predictions_fold{k}.h5     predictions on the held-out fold (layout in docs/evaluation.md)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import save_checkpoint
from .data.h5 import H5Dataset
from .evaluate import Batches, count_parameters, predict, write_predictions
from .models.config import ConfigError, NetworkSpec, parse_config, read_config_dict
from .models.network import CoReNet

HEAD_PREFIXES = ("head_main.", "head_reduce.", "aux_heads.")   # trainable after the freeze epoch
MIN_GHOST_BATCH_SIZE = 8       # single-pixel GBNs go NaN with one-event ghost batches

# training-recipe options: name -> default. Settable in the `trainer:` section of the configuration;
# a command-line flag overrides the file.
RECIPE_DEFAULTS = {
    "epochs": 20,
    "freeze_epoch": 15,
    "lr": 1e-2,
    "lr_milestones": list(range(15, 25)),
    "lr_gamma": 0.25,
    "batch_size_init": 1024,
    "class_norm": True,
    "seed": 42,
    "rest_order": "pt",
}


def batch_schedule(epoch: int, bs_init: int) -> tuple[int, int]:
    """(batch size, ghost batches) of an epoch: the batch size doubles at epochs 1, 3, 6 and 10."""
    stage = 0 if epoch == 0 else 1 if epoch < 3 else 2 if epoch < 6 else 3 if epoch < 10 else 4
    bs = bs_init << stage
    n_gb = (64, 16, 4, 1, 0)[stage]
    return bs, min(n_gb, max(1, bs // MIN_GHOST_BATCH_SIZE))     # no-op for bs_init >= 512


def freeze_heads(model: CoReNet) -> int:
    """Freeze everything but the output heads (head_main, head_reduce, aux_heads); returns #trainable."""
    n = 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(HEAD_PREFIXES)
        n += int(param.requires_grad) * param.numel()
    return n


def normalize_class_weights(labels: np.ndarray, weights: torch.Tensor, idx: np.ndarray, n_classes: int) -> torch.Tensor:
    """Per-event weights with every class summing to 1 over ``idx``."""
    w = weights.clone()
    for c in range(n_classes):
        sel = idx[labels[idx] == c]
        total = float(w[sel].sum())
        if total > 0:
            w[sel] /= total
    return w


def weighted_auc(label: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    """ROC AUC (trapezoidal, weighted) of class 1 against everything else, without scikit-learn."""
    order = np.argsort(-score, kind="stable")
    y, w = label[order] == 1, np.clip(weight[order], 0, None)
    tps, fps = np.cumsum(w * y), np.cumsum(w * ~y)
    if tps[-1] <= 0 or fps[-1] <= 0:
        return float("nan")
    tpr = np.concatenate([[0.0], tps / tps[-1]])
    fpr = np.concatenate([[0.0], fps / fps[-1]])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(tpr, fpr))


def aux_targets(spec: NetworkSpec) -> dict[str, int]:
    """Auxiliary head name -> the class label it discriminates (classes = ['rest'] + outputs)."""
    return {o.name: i + 1 for i, o in enumerate(spec.outputs) if i > 0}


def validation_metrics(model: CoReNet, data: Batches, val_idx: np.ndarray, labels: np.ndarray,
                       weights: torch.Tensor) -> dict:
    """Weighted cross entropy and signal-vs-rest AUC on the validation fold(s).

    Side-effect free: the torch RNG state is saved and restored so the training stream is identical
    to a run without validation; the caller restores ghost batches.
    """
    rng_state = torch.get_rng_state()
    was_training = model.training
    probs, _, _ = predict(model, data, val_idx)
    torch.set_rng_state(rng_state)
    if was_training:
        model.train()
    label = labels[val_idx]
    w = np.clip(weights.numpy()[val_idx], 0, None)
    p = np.clip(probs[np.arange(len(label)), label], 1e-12, None)
    out = {"val_events": int(len(label)), "val_loss": float(-(w * np.log(p)).sum() / max(w.sum(), 1e-12))}
    if 0 < int((label == 1).sum()) < len(label):
        out["val_auc"] = weighted_auc(label, probs[:, 1], w)
    return out


def loss_of_batch(model: CoReNet, batch: dict, label: torch.Tensor, w: torch.Tensor,
                  aux: dict[str, int], head_weights: dict[str, float]) -> torch.Tensor:
    """Weighted CE on the class logits + sum_o head_weight[o] * weighted BCE of the auxiliary heads."""
    out = model(batch)
    w_sum = w.sum()
    loss = (F.cross_entropy(out["logits"], label, reduction="none") * w).sum() / w_sum
    for name, cls in aux.items():
        target = (label == cls).to(out["heads"][name].dtype)
        bce = F.binary_cross_entropy_with_logits(out["heads"][name], target, reduction="none")
        loss = loss + head_weights.get(name, 1.0) * (bce * w).sum() / w_sum
    return loss


def train_fold(ds: H5Dataset, spec: NetworkSpec, config: dict, fold: int, args, device) -> Path:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    n_classes = len(spec.classes)
    labels = ds.labels.numpy()
    train_idx, test_idx = ds.fold_indices(fold)
    val_idx = train_idx[:0]
    if args.exclude_folds:
        keep = ~np.isin(ds.folds.numpy()[train_idx], args.exclude_folds)
        val_idx, train_idx = train_idx[~keep], train_idx[keep]
    weights = normalize_class_weights(labels, ds.weights, train_idx, n_classes) if args.class_norm else ds.weights
    # validation weights are class-normalised over the validation events, so val_loss is on the same
    # (class-balanced) footing as the training loss; training itself uses `weights` only
    val_weights = (normalize_class_weights(labels, ds.weights, val_idx, n_classes)
                   if (args.class_norm and len(val_idx)) else ds.weights)
    logging.info(f"fold {fold}: {len(train_idx):,} training / {len(val_idx):,} validation / {len(test_idx):,} held-out events")

    model = CoReNet(spec, device=str(device)).to(device)
    logging.info(f"{args.config}: classes {spec.classes}, {count_parameters(model):,} parameters")
    data = Batches(ds, device)
    w_dev = weights.to(device)
    aux, head_weights = aux_targets(spec), dict(spec.training.head_weights or {})

    model.train()
    with torch.no_grad():
        for _, batch in data.iter(train_idx, 131072):
            model.updateMeanStd(batch)
    model.initMeanStd()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(args.lr_milestones), gamma=args.lr_gamma)
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe = {k: getattr(args, k) for k in RECIPE_DEFAULTS}
    history, history_path = [], out_dir / f"history_fold{fold}.json"
    for epoch in range(args.epochs):
        bs, n_gb = batch_schedule(epoch, args.batch_size_init)
        model.setGhostBatches(n_gb)
        if epoch == args.freeze_epoch:
            n_trainable = freeze_heads(model)
            logging.info(f"fold {fold}: frozen at epoch {epoch}, {n_trainable:,} trainable head parameters")
        model.train()
        loss_sum, w_sum = 0.0, 0.0
        for chunk, batch in data.iter(train_idx, bs, shuffle=True, drop_last=True):
            sel = torch.as_tensor(chunk, device=device)
            w = w_dev[sel].clamp(min=0)
            if w.sum() <= 0:
                continue
            loss = loss_of_batch(model, batch, data.labels[sel], w, aux, head_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * w.sum().item()
            w_sum += w.sum().item()
        scheduler.step()
        if args.save_every_epoch:
            save_checkpoint(out_dir / f"model_fold{fold}_epoch{epoch:02d}.pkl", model, config, args.config,
                            {"fold": fold, "epoch": epoch, "exclude_folds": args.exclude_folds, "input": str(args.input), **recipe})
        entry = {"epoch": epoch, "batch_size": bs, "ghost_batches": n_gb, "lr": float(optimizer.param_groups[0]["lr"]),
                 "train_loss": loss_sum / max(w_sum, 1e-12)}
        if len(val_idx):
            entry.update(validation_metrics(model, data, val_idx, labels, val_weights))
            model.setGhostBatches(n_gb)                      # predict() set them to 0
        history.append(entry)
        history_path.write_text(json.dumps({"fold": fold, "exclude_folds": args.exclude_folds, "config": str(args.config),
                                            "classes": spec.classes, "recipe": recipe,
                                            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                                            "history": history}, indent=1))
        logging.info(f"fold {fold} | epoch {epoch:02d} | bs {bs:6d} | gbn {n_gb:2d} | lr {optimizer.param_groups[0]['lr']:.2e} | loss {entry['train_loss']:.5f}"
                     + (f" | val loss {entry['val_loss']:.5f} | val auc {entry.get('val_auc', float('nan')):.5f}" if len(val_idx) else ""))

    ckpt = out_dir / f"model_fold{fold}.pkl"
    save_checkpoint(ckpt, model, config, args.config,
                    {"fold": fold, "epoch": args.epochs - 1, "exclude_folds": args.exclude_folds, "input": str(args.input), **recipe})
    probs, views, heads = predict(model, data, test_idx)
    write_predictions(out_dir / f"predictions_fold{fold}.h5", ds, test_idx, probs, views, heads, spec,
                      attrs={"checkpoint": str(ckpt), "config": str(Path(args.config).resolve()),
                             "n_parameters": count_parameters(model), "fold": fold})
    logging.info(f"fold {fold}: saved {ckpt.name}, {history_path.name} and predictions ({time.time() - t0:.0f}s)")
    return ckpt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="corenet-train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="network configuration (.yaml or .json)")
    p.add_argument("--input", required=True, help="HDF5 file with all folds (docs/data_format.md)")
    p.add_argument("--output-dir", default="runs/corenet")
    p.add_argument("--fold", type=int, default=-1, help="held-out (test) fold, or -1 for every fold in turn")
    p.add_argument("--exclude-folds", type=int, nargs="*", default=[],
                   help="folds left out of training and used for per-epoch validation")
    g = p.add_argument_group("training recipe (default: the `trainer:` section of --config, else the built-in value)")
    g.add_argument("--epochs", type=int, help=f"number of epochs [{RECIPE_DEFAULTS['epochs']}]")
    g.add_argument("--freeze-epoch", type=int, help=f"epoch at which all but the output heads are frozen [{RECIPE_DEFAULTS['freeze_epoch']}]")
    g.add_argument("--lr", type=float, help=f"Adam learning rate [{RECIPE_DEFAULTS['lr']}]")
    g.add_argument("--lr-milestones", type=int, nargs="*", help="epochs after which the learning rate is multiplied by --lr-gamma [15 .. 24]")
    g.add_argument("--lr-gamma", type=float, help=f"learning-rate decay factor [{RECIPE_DEFAULTS['lr_gamma']}]")
    g.add_argument("--batch-size-init", type=int, help=f"epoch-0 batch size; doubles at epochs 1, 3, 6, 10 [{RECIPE_DEFAULTS['batch_size_init']}]")
    g.add_argument("--class-norm", action=argparse.BooleanOptionalAction, default=None,
                   help="normalise the weights of every class to 1 over the training set [on]")
    g.add_argument("--seed", type=int, help=f"random seed (the seed of fold k is seed + k) [{RECIPE_DEFAULTS['seed']}]")
    g.add_argument("--rest-order", choices=["pt", "selector"], help="ordering of the 'rest' selector [pt]")
    p.add_argument("--save-every-epoch", action="store_true", help="also save model_fold{k}_epoch{e}.pkl after every epoch")
    p.add_argument("--device", default="cuda", help="torch device; falls back to cpu when CUDA is unavailable")
    return p


def _check_recipe(r: dict) -> None:
    """Types and ranges of the training-recipe options."""
    is_int = lambda x: isinstance(x, int) and not isinstance(x, bool)          # noqa: E731
    is_num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)  # noqa: E731
    rules = {
        "epochs": (lambda x: is_int(x) and x >= 1, "an integer >= 1"),
        "freeze_epoch": (lambda x: is_int(x) and x >= 0, "an integer >= 0"),
        "lr": (lambda x: is_num(x) and x > 0, "a number > 0"),
        "lr_milestones": (lambda x: isinstance(x, list) and all(is_int(e) and e >= 0 for e in x), "a list of integers >= 0"),
        "lr_gamma": (lambda x: is_num(x) and x > 0, "a number > 0"),
        "batch_size_init": (lambda x: is_int(x) and x >= 1, "an integer >= 1"),
        "class_norm": (lambda x: isinstance(x, bool), "true or false"),
        "seed": (lambda x: is_int(x), "an integer"),
        "rest_order": (lambda x: x in ("pt", "selector"), "'pt' or 'selector'"),
    }
    for key, (ok, what) in rules.items():
        if key in r and not ok(r[key]):
            raise ConfigError(f"trainer.{key} must be {what}, got {r[key]!r}")


def resolve_recipe(args, config: dict):
    """Fill the recipe options of ``args``: command line > ``trainer:`` section > built-in default."""
    section = config.get("trainer", {}) or {}
    if not isinstance(section, dict):
        raise ConfigError("trainer must be a mapping")
    unknown = set(section) - set(RECIPE_DEFAULTS)
    if unknown:
        raise ConfigError(f"trainer: unknown keys {sorted(unknown)} (known: {', '.join(RECIPE_DEFAULTS)})")
    for key, default in RECIPE_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, section.get(key, default))
    _check_recipe({k: getattr(args, k) for k in RECIPE_DEFAULTS})
    return args


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = read_config_dict(args.config)
    spec = parse_config(config)
    resolve_recipe(args, config)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    ds = H5Dataset(args.input, spec, rest_order=args.rest_order)
    logging.info(f"{args.input}: {len(ds):,} events, device={device}, recipe "
                 + ", ".join(f"{k}={getattr(args, k)}" for k in RECIPE_DEFAULTS))
    folds = sorted(set(ds.folds.tolist())) if args.fold < 0 else [args.fold]
    for fold in folds:
        train_fold(ds, spec, config, fold, args, device)


if __name__ == "__main__":
    main()
