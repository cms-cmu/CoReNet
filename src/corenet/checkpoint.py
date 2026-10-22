"""Checkpoint format: a ``torch.save``-d dict

{"model":    state dict of CoReNet,
 "config":   the configuration mapping the network was built from (so a checkpoint is self-contained),
 "config_path": path of the configuration file at training time,
 "classes":  ["rest", <outputs>...],
 "metadata": {"fold", "epoch", "exclude_folds", "input", "n_folds", "fold_rule" ("EVENT/fold" or
             "squares(event) % n_folds"), "fold_seed", <training recipe>,
             "finetune_trainable_parameters"},
 "format":   CHECKPOINT_FORMAT}

``format`` 2: every feature of a collection is a network input. Checkpoints without ``format`` were
written when a ``btag`` feature was skipped by the network (used by the ``btag`` selector only); their
stored configuration is translated on loading by dropping ``btag`` from every ``features`` list, which
describes the trained network exactly.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch

from .data.folds import FOLD_SEED
from .data.h5 import FOLD_RULE_EVENT
from .models.config import NetworkSpec, parse_config, read_config_dict
from .models.network import CoReNet

CHECKPOINT_FORMAT = 2
LEGACY_SKIPPED = ("btag",)  # features the network ignored before format 2


def fold_rule_of(metadata: dict) -> tuple[int | None, int]:
    """(``n_folds``, ``fold_seed``) to pass to ``H5Dataset`` for a checkpoint: its ``n_folds`` and seed when
    it was trained with fold = squares(event) % n_folds, else (None, seed) for the ``EVENT/fold`` column."""
    seed = int(metadata.get("fold_seed", FOLD_SEED))
    if metadata.get("fold_rule") == FOLD_RULE_EVENT:
        return metadata.get("n_folds"), seed
    return None, seed


def save_checkpoint(
    path, model: CoReNet, config: dict, config_path, metadata: dict
) -> Path:
    path = Path(path)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "config_path": str(Path(config_path).resolve()),
            "classes": model.spec.classes,
            "metadata": metadata,
            "format": CHECKPOINT_FORMAT,
        },
        path,
    )
    return path


def legacy_config(cfg: dict) -> tuple[dict, list[str]]:
    """The configuration of a pre-format-2 checkpoint as the current code reads it: ``btag`` removed from
    every ``features`` list (the network never used it). Returns (config, names of the edited collections)."""
    cfg = copy.deepcopy(cfg)
    edited = []
    for name, body in (cfg.get("objects") or {}).items():
        if isinstance(body, dict) and isinstance(body.get("features"), list):
            kept = [f for f in body["features"] if f not in LEGACY_SKIPPED]
            if kept != body["features"]:
                body["features"] = kept
                edited.append(name)
    return cfg, edited


def load_checkpoint(
    path, config=None, device="cpu"
) -> tuple[CoReNet, NetworkSpec, dict]:
    """(model in eval mode, spec, raw checkpoint dict).

    The network is built from ``config`` (a path) when given, else from the configuration stored in the
    checkpoint. The state dict is loaded strictly: the configuration must describe the trained network.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if config is not None:
        cfg = read_config_dict(config)
    elif isinstance(ckpt.get("config"), dict):
        cfg = ckpt["config"]
        if "format" not in ckpt:
            cfg, edited = legacy_config(cfg)
            if edited:
                logging.info(
                    f"{path}: checkpoint written before format {CHECKPOINT_FORMAT}; "
                    f"{list(LEGACY_SKIPPED)} dropped from the features of {edited} (the network never used them)"
                )
    else:
        raise ValueError(
            f"{path}: the checkpoint stores no configuration; pass the configuration file explicitly"
        )
    spec = parse_config(cfg)
    model = CoReNet(spec, device=str(device)).to(device)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        hint = ""
        if "format" not in ckpt and config is not None:
            hint = (
                f"; the checkpoint predates format {CHECKPOINT_FORMAT}, in which every feature is a network input: "
                f"if the network was trained with {list(LEGACY_SKIPPED)} listed under features, remove them from "
                f"the features of {config} (or load without --config, which does so automatically)"
            )
        raise RuntimeError(
            f"{path}: the state dict does not fit the network built from the configuration{hint}\n{e}"
        ) from e
    model.eval()
    model.setGhostBatches(0)
    return model, spec, ckpt


__all__ = [
    "CHECKPOINT_FORMAT",
    "fold_rule_of",
    "legacy_config",
    "load_checkpoint",
    "save_checkpoint",
]
