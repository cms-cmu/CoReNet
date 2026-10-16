"""Checkpoint format: a ``torch.save``-d dict

{"model":    state dict of CoReNet,
 "config":   the configuration mapping the network was built from (so a checkpoint is self-contained),
 "config_path": path of the configuration file at training time,
 "classes":  ["rest", <outputs>...],
 "metadata": {"fold", "epoch", "exclude_folds", "input", <training recipe>}}
"""

from __future__ import annotations

from pathlib import Path

import torch

from .models.config import NetworkSpec, parse_config, read_config_dict
from .models.network import CoReNet


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
        },
        path,
    )
    return path


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
    else:
        raise ValueError(
            f"{path}: the checkpoint stores no configuration; pass the configuration file explicitly"
        )
    spec = parse_config(cfg)
    model = CoReNet(spec, device=str(device)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.setGhostBatches(0)
    return model, spec, ckpt


__all__ = ["save_checkpoint", "load_checkpoint"]
