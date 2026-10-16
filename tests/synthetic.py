"""Small synthetic files in the example H5 layout (separable signal, 5 folds), for the tests."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

N_EVENTS = 2000
N_FOLDS = 5


def _jets(rng, n, n_slots, n_present, boost):
    """(features dict, mask) of a padded jet collection; signal events get harder jets."""
    pt = rng.uniform(30, 150, (n, n_slots)).astype(np.float32) + boost[:, None].astype(
        np.float32
    )
    eta = rng.uniform(-2.4, 2.4, (n, n_slots)).astype(np.float32)
    phi = rng.uniform(-math.pi, math.pi, (n, n_slots)).astype(np.float32)
    mass = rng.uniform(5, 25, (n, n_slots)).astype(np.float32)
    btag = (rng.uniform(0, 1, (n, n_slots)) < 0.4).astype(np.float32)
    mask = np.zeros((n, n_slots), dtype=bool)
    for i, k in enumerate(n_present):
        mask[i, :k] = True
    for a in (pt, eta, phi, mass, btag):
        a[~mask] = 0.0
    return {"pt": pt, "eta": eta, "phi": phi, "mass": mass, "btag": btag}, mask


def write_synthetic(
    path: Path, topology: str, seed: int = 0, n: int = N_EVENTS
) -> Path:
    """A small file in the layout of examples/data/{hh4b,bbww}_small.h5."""
    rng = np.random.default_rng(seed)
    if topology == "hh4b":
        label = (rng.uniform(size=n) < 0.3).astype(np.int64)  # 0 rest/QCD, 1 signal
    else:
        label = np.where(rng.uniform(size=n) < 0.3, 1, 2).astype(
            np.int64
        )  # 1 signal, 2 ttbar
    signal = label == 1
    n_present = rng.integers(4, 9, size=n)
    feats, mask = _jets(rng, n, 12, n_present, boost=np.where(signal, 60.0, 0.0))
    group = "Source" if topology == "hh4b" else "Jets"
    with h5py.File(path, "w") as f:
        for k, v in feats.items():
            f.create_dataset(f"INPUTS/{group}/{k}", data=v)
        f.create_dataset(f"INPUTS/{group}/MASK", data=mask)
        if topology == "bbww":
            f.create_dataset(
                "INPUTS/Lepton/pt",
                data=(25 + 60 * rng.uniform(size=n) + 30 * signal).astype(np.float32),
            )
            f.create_dataset(
                "INPUTS/Lepton/eta", data=rng.uniform(-2.4, 2.4, n).astype(np.float32)
            )
            f.create_dataset(
                "INPUTS/Lepton/phi",
                data=rng.uniform(-math.pi, math.pi, n).astype(np.float32),
            )
            f.create_dataset(
                "INPUTS/Lepton/mass", data=np.full(n, 0.105, dtype=np.float32)
            )
            isE = (rng.uniform(size=n) < 0.5).astype(np.float32)
            f.create_dataset("INPUTS/Lepton/isE", data=isE)
            f.create_dataset("INPUTS/Lepton/isM", data=1 - isE)
            f.create_dataset(
                "INPUTS/Met/pt", data=(20 + 80 * rng.uniform(size=n)).astype(np.float32)
            )
            f.create_dataset(
                "INPUTS/Met/phi",
                data=rng.uniform(-math.pi, math.pi, n).astype(np.float32),
            )
        f.create_dataset("EVENT/label", data=label)
        f.create_dataset(
            "EVENT/fold", data=rng.integers(0, N_FOLDS, size=n).astype(np.int64)
        )
        f.create_dataset("EVENT/event", data=np.arange(n, dtype=np.int64))
        w = np.ones(n, dtype=np.float32)
        for c in np.unique(label):  # per-class sum = 1, as the real files
            w[label == c] /= float((label == c).sum())
        f.create_dataset("EVENT/weight", data=w)
    return path
