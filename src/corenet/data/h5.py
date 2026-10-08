"""
CoReNet input tensors from an HDF5 file, driven by a ``NetworkSpec`` (docs/Dataset.md).

One loader serves every topology: it reads the collections named in the configuration from
``INPUTS/<Group>``, applies the role selectors and produces exactly the ``inputs`` dict that
``CoReNet.forward`` expects,

    inputs[role]      (N, len(collection.features) + 1, n)  features in the collection's order,
                                                            presence flag (+1 / -1) as the LAST channel,
                                                            padded slots = -1 everywhere
    inputs["event"]   (N, len(spec.event))                  event scalars in the configured order

plus ``labels`` (``EVENT/label``, or ``CLASSIFICATIONS/EVENT/signal`` when absent), ``weights``
(``EVENT/weight``), ``folds`` (``EVENT/fold``) and ``events`` (``EVENT/event``). No generator-level
information is read: the roles are chosen from reconstructed quantities only.

    ds = H5Dataset("examples/data/bbww_small.h5", load_config("examples/bbww/bbww.yaml"))
    out = model(ds.inputs)

Selectors, applied to the collection's present members in configuration order:

    btag n   the n highest-btag members, ties broken by pT
    pt   n   the n highest-pT members
    rest n   up to n of the members no earlier role consumed; ``rest_order`` decides their order:
             "pt" (default) or "selector", which continues the ordering of the preceding role on the
             same collection.

``nSelJets`` and ``HT`` are always computed from the first collection with ``max > 1`` (number of
present members and scalar sum of their pT); every other event scalar is read from ``EVENT/<name>``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.config import Collection, NetworkSpec, RoleSpec


# collection name (lower case) -> group names accepted in INPUTS, in order of preference
GROUP_ALIASES = {
    "jets": ("Jets", "Source"),
    "jet": ("Jets", "Source"),
    "lepton": ("Lepton", "Leptons"),
    "met": ("Met", "MET"),
}
COMPUTED_EVENT = ("nSelJets", "HT")
BTAG_PRIORITY = 1e6                       # btag dominates the sort key, pT breaks ties
LABEL_KEYS = ("EVENT/label", "CLASSIFICATIONS/EVENT/signal")   # first one present is used
WEIGHT_KEY, FOLD_KEY, EVENT_KEY = "EVENT/weight", "EVENT/fold", "EVENT/event"


# ------------------------------------------------------------------ what the loader reads (shared with corenet-validate)
def group_candidates(name: str) -> tuple[str, ...]:
    """INPUTS group names accepted for a collection, in order of preference (case-insensitive)."""
    return (name, *GROUP_ALIASES.get(name.lower(), ()))


def used_collections(spec: NetworkSpec) -> list[Collection]:
    """Collections the loader reads: those used by a role, and every max-1 collection (a leaf)."""
    return [c for c in spec.collections.values() if c.max == 1 or any(r.collection == c.name for r in spec.roles.values())]


def event_source(spec: NetworkSpec, name: str) -> tuple[str, str]:
    """Where an event scalar comes from: ("computed", collection) for nSelJets / HT when a read
    collection has max > 1 (the first one), else ("dataset", "EVENT/<name>")."""
    seq = next((c.name for c in used_collections(spec) if c.max > 1), None)
    if name in COMPUTED_EVENT and seq is not None:
        return "computed", seq
    return "dataset", f"EVENT/{name}"


def resolve_group(f: h5py.File, name: str) -> str:
    """INPUTS group of a collection: its own name, then the aliases, case-insensitively."""
    available = {k.lower(): k for k in f["INPUTS"]} if "INPUTS" in f else {}
    for cand in group_candidates(name):
        if cand.lower() in available:
            return available[cand.lower()]
    raise KeyError(f"collection {name!r}: no INPUTS group named {name!r} "
                   f"or {list(GROUP_ALIASES.get(name.lower(), ()))} in the file (has {sorted(available.values())})")


def _read_collection(f: h5py.File, coll: Collection) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """(features, present) of one collection, both shaped (N, slots in the file)."""
    group = f["INPUTS"][resolve_group(f, coll.name)]
    missing = [k for k in coll.features if k not in group]
    if missing:
        raise KeyError(f"collection {coll.name!r} (INPUTS/{group.name.split('/')[-1]}): missing features {missing}")
    feats = {k: np.atleast_2d(group[k][...].astype(np.float32).T).T for k in coll.features}
    n, slots = feats[coll.features[0]].shape
    if coll.max > 1 and slots < coll.max:
        raise ValueError(f"collection {coll.name!r} has {slots} slots in the file, configuration asks for max {coll.max}")
    present = group["MASK"][...].astype(bool) if "MASK" in group else np.ones((n, slots), dtype=bool)
    return feats, present


def _selector_key(selector: str, feats: dict[str, np.ndarray]) -> np.ndarray:
    """Sort key (ascending = preferred first) of a selector."""
    if selector == "pt":
        return -feats["pt"]
    if selector == "btag":
        return -(feats["btag"] * BTAG_PRIORITY + feats["pt"])
    raise ValueError(f"unknown selector {selector!r}")


def _take(key: np.ndarray, avail: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(indices, valid) of the n preferred available members: (N, n) each, stable ties."""
    order = np.argsort(np.where(avail, key, np.inf), axis=1, kind="stable")[:, :n]
    rows = np.arange(len(key))[:, None]
    return order, avail[rows, order]


class H5Dataset(Dataset):
    def __init__(self, path: str | Path, spec: NetworkSpec, rest_order: str = "pt"):
        if rest_order not in ("pt", "selector"):
            raise ValueError(f"rest_order must be 'pt' or 'selector', got {rest_order!r}")
        self.spec, self.path, self.rest_order = spec, Path(path), rest_order
        self._taken: dict[str, np.ndarray] = {}        # per collection: members already consumed by a role
        self._last_key: dict[str, np.ndarray] = {}     # per collection: sort key of the last explicit selector
        inputs: dict[str, torch.Tensor] = {}

        with h5py.File(path, "r") as f:
            collections = {c.name: _read_collection(f, c) for c in used_collections(spec)}
            for role in spec.roles.values():
                feats, present = collections[role.collection]
                inputs[role.name] = self._role_tensor(role, spec.collections[role.collection], feats, present)
            for c in spec.collections.values():
                if c.max == 1 and c.name not in inputs and c.name in collections:
                    feats, present = collections[c.name]
                    inputs[c.name] = self._stack(c, feats, np.zeros((len(present), 1), dtype=np.int64), present)
            inputs["event"] = self._event_tensor(f, collections)
            n = len(inputs["event"])

            self.labels = torch.from_numpy(self._labels(f, n))
            self.weights = torch.from_numpy(f[WEIGHT_KEY][...].astype(np.float32)) if WEIGHT_KEY in f \
                else torch.ones(n, dtype=torch.float32)
            self.folds = torch.from_numpy(f[FOLD_KEY][...].astype(np.int64)) if FOLD_KEY in f \
                else torch.zeros(n, dtype=torch.int64)
            self.events = torch.from_numpy(f[EVENT_KEY][...].astype(np.int64)) if EVENT_KEY in f \
                else torch.arange(n, dtype=torch.int64)
        self.inputs = inputs

    # ---------------------------------------------------------------- construction
    def _role_tensor(self, role: RoleSpec, coll: Collection, feats: dict[str, np.ndarray], present: np.ndarray) -> torch.Tensor:
        taken = self._taken.setdefault(role.collection, np.zeros_like(present))
        if role.selector == "rest":
            key = self._last_key.get(role.collection) if self.rest_order == "selector" else None
            key = _selector_key("pt", feats) if key is None else key
        else:
            key = _selector_key(role.selector, feats)
            self._last_key[role.collection] = key
        idx, valid = _take(key, present & ~taken, role.n)
        rows = np.arange(len(present))[:, None]
        for j in range(role.n):                                  # mark the chosen members as consumed
            taken[rows[:, 0], idx[:, j]] |= valid[:, j]
        return self._stack(coll, feats, idx, valid)

    @staticmethod
    def _stack(coll: Collection, feats: dict[str, np.ndarray], idx: np.ndarray, valid: np.ndarray) -> torch.Tensor:
        """(N, len(features) + 1, n): features in configuration order, presence flag last, padding -1."""
        n, slots = idx.shape
        rows = np.arange(n)[:, None]
        x = np.full((n, len(coll.features) + 1, slots), -1.0, dtype=np.float32)
        for i, k in enumerate(coll.features):
            x[:, i] = np.where(valid, feats[k][rows, idx], -1.0)
        x[:, -1] = np.where(valid, 1.0, -1.0)
        return torch.from_numpy(x)

    def _event_tensor(self, f: h5py.File, collections: dict) -> torch.Tensor:
        seq = next((c.name for c in used_collections(self.spec) if c.max > 1), None)
        n = len(collections[seq][1]) if seq else len(f[WEIGHT_KEY])
        out = np.zeros((n, len(self.spec.event)), dtype=np.float32)
        for i, name in enumerate(self.spec.event):
            kind, where = event_source(self.spec, name)
            if kind == "computed":
                feats, present = collections[where]
                out[:, i] = present.sum(1) if name == "nSelJets" else (feats["pt"] * present).sum(1)
            elif where in f:
                out[:, i] = f[where][...].astype(np.float32)
            else:
                raise KeyError(f"event scalar {name!r}: no {where} in {self.path.name} and it is not computable")
        return torch.from_numpy(out)

    def _labels(self, f: h5py.File, n: int) -> np.ndarray:
        key = LABEL_KEYS[0] if LABEL_KEYS[0] in f else LABEL_KEYS[1]
        if key not in f:
            return np.zeros(n, dtype=np.int64)
        lab = f[key][...].astype(np.int64)
        bad = np.unique(lab[(lab < 0) | (lab > len(self.spec.outputs))])
        if len(bad):
            raise ValueError(f"{self.path.name}: labels {bad.tolist()} outside 0..{len(self.spec.outputs)} "
                             f"({len(self.spec.outputs)} outputs: {[o.name for o in self.spec.outputs]})")
        return lab

    # ---------------------------------------------------------------- Dataset API
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # slices (ds[:512]) cannot go through torch.as_tensor; expand them to explicit indices
        index = torch.arange(*idx.indices(len(self))) if isinstance(idx, slice) else torch.as_tensor(idx)
        return {
            "index": index,
            "inputs": {k: v[idx] for k, v in self.inputs.items()},
            "label": self.labels[idx],
            "weight": self.weights[idx],
            "fold": self.folds[idx],
        }

    def fold_indices(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """(train_idx, val_idx) holding out ``fold``."""
        val = (self.folds == fold).numpy()
        return np.flatnonzero(~val), np.flatnonzero(val)


__all__ = ["H5Dataset", "GROUP_ALIASES", "COMPUTED_EVENT", "LABEL_KEYS", "WEIGHT_KEY", "FOLD_KEY", "EVENT_KEY",
           "group_candidates", "used_collections", "event_source", "resolve_group"]
