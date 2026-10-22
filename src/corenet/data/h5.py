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
(``EVENT/weight``), ``folds`` and ``events`` (``EVENT/event``). The cross-validation fold of an event
is ``EVENT/fold`` by default, whose ids must be exactly 0..K-1 for K folds; with ``n_folds`` it is
``fold_of(event, n_folds, fold_seed)`` = ``squares(event) % n_folds`` (``corenet.data.folds``)
computed from the event numbers, and the file needs no ``EVENT/fold`` column. A file with neither
loads without folds (``has_folds`` False, every fold -1): evaluating or exporting on every event works,
selecting a fold (``fold_indices``: training, evaluating one fold) raises. No generator-level
information is read: the roles are chosen from reconstructed quantities only.

A collection's ``features`` are exactly the channels of its tensors. A selector reads the column it
needs (its column and ``pt``) from the file on its own, whether or not it is a feature; such a column
is used for the selection only (``columns_to_read``). A column listed under ``features`` is always a
network input, ``btag`` included. Columns are read at the file's dtype and sorted at that precision;
they are cast to float32 only when the input tensors are built.

    ds = H5Dataset("examples/data/bbww_small.h5", load_config("examples/bbww/bbww.yaml"))
    out = model(ds.inputs)

Selectors, applied to the collection's present members in configuration order:

    <column> n          the n members with the highest value of <column> (any dataset of the collection's
                        group: btag, pt, mass, ...), ties broken by the highest pT, then file order
    <column> n lowest   the n members with the lowest value of <column>, ties again by the highest pT
    rest n              up to n of the members no earlier role consumed; ``rest_order`` decides their order:
                        "pt" (default) or "selector", which continues the ordering of the preceding role on
                        the same collection.

``nSelJets`` and ``HT`` are always computed from the first collection with ``max > 1`` (number of
present members and scalar sum of their pT); every other event scalar is read from ``EVENT/<name>``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.config import REST, Collection, NetworkSpec, RoleSpec
from .folds import FOLD_SEED, fold_of, fold_rule

# collection name (lower case) -> group names accepted in INPUTS, in order of preference
GROUP_ALIASES = {
    "jets": ("Jets", "Source"),
    "jet": ("Jets", "Source"),
    "lepton": ("Lepton", "Leptons"),
    "met": ("Met", "MET"),
}
COMPUTED_EVENT = ("nSelJets", "HT")
LABEL_KEYS = (
    "EVENT/label",
    "CLASSIFICATIONS/EVENT/signal",
)  # first one present is used
WEIGHT_KEY, FOLD_KEY, EVENT_KEY = "EVENT/weight", "EVENT/fold", "EVENT/event"
FOLD_RULE_EVENT, FOLD_RULE_COLUMN, FOLD_RULE_NONE = (
    "squares(event) % n_folds",
    "EVENT/fold",
    "none",
)  # H5Dataset.fold_rule; "none": the file has no EVENT/fold and no n_folds was given
NO_FOLD = -1  # the fold of every event of a file loaded without folds


# ------------------------------------------------------------------ what the loader reads (shared with corenet-validate)
def group_candidates(name: str) -> tuple[str, ...]:
    """INPUTS group names accepted for a collection, in order of preference (case-insensitive)."""
    return (name, *GROUP_ALIASES.get(name.lower(), ()))


def used_collections(spec: NetworkSpec) -> list[Collection]:
    """Collections the loader reads: those used by a role, and every max-1 collection (a leaf)."""
    return [
        c
        for c in spec.collections.values()
        if c.max == 1 or any(r.collection == c.name for r in spec.roles.values())
    ]


def columns_to_read(spec: NetworkSpec, coll: Collection) -> list[str]:
    """Datasets read from a collection's group: its features, then the selector columns that are not features."""
    return spec.columns_to_read(coll.name)


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
    raise KeyError(
        f"collection {name!r}: no INPUTS group named {name!r} "
        f"or {list(GROUP_ALIASES.get(name.lower(), ()))} in the file (has {sorted(available.values())})"
    )


def _sortable(a: np.ndarray) -> np.ndarray:
    """A column as read from the file, in a dtype that sorts exactly and can be negated: floats and
    signed integers unchanged, bool -> int8, unsigned integers -> int64."""
    if a.dtype.kind == "b":
        return a.astype(np.int8)
    if a.dtype.kind == "u":
        return a.astype(np.int64)
    return a


def _read_collection(
    f: h5py.File, coll: Collection, columns: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """(columns, present) of one collection, both shaped (N, slots in the file); ``columns`` are the
    features plus the selector columns (``columns_to_read``), at the file's dtype (``_sortable``) so that
    the selectors sort at full precision; ``_stack`` casts to float32."""
    group = f["INPUTS"][resolve_group(f, coll.name)]
    missing = [k for k in columns if k not in group]
    if missing:
        raise KeyError(
            f"collection {coll.name!r} (INPUTS/{group.name.split('/')[-1]}): missing columns {missing} "
            f"(features {coll.features}, selector columns {[c for c in columns if c not in coll.features]})"
        )
    feats = {k: np.atleast_2d(_sortable(group[k][...]).T).T for k in columns}
    n, slots = feats[columns[0]].shape
    if coll.max > 1 and slots < coll.max:
        raise ValueError(
            f"collection {coll.name!r} has {slots} slots in the file, configuration asks for max {coll.max}"
        )
    present = (
        group["MASK"][...].astype(bool)
        if "MASK" in group
        else np.ones((n, slots), dtype=bool)
    )
    return feats, present


def _selector_keys(
    role: RoleSpec, feats: dict[str, np.ndarray]
) -> tuple[np.ndarray, ...]:
    """Sort keys (smaller = preferred) of an explicit selector: the column, negated unless ``lowest``, then
    -pt as the tie breaker. A lexicographic sort, so the column's scale does not matter."""
    col = feats[role.selector]
    return (col if role.ascending else -col, -feats["pt"])


def _take(
    keys: tuple[np.ndarray, ...], avail: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """(indices, valid) of the n preferred available members: (N, n) each. Available members first, then
    by ``keys`` in order (the first key dominates), remaining ties in file order (stable)."""
    order = np.lexsort((*reversed(keys), ~avail), axis=1)[
        :, :n
    ]  # lexsort: the last key is the primary one
    rows = np.arange(len(avail))[:, None]
    return order, avail[rows, order]


class H5Dataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        spec: NetworkSpec,
        rest_order: str = "pt",
        n_folds: int | None = None,
        fold_seed: int = FOLD_SEED,
    ):
        """``n_folds``: use fold = squares(event) % n_folds (``fold_of``, seed ``fold_seed``) instead of the
        ``EVENT/fold`` column. Without both, the file loads without folds (``has_folds`` is False)."""
        if rest_order not in ("pt", "selector"):
            raise ValueError(
                f"rest_order must be 'pt' or 'selector', got {rest_order!r}"
            )
        self.spec, self.path, self.rest_order = spec, Path(path), rest_order
        self._taken: dict[
            str, np.ndarray
        ] = {}  # per collection: members already consumed by a role
        self._last_keys: dict[
            str, tuple[np.ndarray, ...]
        ] = {}  # per collection: sort keys of the last explicit selector
        inputs: dict[str, torch.Tensor] = {}

        with h5py.File(path, "r") as f:
            collections = {
                c.name: _read_collection(f, c, columns_to_read(spec, c))
                for c in used_collections(spec)
            }
            for role in spec.roles.values():
                feats, present = collections[role.collection]
                inputs[role.name] = self._role_tensor(
                    role, spec.collections[role.collection], feats, present
                )
            for c in spec.collections.values():
                if c.max == 1 and c.name not in inputs and c.name in collections:
                    feats, present = collections[c.name]
                    inputs[c.name] = self._stack(
                        c, feats, np.zeros((len(present), 1), dtype=np.int64), present
                    )
            inputs["event"] = self._event_tensor(f, collections)
            n = len(inputs["event"])

            self.labels = torch.from_numpy(self._labels(f, n))
            self.weights = (
                torch.from_numpy(f[WEIGHT_KEY][...].astype(np.float32))
                if WEIGHT_KEY in f
                else torch.ones(n, dtype=torch.float32)
            )
            self.events = (
                torch.from_numpy(f[EVENT_KEY][...].astype(np.int64))
                if EVENT_KEY in f
                else torch.arange(n, dtype=torch.int64)
            )
            self.fold_seed = int(fold_seed)
            self.folds, self.n_folds, self.fold_rule = self._folds(f, n_folds, n)
        self.inputs = inputs

    @property
    def has_folds(self) -> bool:
        """False when the file has no ``EVENT/fold`` and no ``n_folds`` was given: every fold is NO_FOLD."""
        return self.fold_rule != FOLD_RULE_NONE

    def require_folds(self, what: str = "selecting a fold") -> None:
        """Raise KeyError with the remedy when the file was loaded without folds."""
        if not self.has_folds:
            raise KeyError(
                f"{self.path.name}: no {FOLD_KEY} dataset; {what} needs the fold of every event: pass --n-folds K "
                f"(H5Dataset(n_folds=K)) to use fold = squares(event) % K from {EVENT_KEY}"
            )

    def _folds(
        self, f: h5py.File, n_folds: int | None, n: int
    ) -> tuple[torch.Tensor, int, str]:
        """(fold of every event, number of folds K, rule): with ``n_folds``, ``fold_of(EVENT/event, n_folds,
        self.fold_seed)``; else ``EVENT/fold``, whose ids must be exactly 0..K-1; else (no column) every fold
        is NO_FOLD, K = 0 and the rule is FOLD_RULE_NONE."""
        if n_folds is not None:
            rule = fold_rule(n_folds, self.fold_seed)
            if EVENT_KEY not in f:
                raise KeyError(
                    f"{self.path.name}: no {EVENT_KEY} dataset; fold = {rule} needs the event numbers"
                )
            if FOLD_KEY in f:
                logging.info(
                    f"{self.path.name}: {FOLD_KEY} ignored, folds = {rule} from {EVENT_KEY}"
                )
            folds = fold_of(f[EVENT_KEY][...], n_folds, self.fold_seed)
            return torch.from_numpy(folds), n_folds, FOLD_RULE_EVENT
        if FOLD_KEY not in f:
            return torch.full((n,), NO_FOLD, dtype=torch.int64), 0, FOLD_RULE_NONE
        folds = f[FOLD_KEY][...].astype(np.int64)
        ids = np.unique(folds)
        if not np.array_equal(ids, np.arange(len(ids))):
            shown = ids[:12].tolist() + (["..."] if len(ids) > 12 else [])
            raise ValueError(
                f"{self.path.name}: {FOLD_KEY} holds the fold ids {shown}; the ids of K folds must be exactly "
                f"0..K-1 (--exclude-folds next = fold (k + 1) % K); renumber the column or pass --n-folds K"
            )
        return torch.from_numpy(folds), len(ids), FOLD_RULE_COLUMN

    # ---------------------------------------------------------------- construction
    def _role_tensor(
        self,
        role: RoleSpec,
        coll: Collection,
        feats: dict[str, np.ndarray],
        present: np.ndarray,
    ) -> torch.Tensor:
        taken = self._taken.setdefault(role.collection, np.zeros_like(present))
        if role.selector == REST:
            keys = (
                self._last_keys.get(role.collection)
                if self.rest_order == "selector"
                else None
            )
            keys = (-feats["pt"],) if keys is None else keys
        else:
            keys = _selector_keys(role, feats)
            self._last_keys[role.collection] = keys
        idx, valid = _take(keys, present & ~taken, role.n)
        rows = np.arange(len(present))[:, None]
        for j in range(role.n):  # mark the chosen members as consumed
            taken[rows[:, 0], idx[:, j]] |= valid[:, j]
        return self._stack(coll, feats, idx, valid)

    @staticmethod
    def _stack(
        coll: Collection,
        feats: dict[str, np.ndarray],
        idx: np.ndarray,
        valid: np.ndarray,
    ) -> torch.Tensor:
        """(N, len(features) + 1, n): features in configuration order, presence flag last, padding -1;
        float32 (the columns are cast here). Columns read only for a selector are not stacked."""
        n, slots = idx.shape
        rows = np.arange(n)[:, None]
        x = np.full((n, len(coll.features) + 1, slots), -1.0, dtype=np.float32)
        for i, k in enumerate(coll.features):
            x[:, i] = np.where(valid, feats[k][rows, idx], -1.0).astype(np.float32)
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
                out[:, i] = (
                    present.sum(1)
                    if name == "nSelJets"
                    else (feats["pt"] * present).sum(1)
                )
            elif where in f:
                out[:, i] = f[where][...].astype(np.float32)
            else:
                raise KeyError(
                    f"event scalar {name!r}: no {where} in {self.path.name} and it is not computable"
                )
        return torch.from_numpy(out)

    def _labels(self, f: h5py.File, n: int) -> np.ndarray:
        key = LABEL_KEYS[0] if LABEL_KEYS[0] in f else LABEL_KEYS[1]
        if key not in f:
            return np.zeros(n, dtype=np.int64)
        lab = f[key][...].astype(np.int64)
        bad = np.unique(lab[(lab < 0) | (lab > len(self.spec.outputs))])
        if len(bad):
            raise ValueError(
                f"{self.path.name}: labels {bad.tolist()} outside 0..{len(self.spec.outputs)} "
                f"({len(self.spec.outputs)} outputs: {[o.name for o in self.spec.outputs]})"
            )
        return lab

    # ---------------------------------------------------------------- Dataset API
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        # slices (ds[:512]) cannot go through torch.as_tensor; expand them to explicit indices
        index = (
            torch.arange(*idx.indices(len(self)))
            if isinstance(idx, slice)
            else torch.as_tensor(idx)
        )
        return {
            "index": index,
            "inputs": {k: v[idx] for k, v in self.inputs.items()},
            "label": self.labels[idx],
            "weight": self.weights[idx],
            "fold": self.folds[idx],
        }

    def fold_indices(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """(train_idx, val_idx) holding out ``fold``; raises KeyError on a file loaded without folds."""
        self.require_folds(f"holding out fold {fold}")
        val = (self.folds == fold).numpy()
        return np.flatnonzero(~val), np.flatnonzero(val)


__all__ = [
    "COMPUTED_EVENT",
    "EVENT_KEY",
    "FOLD_KEY",
    "FOLD_RULE_COLUMN",
    "FOLD_RULE_EVENT",
    "FOLD_RULE_NONE",
    "GROUP_ALIASES",
    "LABEL_KEYS",
    "NO_FOLD",
    "WEIGHT_KEY",
    "H5Dataset",
    "columns_to_read",
    "event_source",
    "fold_of",
    "group_candidates",
    "resolve_group",
    "used_collections",
]
