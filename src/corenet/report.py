"""
corenet-report: one self-contained HTML page with the results and the input checks of a run directory.

    corenet-report --run-dir runs/hh4b --input examples/data/hh4b_all.h5 --output runs/hh4b/report.html

Per fold k the run directory is read for ``history_fold{k}.json``, ``model_fold{k}.pkl``,
``predictions_fold{k}.h5`` (the held-out fold) and, when present, ``eval_fold{k}.h5`` written by
``corenet-evaluate`` (with ``--fold -1`` it holds every event of the file, which gives the training and
validation folds their own curves). The input file provides the events, labels, weights, folds and, when
present, the generator-level ``TARGETS``.

Sections: run summary, training curves, classification (ROC, AUC, scores, confusion matrix, auxiliary
heads), hypotheses and pairing (view probabilities, pairing accuracy against ``TARGETS``), the inputs
as the network embeds them, the raw inputs, the event weights, further checks, and the list of what
could not be produced for the run. Every figure is embedded as PNG; the page needs no other file.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import base64
import datetime
import html
import io
import json
import logging
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import __version__  # noqa: E402
from .checkpoint import fold_rule_of, load_checkpoint  # noqa: E402
from .data.h5 import (  # noqa: E402
    FOLD_RULE_EVENT,
    H5Dataset,
    event_source,
    resolve_group,
)
from .models.config import NetworkSpec, parse_config, read_config_dict  # noqa: E402
from .models.network import CoReNet  # noqa: E402

SECTIONS = [
    ("summary", "Run summary"),
    ("training", "Training curves"),
    ("classification", "Classification"),
    ("hypotheses", "Hypotheses and pairing"),
    ("inputs", "Inputs as the network sees them"),
    ("raw", "Raw inputs"),
    ("weights", "Weights"),
    ("other", "Other checks"),
    ("gaps", "Not produced"),
]
SPLITS = ("test", "validation", "train")
SPLIT_STYLE = {"test": "-", "train": "--", "validation": "-."}
ROC_GRID = np.unique(np.concatenate([np.logspace(-4, 0, 300), np.linspace(0, 1, 300)]))
VIEW_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
NBINS = 40

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ----------------------------------------------------------------------------- small helpers
def esc(x) -> str:
    return html.escape(str(x))


def fmt(x, nd: int = 4) -> str:
    """A number for a table cell; '-' for None / NaN."""
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if not np.isfinite(xf):
        return "-"
    return f"{xf:.{nd}f}"


def mean_sd(values) -> str:
    """'mean ± sd' over the finite entries (sd with ddof=1; '-' for fewer than two)."""
    v = np.array([float(x) for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.4f}"
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"


def table(header: list[str], rows: list[list], caption: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{c if isinstance(c, str) else fmt(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figure(fig, caption: str = "", dpi: int = 100) -> str:
    """The figure as an inline PNG (the figure is closed)."""
    buf = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:  # a figure that cannot be laid out is still saved
        pass
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


def note(text: str) -> str:
    return f'<p class="note">{esc(text)}</p>'


def class_color(i: int) -> str:
    return "0.45" if i == 0 else f"C{(i - 1) % 10}"


def shared_bins(arrays, n: int = NBINS) -> np.ndarray:
    """Bin edges over the 0.1-99.9 percentiles of every array; unit bins for integer-valued data."""
    vals = [np.asarray(a, dtype=float) for a in arrays if a is not None and len(a)]
    vals = [v[np.isfinite(v)] for v in vals]
    vals = [v for v in vals if len(v)]
    if not vals:
        return np.linspace(0, 1, n + 1)
    allv = np.concatenate(vals)
    lo, hi = np.percentile(allv, [0.1, 99.9])
    if np.all(np.mod(allv, 1) == 0) and hi - lo <= 60:
        return np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    return np.linspace(lo, hi, n + 1)


def hist_panel(ax, series, bins, xlabel: str, logy: bool = False) -> None:
    """``series``: (label, values, weights, line style dict); weighted densities as step histograms."""
    for lab, v, w, st in series:
        v, w = np.asarray(v, dtype=float), np.asarray(w, dtype=float)
        ok = np.isfinite(v) & np.isfinite(w)
        if not ok.any() or w[ok].sum() <= 0:
            continue
        ax.hist(
            v[ok], bins=bins, weights=w[ok], density=True, histtype="step", label=lab, **st
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if logy:
        ax.set_yscale("log")


def grid_figure(n_panels: int, per_row: int = 4, w: float = 2.6, h: float = 2.1):
    rows = max(1, int(np.ceil(n_panels / per_row)))
    cols = min(per_row, max(1, n_panels))
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.axis("off")
    return fig, flat


def mean_curve(curves, grid=ROC_GRID):
    """(mean, sd) of the TPR of several (fpr, tpr) curves on ``grid`` (sd None for one curve)."""
    tprs = np.array([np.interp(grid, f, t) for f, t in curves])
    return tprs.mean(0), (tprs.std(0, ddof=1) if len(tprs) > 1 else None)


def roc_points(y: np.ndarray, s: np.ndarray, w: np.ndarray):
    from sklearn.metrics import roc_curve

    f, t, _ = roc_curve(y, s, sample_weight=w)
    return f, t


def auc_of(y: np.ndarray, s: np.ndarray, w: np.ndarray):
    from sklearn.metrics import roc_auc_score

    if y.all() or not y.any() or w.sum() <= 0:
        return None
    return float(roc_auc_score(y, s, sample_weight=w))


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ----------------------------------------------------------------------------- run files
@dataclass
class Split:
    index: np.ndarray  # rows of the input file
    label: np.ndarray
    weight: np.ndarray
    probs: np.ndarray
    views: dict[str, np.ndarray]
    heads: dict[str, np.ndarray]
    source: str


@dataclass
class FoldData:
    fold: int
    history: dict | None = None
    checkpoint: Path | None = None
    checkpoint_meta: dict = field(default_factory=dict)
    splits: dict[str, Split] = field(default_factory=dict)
    n_parameters: int | None = None

    @property
    def exclude_folds(self) -> list[int] | None:
        for src in (self.history, self.checkpoint_meta):
            if src and src.get("exclude_folds") is not None:
                return [int(e) for e in src["exclude_folds"]]
        return None

    @property
    def best_epoch(self) -> int | None:
        if not self.history:
            return None
        if self.history.get("best_epoch") is not None:
            return int(self.history["best_epoch"])
        aucs = [(e.get("val_auc"), e["epoch"]) for e in self.history.get("history", [])]
        aucs = [(a, ep) for a, ep in aucs if a is not None and np.isfinite(a)]
        return max(aucs)[1] if aucs else None


def read_prediction_file(path: Path):
    """(fold attribute, Split without a source) of a prediction file."""
    with h5py.File(path, "r") as f:
        views = {h: f["views"][h][...] for h in f["views"]} if "views" in f else {}
        heads = {k: f["heads"][k][...] for k in f["heads"]} if "heads" in f else {}
        split = Split(
            f["EVENT/index"][...].astype(np.int64),
            f["EVENT/label"][...].astype(np.int64),
            f["EVENT/weight"][...].astype(np.float64),
            f["class_probabilities"][...].astype(np.float64),
            views,
            heads,
            path.name,
        )
        fold = int(f.attrs.get("fold", -1))
        n_par = f.attrs.get("n_parameters")
    return fold, split, (int(n_par) if n_par is not None else None)


def sub_split(s: Split, sel: np.ndarray, source: str) -> Split:
    return Split(
        s.index[sel],
        s.label[sel],
        s.weight[sel],
        s.probs[sel],
        {k: v[sel] for k, v in s.views.items()},
        {k: v[sel] for k, v in s.heads.items()},
        source,
    )


def discover_folds(run_dir: Path, wanted: list[int] | None) -> list[int]:
    folds = set()
    for pattern in ("history_fold*.json", "model_fold*.pkl", "predictions_fold*.h5", "eval_fold*.h5"):
        for p in run_dir.glob(pattern):
            m = re.fullmatch(r"(?:history|model|predictions|eval)_fold(\d+)\.(?:json|pkl|h5)", p.name)
            if m:
                folds.add(int(m.group(1)))
    if wanted is not None:
        return sorted(wanted)
    return sorted(folds)


# ----------------------------------------------------------------------------- the report
class Report:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.input = Path(args.input)
        self.gaps: list[str] = []
        self.generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.folds: list[FoldData] = []
        self.config: dict | None = None
        self.config_source = ""
        self.spec: NetworkSpec | None = None
        self.model: CoReNet | None = None
        self.ds: H5Dataset | None = None
        self.rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        a = self.args
        if not self.run_dir.is_dir():
            raise SystemExit(f"{self.run_dir}: not a directory")
        fold_ids = discover_folds(self.run_dir, a.folds)
        if not fold_ids:
            raise SystemExit(f"{self.run_dir}: no history_fold*.json, model_fold*.pkl or predictions_fold*.h5")
        for k in fold_ids:
            fd = FoldData(k)
            hp = self.run_dir / f"history_fold{k}.json"
            if hp.is_file():
                fd.history = json.loads(hp.read_text())
            cp = self.run_dir / f"model_fold{k}.pkl"
            if cp.is_file():
                fd.checkpoint = cp
            self.folds.append(fd)

        # configuration and network: --config, else the first checkpoint, else the history's config path
        first_ckpt = next((fd for fd in self.folds if fd.checkpoint), None)
        if first_ckpt is not None:
            self.model, self.spec, ckpt = load_checkpoint(first_ckpt.checkpoint, a.config, "cpu")
            self.config = read_config_dict(a.config) if a.config else ckpt["config"]
            self.config_source = a.config or f"{first_ckpt.checkpoint.name} (stored configuration)"
        else:
            cfg_path = a.config
            if cfg_path is None:
                for fd in self.folds:
                    if fd.history and fd.history.get("config") and Path(fd.history["config"]).is_file():
                        cfg_path = fd.history["config"]
                        break
            if cfg_path is None:
                raise SystemExit(
                    f"{self.run_dir}: no checkpoint and no readable configuration; pass --config"
                )
            self.config = read_config_dict(cfg_path)
            self.spec = parse_config(self.config)
            self.model = CoReNet(self.spec, device="cpu")
            self.model.eval()
            self.config_source = str(cfg_path)
            self.gaps.append("no checkpoint in the run directory: the network was built from the configuration without trained weights (parameter count only)")
        for fd in self.folds:
            if fd.checkpoint:
                meta = torch.load(fd.checkpoint, map_location="cpu", weights_only=False).get("metadata", {})
                fd.checkpoint_meta = dict(meta)

        # the dataset, with the fold rule of the run
        meta = first_ckpt.checkpoint_meta if first_ckpt else {}
        hist = next((fd.history for fd in self.folds if fd.history), None) or {}
        n_folds, fold_seed = fold_rule_of(meta) if meta else (None, 0)
        if not meta and hist.get("fold_rule") == FOLD_RULE_EVENT:
            n_folds, fold_seed = hist.get("n_folds"), int(hist.get("fold_seed", 0))
        if a.n_folds is not None:
            n_folds = a.n_folds
        if a.fold_seed is not None:
            fold_seed = a.fold_seed
        rest_order = a.rest_order or meta.get("rest_order") or (hist.get("recipe") or {}).get("rest_order", "pt")
        self.ds = H5Dataset(self.input, self.spec, rest_order=rest_order, n_folds=n_folds, fold_seed=fold_seed)
        self.labels = self.ds.labels.numpy()
        self.weights = self.ds.weights.numpy().astype(np.float64)
        self.fold_of = self.ds.folds.numpy()
        logging.info(f"{self.input}: {len(self.ds):,} events, folds {self.ds.fold_rule}, classes {self.spec.classes}")

        # predictions: predictions_fold{k}.h5 (held-out fold) and eval_fold{k}.h5 (any fold, or every event)
        for fd in self.folds:
            k = fd.fold
            for name in (f"predictions_fold{k}.h5", f"eval_fold{k}.h5"):
                p = self.run_dir / name
                if not p.is_file():
                    continue
                fold_attr, split, n_par = read_prediction_file(p)
                if n_par is not None and fd.n_parameters is None:
                    fd.n_parameters = n_par
                if split.index.max(initial=-1) >= len(self.ds):
                    self.gaps.append(f"{name}: event indices exceed the input file ({len(self.ds):,} events); the file does not belong to this input")
                    continue
                if fold_attr == k:
                    fd.splits.setdefault("test", sub_split(split, np.ones(len(split.index), bool), name))
                    continue
                if fold_attr >= 0:
                    self.gaps.append(f"{name}: evaluated fold {fold_attr}, not the held-out fold {k} nor every event; ignored")
                    continue
                folds_of_rows = self.fold_of[split.index]
                test = folds_of_rows == k
                val = np.isin(folds_of_rows, fd.exclude_folds or [])
                train = ~test & ~val
                fd.splits.setdefault("test", sub_split(split, test, name))
                if fd.exclude_folds:
                    if val.any():
                        fd.splits["validation"] = sub_split(split, val, name)
                else:
                    self.gaps.append(f"fold {k}: the validation folds are unknown (no history / checkpoint metadata); every event outside fold {k} in {name} counts as training")
                if train.any():
                    fd.splits["train"] = sub_split(split, train, name)
            if "test" not in fd.splits:
                self.gaps.append(f"fold {k}: no predictions_fold{k}.h5 or eval_fold{k}.h5: no ROC, scores or view probabilities for this fold")
            if "train" not in fd.splits:
                self.gaps.append(f"fold {k}: no predictions on the training folds (run corenet-evaluate with --fold -1 and keep eval_fold{k}.h5 in the run directory): no overtraining check for this fold")
            if fd.history is None:
                self.gaps.append(f"fold {k}: no history_fold{k}.json: no training curves for this fold")
            if fd.n_parameters is None and self.model is not None:
                fd.n_parameters = int(sum(p.numel() for p in self.model.parameters()))

    # ---------------------------------------------------------------- shared data
    @property
    def classes(self) -> list[str]:
        return self.spec.classes

    def class_series(self, values, labels, weights, style=None):
        """(class name, values, clipped weights, style) per class present in ``labels``."""
        out = []
        for i, c in enumerate(self.classes):
            sel = labels == i
            if not sel.any():
                continue
            st = {"color": class_color(i), "lw": 1.3}
            st.update(style or {})
            out.append((c, values[sel], np.clip(weights[sel], 0, None), st))
        return out

    def splits_of(self, name: str):
        return [(fd, fd.splits[name]) for fd in self.folds if name in fd.splits]

    def selected_subset(self, n_max: int) -> np.ndarray:
        n = len(self.ds)
        if n_max and n > n_max:
            return np.sort(self.rng.choice(n, n_max, replace=False))
        return np.arange(n)

    # ---------------------------------------------------------------- sections
    def section_summary(self) -> str:
        parts = []
        rows = [
            ["input file", esc(self.input)],
            ["events", f"{len(self.ds):,}"],
            ["classes", esc(", ".join(f"{i} = {c}" for i, c in enumerate(self.classes)))],
            ["folds", f"{self.ds.n_folds} ({esc(self.ds.fold_rule)})"],
            ["configuration", esc(self.config_source)],
            ["run directory", esc(self.run_dir)],
            ["package", esc(f"corenet {__version__}" + (f", git {git_hash()}" if git_hash() else ""))],
            ["generated", esc(self.generated)],
        ]
        parts.append(table(["", ""], rows, "Run"))
        cfg_text = yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True)
        parts.append(f"<details><summary>Configuration ({esc(self.config_source)})</summary><pre>{esc(cfg_text)}</pre></details>")

        # events per class
        rows = []
        for i, c in enumerate(self.classes):
            sel = self.labels == i
            rows.append([c, int(sel.sum()), float(self.weights[sel].sum()), int((self.weights[sel] < 0).sum())])
        parts.append(table(["class", "events", "sum of weights", "negative weights"], rows, "Events per class in the input file"))

        # recipe
        hist = next((fd.history for fd in self.folds if fd.history), None)
        meta = next((fd.checkpoint_meta for fd in self.folds if fd.checkpoint_meta), None)
        recipe = (hist or {}).get("recipe") or {k: v for k, v in (meta or {}).items() if k not in ("fold", "epoch", "exclude_folds", "input", "n_folds", "fold_rule", "fold_seed", "finetune_trainable_parameters")}
        if recipe:
            parts.append(table(["setting", "value"], [[k, esc(v)] for k, v in recipe.items()], "Training recipe"))
        else:
            parts.append(note("No training recipe recorded (no history or checkpoint)."))
            self.gaps.append("training recipe: no history_fold*.json or model_fold*.pkl")

        # per-fold table
        rows = []
        wall_recorded = False
        for fd in self.folds:
            k = fd.fold
            h = fd.history or {}
            entries = h.get("history", [])
            n_test = int((self.fold_of == k).sum())
            n_train = h.get("n_train")
            n_val = h.get("n_val")
            if n_train is None:
                excl = fd.exclude_folds or []
                n_val = int(np.isin(self.fold_of, excl).sum())
                n_train = len(self.ds) - n_test - n_val
            wall = None
            for key in ("wall_time_s", "seconds", "time"):
                if entries and key in entries[-1]:
                    wall = entries[-1][key]
                    wall_recorded = True
            rows.append(
                [
                    k,
                    esc(fd.exclude_folds if fd.exclude_folds is not None else "-"),
                    n_train,
                    n_val,
                    n_test,
                    len(entries) if entries else "-",
                    fmt(fd.best_epoch) if fd.best_epoch is not None else "-",
                    fmt(h.get("stopped_after_epoch")) if h.get("stopped_after_epoch") is not None else "-",
                    fd.n_parameters if fd.n_parameters is not None else "-",
                    fmt(h.get("finetune_trainable_parameters")) if h.get("finetune_trainable_parameters") is not None else "-",
                    fmt(wall, 0) if wall is not None else "-",
                    esc(", ".join(sorted(fd.splits))) or "-",
                ]
            )
        parts.append(
            table(
                ["fold", "validation folds", "training events", "validation events", "test events", "epochs run", "best epoch", "stopped after", "parameters", "fine-tuned parameters", "wall time [s]", "predictions"],
                rows,
                "Folds",
            )
        )
        if not wall_recorded:
            parts.append(note("Wall time: not recorded in the history files."))
            self.gaps.append("wall time: corenet-train does not record it in history_fold*.json")
        if any(fd.history and fd.history.get("early_stop_patience") for fd in self.folds):
            parts.append(note("Best epoch: the epoch with the highest validation AUC; its weights are the ones in the checkpoint (early stopping). 'Stopped after' is the epoch at which training stopped, '-' when every epoch was run."))
        else:
            parts.append(note("Best epoch: the epoch with the highest validation AUC in the history; without early stopping the checkpoint holds the weights after the last epoch."))
        return "".join(parts)

    def section_training(self) -> str:
        with_hist = [fd for fd in self.folds if fd.history and fd.history.get("history")]
        if not with_hist:
            self.gaps.append("training curves: no history_fold*.json with epochs")
            return note("No history file with epochs: no training curves.")
        keys = [("train_loss", "training loss"), ("val_loss", "validation loss"), ("val_auc", "validation AUC")]
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
        any_val = False
        for i, (key, lab) in enumerate(keys):
            ax = axes[i]
            for j, fd in enumerate(with_hist):
                entries = fd.history["history"]
                ep = [e["epoch"] for e in entries if e.get(key) is not None]
                val = [e[key] for e in entries if e.get(key) is not None]
                if not ep:
                    continue
                any_val |= key != "train_loss"
                ax.plot(ep, val, "-", color=f"C{j % 10}", lw=1.2, label=f"fold {fd.fold}")
                best = fd.best_epoch
                if best is not None and best in ep:
                    ax.plot([best], [val[ep.index(best)]], "*", color=f"C{j % 10}", ms=9)
            ax.set_xlabel("epoch")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25, lw=0.5)
        axes[0].legend(frameon=False)
        out = figure(fig, "Training loss, validation loss and validation AUC per epoch, one line per fold; the star marks the best epoch (highest validation AUC).")
        if not any_val:
            self.gaps.append("validation curves: the histories have no validation entries (train with --exclude-folds)")
            out += note("No validation fold in the histories: validation loss and AUC are empty.")
        rows = []
        for fd in with_hist:
            e = fd.history["history"][-1]
            rows.append([fd.fold, e["epoch"], e.get("batch_size"), fmt(e.get("lr"), 6), e.get("train_loss"), e.get("val_loss"), e.get("val_auc")])
        out += table(["fold", "last epoch", "batch size", "learning rate", "training loss", "validation loss", "validation AUC"], rows, "Last epoch of every fold")
        return out

    # ---- classification
    def _main_curves(self):
        """{split: [(fold, fpr, tpr, auc)]} of the main score (class 1 against the rest)."""
        out = {}
        for name in SPLITS:
            for fd, s in self.splits_of(name):
                y, sc, w = s.label == 1, s.probs[:, 1], np.clip(s.weight, 0, None)
                if y.all() or not y.any():
                    continue
                f, t = roc_points(y, sc, w)
                out.setdefault(name, []).append((fd.fold, f, t, auc_of(y, sc, w)))
        return out

    def section_classification(self) -> str:
        if not any("test" in fd.splits for fd in self.folds):
            self.gaps.append("classification: no prediction file for any fold")
            return note("No prediction file: no ROC curves, AUC table or score distributions.")
        parts = []
        curves = self._main_curves()
        if not curves:
            self.gaps.append("ROC curves: the predicted events have only one class (no signal or no background)")
            parts.append(note("The predicted events hold a single class: no ROC curve."))
        else:
            fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.4))
            colors = {"test": "C0", "train": "C3", "validation": "C2"}
            for name in SPLITS:
                if name not in curves:
                    continue
                cs = curves[name]
                mu, sd = mean_curve([(f, t) for _, f, t, _ in cs])
                aucs = [c[3] for c in cs]
                lab = f"{name} ({len(cs)} fold{'s' if len(cs) > 1 else ''})  AUC {mean_sd(aucs)}"
                for ax in (a, b):
                    if name == "test":
                        for _, f, t, _ in cs:
                            ax.plot(f, t, color=colors[name], lw=0.6, alpha=0.35)
                    ax.plot(ROC_GRID, mu, color=colors[name], ls=SPLIT_STYLE[name], lw=1.5, label=lab if ax is a else None)
                    if sd is not None:
                        ax.fill_between(ROC_GRID, mu - sd, mu + sd, color=colors[name], alpha=0.2, lw=0)
            a.plot([0, 1], [0, 1], "k:", lw=0.8)
            a.set_xlabel("background efficiency")
            a.set_ylabel("signal efficiency")
            a.set_xlim(0, 1)
            a.set_ylim(0, 1.02)
            a.legend(loc="lower right", frameon=False)
            a.set_title("(a)", loc="left")
            b.set_xscale("log")
            b.set_xlim(1e-4, 1)
            b.set_ylim(0, 1.02)
            b.set_xlabel("background efficiency (log)")
            b.set_title("(b)", loc="left")
            b.grid(alpha=0.25, which="both", lw=0.5)
            parts.append(figure(fig, f"ROC of the main score P({self.classes[1]}) (class 1 against every other class, weighted): thin lines are the test folds, thick lines the mean over folds with the ± 1 sd band; dashed = training folds, dash-dotted = validation fold, where predictions exist."))
            # AUC table
            rows = []
            folds = sorted({c[0] for cs in curves.values() for c in cs})
            for k in folds:
                row = [k]
                for name in SPLITS:
                    v = next((c[3] for c in curves.get(name, []) if c[0] == k), None)
                    row.append(fmt(v) if v is not None else "-")
                rows.append(row)
            rows.append(["mean ± sd"] + [mean_sd([c[3] for c in curves.get(name, [])]) for name in SPLITS])
            parts.append(table(["fold", "test", "validation", "train"], rows, "AUC of the main score (sd over folds, ddof = 1)"))
            for name in ("validation", "train"):
                if name not in curves:
                    parts.append(note(f"No predictions on the {name} folds (corenet-evaluate --fold -1 provides them)."))

        # score distributions
        parts.append(self._score_figure())
        # multi-class
        if len(self.classes) > 2:
            parts.append(self._confusion_figure())
            parts.append(self._ovr_and_heads_figure())
        elif any(s.heads for fd in self.folds for s in fd.splits.values()):
            parts.append(self._ovr_and_heads_figure())
        return "".join(parts)

    def _score_figure(self) -> str:
        test = self.splits_of("test")
        if not test:
            return ""
        bins = np.linspace(0, 1, 41)
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.2))
        for name in ("test", "train"):
            sp = self.splits_of(name)
            if not sp:
                continue
            lab = np.concatenate([s.label for _, s in sp])
            sc = np.concatenate([s.probs[:, 1] for _, s in sp])
            w = np.concatenate([s.weight for _, s in sp])
            series = self.class_series(sc, lab, w, {"ls": SPLIT_STYLE[name]})
            series = [(f"{c} ({name})", v, ww, st) for c, v, ww, st in series]
            hist_panel(a, series, bins, f"P({self.classes[1]})")
            hist_panel(b, series, bins, f"P({self.classes[1]})", logy=True)
        a.legend(frameon=False)
        a.set_title("(a) linear", loc="left")
        b.set_title("(b) log", loc="left")
        cap = "Main score per class on the test folds (solid), weighted densities"
        if self.splits_of("train"):
            cap += "; dashed: the same on the training folds (overtraining check)"
        return figure(fig, cap + ".")

    def _confusion_figure(self) -> str:
        test = self.splits_of("test")
        n = len(self.classes)
        cm = np.zeros((n, n))
        for _, s in test:
            pred = s.probs.argmax(1)
            w = np.clip(s.weight, 0, None)
            for i in range(n):
                for j in range(n):
                    cm[i, j] += w[(s.label == i) & (pred == j)].sum()
        present = [i for i in range(n) if cm[i].sum() > 0]
        if not present:
            return note("Confusion matrix: no test events.")
        norm = cm / np.where(cm.sum(1, keepdims=True) > 0, cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(1.2 * n + 2.2, 1.0 * n + 1.6))
        im = ax.imshow(norm[present], cmap="Blues", vmin=0, vmax=1)
        for a, i in enumerate(present):
            for j in range(n):
                ax.text(j, a, f"{norm[i, j]:.3f}", ha="center", va="center", color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(self.classes)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([self.classes[i] for i in present])
        ax.set_xlabel("predicted class (argmax)")
        ax.set_ylabel("true class")
        fig.colorbar(im, ax=ax, fraction=0.046)
        return figure(fig, "Confusion matrix at the argmax of the class probabilities on the test folds, weighted, each true class normalised to 1.")

    def _ovr_and_heads_figure(self) -> str:
        test = self.splits_of("test")
        parts = []
        n = len(self.classes)
        has_heads = any(s.heads for _, s in test)
        fig, axes = plt.subplots(1, 2 if has_heads else 1, figsize=(7.8 if has_heads else 4.0, 3.4), squeeze=False)
        ax = axes[0, 0]
        rows = []
        for i, c in enumerate(self.classes):
            cs = [(fd.fold, *roc_points(s.label == i, s.probs[:, i], np.clip(s.weight, 0, None)), auc_of(s.label == i, s.probs[:, i], np.clip(s.weight, 0, None))) for fd, s in test if (s.label == i).any() and not (s.label == i).all()]
            if not cs:
                continue
            mu, sd = mean_curve([(f, t) for _, f, t, _ in cs])
            ax.plot(ROC_GRID, mu, color=class_color(i), lw=1.4, label=f"{c} vs rest  AUC {mean_sd([x[3] for x in cs])}")
            if sd is not None:
                ax.fill_between(ROC_GRID, mu - sd, mu + sd, color=class_color(i), alpha=0.2, lw=0)
            rows.append([f"P({c}) vs label = {c}", mean_sd([x[3] for x in cs])])
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("efficiency of the other classes")
        ax.set_ylabel("efficiency of the class")
        ax.set_title("(a) one class against the rest", loc="left")
        ax.legend(loc="lower right", frameon=False)
        if has_heads:
            ax = axes[0, 1]
            for name in sorted({k for _, s in test for k in s.heads}):
                i = self.classes.index(name) if name in self.classes else None
                if i is None:
                    continue
                cs = [(fd.fold, *roc_points(s.label == i, s.heads[name], np.clip(s.weight, 0, None)), auc_of(s.label == i, s.heads[name], np.clip(s.weight, 0, None))) for fd, s in test if name in s.heads and (s.label == i).any() and not (s.label == i).all()]
                if not cs:
                    continue
                mu, sd = mean_curve([(f, t) for _, f, t, _ in cs])
                ax.plot(ROC_GRID, mu, color=class_color(i), lw=1.4, label=f"head {name}  AUC {mean_sd([x[3] for x in cs])}")
                if sd is not None:
                    ax.fill_between(ROC_GRID, mu - sd, mu + sd, color=class_color(i), alpha=0.2, lw=0)
                rows.append([f"head {name} vs label = {name}", mean_sd([x[3] for x in cs])])
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_xlabel("efficiency of the other classes")
            ax.set_ylabel("efficiency of the class")
            ax.set_title("(b) auxiliary heads", loc="left")
            ax.legend(loc="lower right", frameon=False)
        parts.append(figure(fig, "Test folds, mean over folds with the ± 1 sd band: (a) every class scored by its own probability against all other classes" + ("; (b) the auxiliary head of every further output against its class." if has_heads else ".")))
        parts.append(table(["score", "AUC (mean ± sd over test folds)"], rows, "One-against-the-rest and auxiliary-head AUC"))
        return "".join(parts)

    # ---- hypotheses and pairing
    def section_hypotheses(self) -> str:
        test = self.splits_of("test")
        hyps = sorted({h for _, s in test for h in s.views})
        parts = []
        if not hyps:
            if test:
                parts.append(note("Every hypothesis has a single candidate: no view probabilities."))
            else:
                self.gaps.append("hypotheses: no prediction file")
                parts.append(note("No prediction file: no view probabilities."))
            return "".join(parts)
        fig, axes = grid_figure(len(hyps), per_row=3, w=3.2, h=2.6)
        for ax, h in zip(axes, hyps):
            V = next(s.views[h].shape[1] for _, s in test if h in s.views)
            bins = np.linspace(1.0 / V, 1.0, 31)
            for name in ("test", "train"):
                sp = [(fd, s) for fd, s in self.splits_of(name) if h in s.views]
                if not sp:
                    continue
                mx = np.concatenate([s.views[h].max(1) for _, s in sp])
                lab = np.concatenate([s.label for _, s in sp])
                w = np.concatenate([s.weight for _, s in sp])
                series = [(f"{c} ({name})", v, ww, st) for c, v, ww, st in self.class_series(mx, lab, w, {"ls": SPLIT_STYLE[name]})]
                hist_panel(ax, series, bins, f"max view probability, {h} ({V} views)")
            ax.legend(frameon=False)
        parts.append(figure(fig, "Highest view probability per event and class on the test folds (dashed: training folds when available). The views of a hypothesis are its candidate assignments, e.g. the pairings of the b jets into two Higgs bosons."))
        # argmax view shares
        rows = []
        for h in hyps:
            V = next(s.views[h].shape[1] for _, s in test if h in s.views)
            names = self._view_names(h, V)
            for i, c in enumerate(self.classes):
                counts = np.zeros(V)
                for _, s in test:
                    if h not in s.views:
                        continue
                    sel = s.label == i
                    if not sel.any():
                        continue
                    am = s.views[h][sel].argmax(1)
                    w = np.clip(s.weight[sel], 0, None)
                    counts += np.bincount(am, weights=w, minlength=V)[:V]
                if counts.sum() > 0:
                    rows.append([h, c] + [fmt(x) for x in counts / counts.sum()])
            if rows:
                parts.append(table(["hypothesis", "class"] + names, [r for r in rows if r[0] == h], f"Weighted share of the argmax view, test folds: {h}"))
        parts.append(self._pairing_accuracy(test, hyps))
        return "".join(parts)

    def _view_names(self, h: str, V: int) -> list[str]:
        T = self.model.tables.get(h)
        if T is None or len(T.views) != V:
            return [f"view {v}" for v in range(V)]
        names = []
        for view in T.views:
            names.append("|".join("".join(str(slot) for _, slot in child) for child in view))
        return names

    def _targets(self) -> dict[str, dict[str, np.ndarray]]:
        out = {}
        with h5py.File(self.input, "r") as f:
            if "TARGETS" not in f:
                return out
            for g in f["TARGETS"]:
                grp = f["TARGETS"][g]
                if not isinstance(grp, h5py.Group):
                    continue
                leaves = {k: grp[k][...].astype(np.int64) for k in grp if isinstance(grp[k], h5py.Dataset) and grp[k].dtype.kind in "iu" and grp[k].ndim == 1 and len(grp[k]) == len(self.ds)}
                if leaves:
                    out[g] = leaves
        return out

    def _view_assignments(self, h: str, v: int) -> dict[str, tuple]:
        """{declared node: assignment (tuples of (role, slot))} of view ``v`` of hypothesis ``h``."""
        T = self.model.tables[h]
        out = {}

        def rec(declared, canonical, cand):
            nt = T.nodes[canonical]
            out[declared] = nt.candidates[cand]
            for j, c in enumerate(self.spec.nodes[declared].children):
                if c in self.model.leaves:
                    continue
                rec(c, T.aliases.get(c, c), int(nt.child_index[cand, j]))

        rec(h, h, int(T.view_index[v]))
        return out

    def _pairing_accuracy(self, test, hyps) -> str:
        targets = self._targets()
        if not targets:
            self.gaps.append("pairing accuracy: the input file has no TARGETS group (generator-matched jets)")
            return note("No TARGETS in the input file: the pairing accuracy against the generator matching cannot be computed.")
        parts = []
        sel = self.ds.selected
        for h in hyps:
            V = next(s.views[h].shape[1] for _, s in test if h in s.views)
            T = self.model.tables.get(h)
            if T is None or len(T.views) != V:
                continue
            declared = self.spec.nodes_in(h)
            matched = {g: n for g in targets for n in declared if norm_name(g) == norm_name(n)}
            if not matched:
                self.gaps.append(f"pairing accuracy of {h}: no TARGETS group named like one of its nodes ({sorted(declared)}; TARGETS has {sorted(targets)})")
                parts.append(note(f"{h}: no TARGETS group matches a node of the hypothesis ({', '.join(sorted(declared))}); TARGETS groups: {', '.join(sorted(targets))}."))
                continue
            # per matched node: the sorted file slots of its generator-matched jets (truth) and of its
            # candidate under every view; sibling nodes that are interchangeable (H1 / H2) are compared as
            # a set, so that swapping them does not count as a wrong assignment
            n = len(self.ds)
            correct = np.ones((V, n), bool)
            complete = np.ones(n, bool)
            contained = np.ones(n, bool)
            usable = []
            truth_of, views_of = {}, {}
            for g, node in matched.items():
                truth = np.stack(list(targets[g].values()), 1)  # (n, k) file slots
                assignments = [self._view_assignments(h, v).get(node) for v in range(V)]
                pairs = [[(r, s) for r, s in a if r in sel] if a is not None else None for a in assignments]
                if any(p is None or len(p) != truth.shape[1] for p in pairs):
                    continue  # the group lists a different number of jets than the node has candidates
                usable.append((g, node))
                complete &= (truth >= 0).all(1)
                roles = sorted({r for p in pairs for r, _ in p})
                pool = np.concatenate([sel[r] for r in roles], 1)  # (n, slots) file slots of the roles
                for col in range(truth.shape[1]):
                    contained &= (pool == truth[:, col : col + 1]).any(1)
                truth_of[node] = np.sort(truth, 1)
                views_of[node] = [np.sort(np.stack([sel[r][:, slot] for r, slot in p], 1), 1) for p in pairs]
            # groups of interchangeable siblings that are all matched
            grouped, groups = set(), []
            for parent in declared:
                nt = T.nodes.get(T.aliases.get(parent, parent))
                kids = self.spec.nodes[parent].children if parent in self.spec.nodes else []
                if nt is not None and nt.symmetric_children and len(kids) > 1 and all(k in truth_of for k in kids):
                    groups.append(list(kids))
                    grouped.update(kids)

            def key(slots):  # one integer per sorted slot tuple
                base = 1 << 16
                return sum((slots[:, j].astype(np.int64) + 1) * base**j for j in range(slots.shape[1]))

            for node in truth_of:
                if node in grouped:
                    continue
                for v in range(V):
                    correct[v] &= (views_of[node][v] == truth_of[node]).all(1)
            for kids in groups:
                t = np.sort(np.stack([key(truth_of[k]) for k in kids], 1), 1)
                for v in range(V):
                    correct[v] &= (np.sort(np.stack([key(views_of[k][v]) for k in kids], 1), 1) == t).all(1)
            if not usable:
                parts.append(note(f"{h}: the matched TARGETS groups ({', '.join(matched)}) list a different number of jets than the nodes they name."))
                continue
            if all(np.array_equal(correct[0], correct[v]) for v in range(1, V)):
                parts.append(note(f"{h}: the TARGETS groups {', '.join(g for g, _ in usable)} do not distinguish the {V} views (every view assigns their jets the same way): no pairing accuracy."))
                self.gaps.append(f"pairing accuracy of {h}: the TARGETS groups do not distinguish its views")
                continue
            rows = []
            per_fold = []
            thr_rows = {}
            for fd, s in test:
                if h not in s.views:
                    continue
                idx = s.index
                comp, cont = complete[idx], contained[idx] & complete[idx]
                am = s.views[h].argmax(1)
                mx = s.views[h].max(1)
                ok = correct[am, idx]
                acc_m = ok[comp].mean() if comp.any() else np.nan
                acc_c = ok[cont].mean() if cont.any() else np.nan
                per_fold.append((acc_m, acc_c))
                rows.append([fd.fold, int(comp.sum()), int(cont.sum()), fmt(acc_m), fmt(acc_c)])
                thr = [1.0 / V] + [t for t in VIEW_THRESHOLDS if t > 1.0 / V + 1e-9]
                thr_rows[fd.fold] = [(t, ok[cont & (mx > t)].mean() if (cont & (mx > t)).any() else np.nan, int((cont & (mx > t)).sum())) for t in thr]
            if not rows:
                continue
            rows.append(["mean ± sd", "", "", mean_sd([a for a, _ in per_fold]), mean_sd([c for _, c in per_fold])])
            parts.append(table(["fold", "events with complete truth", "of which contained", "accuracy (complete)", "accuracy (contained)"], rows, f"Pairing accuracy of the argmax view of {h} against TARGETS {', '.join(g for g, _ in usable)} (random: 1/{V} = {1.0 / V:.3f})"))
            fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.2))
            thr = [t for t, _, _ in next(iter(thr_rows.values()))]
            accs = np.array([[acc for _, acc, _ in thr_rows[k]] for k in thr_rows])
            ns = np.array([[nn for _, _, nn in thr_rows[k]] for k in thr_rows]).sum(0)
            x = np.arange(len(thr))
            for j, k in enumerate(thr_rows):
                a.plot(x, 100 * accs[j], "-", color=f"C{j % 10}", lw=0.8, alpha=0.5, label=f"fold {k}")
            mu = np.nanmean(accs, 0)
            a.plot(x, 100 * mu, "o-", color="k", lw=1.4, ms=4, label="mean over folds")
            for xi, (m, nn) in enumerate(zip(mu, ns)):
                a.annotate(f"N = {nn:,}", (xi, 100 * m), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=6)
            a.axhline(100.0 / V, color="k", ls="--", lw=0.8)
            a.set_xticks(x)
            a.set_xticklabels([f"> {t:.2f}" for t in thr])
            a.set_xlabel(f"max view probability of {h}")
            a.set_ylabel("pairing accuracy [%]")
            a.set_ylim(0, 105)
            a.legend(frameon=False)
            a.grid(alpha=0.25, lw=0.5)
            a.set_title("(a) accuracy above a confidence threshold", loc="left")
            # binned accuracy
            edges = np.linspace(1.0 / V, 1.0, 11)
            for j, (fd, s) in enumerate([(fd, s) for fd, s in test if h in s.views]):
                idx = s.index
                cont = contained[idx] & complete[idx]
                am, mx = s.views[h].argmax(1), s.views[h].max(1)
                ok = correct[am, idx]
                which = np.clip(np.digitize(mx, edges) - 1, 0, len(edges) - 2)
                acc = [ok[cont & (which == b)].mean() if (cont & (which == b)).any() else np.nan for b in range(len(edges) - 1)]
                b.plot(0.5 * (edges[1:] + edges[:-1]), 100 * np.array(acc), "o-", color=f"C{j % 10}", lw=1.0, ms=3, label=f"fold {fd.fold}")
            b.plot([1.0 / V, 1], [100.0 / V, 100], "k:", lw=0.8, label="perfect calibration")
            b.set_xlabel(f"max view probability of {h}")
            b.set_ylabel("pairing accuracy [%]")
            b.set_ylim(0, 105)
            b.legend(frameon=False)
            b.grid(alpha=0.25, lw=0.5)
            b.set_title("(b) accuracy per probability bin", loc="left")
            parts.append(figure(fig, f"Pairing accuracy of {h} on contained events of the test folds (all generator-matched jets among the candidates): (a) for events whose highest view probability exceeds a threshold, N summed over folds, dashed = random; (b) in bins of the highest view probability, dotted = accuracy equal to the probability."))
        return "".join(parts)

    # ---- inputs as the network sees them
    def _node_phi_label(self, h: str, node: str) -> str:
        T = self.model.tables[h]
        parents = [(pn, j) for pn in self.model.levels[h] for j, c in enumerate(T.nodes[pn].children) if c == node]
        pnodes = {pn for pn, _ in parents}
        if len(pnodes) == 1:
            pnode = next(iter(pnodes))
            P = T.nodes[node].n
            allidx = torch.cat([T.nodes[pnode].child_index[:, j] for _, j in parents])
            if allidx.numel() == P and allidx.unique().numel() == P:
                return f"Δφ to {pnode}"
        return "φ"

    def collect_network_inputs(self, idx: np.ndarray, chunk: int = 16384) -> dict:
        """{key: (feature names, values (m, F), labels (m,), weights (m,))} of every tensor the network
        embeds, over the events ``idx`` and all present candidates."""
        spec, model = self.spec, self.model
        acc: dict[str, tuple] = {}

        def add(key, names, x, present, rows):
            x = x.detach().cpu().numpy()
            n, F, P = x.shape
            pres = np.ones((n, P), bool) if present is None else present.detach().cpu().numpy().astype(bool)
            vals = x.transpose(0, 2, 1)[pres]
            lab = np.broadcast_to(self.labels[rows][:, None], (n, P))[pres]
            w = np.broadcast_to(self.weights[rows][:, None], (n, P))[pres]
            e = acc.setdefault(key, (list(names), [], [], []))
            e[1].append(vals)
            e[2].append(lab)
            e[3].append(w)

        def event_name(name):
            return f"log({name} - {spec.min_jets - 1})" if "nSelJets" in name else name

        with torch.no_grad():
            for start in range(0, len(idx), chunk):
                rows = idx[start : start + chunk]
                sel = torch.as_tensor(rows)
                inputs = {k: v[sel] for k, v in self.ds.inputs.items()}
                prep = model._prepare(inputs)
                if prep["event"] is not None:
                    add("event scalars", [event_name(n) for n in spec.event], prep["event"], None, rows)
                seen = set()
                for h, T in model.tables.items():
                    hp = prep["hyp"][h]
                    for node in model.levels[h]:
                        if node in seen:
                            continue
                        seen.add(node)
                        nt = T.nodes[node]
                        for j, child in enumerate(nt.children):
                            names = ["log(1+pT)", "η", f"Δφ to {node}", "log(1+m)"]
                            if child in model.leaves:
                                names += model._extras(child)
                            add(f"{node} ← {child}", names, hp["slot_feats"][node][j], hp["slot_present"][node][j], rows)
                        if node in model.tops:
                            names = ["log(1+pT)", "η", "log(1+m)"]
                        else:
                            owner = spec.owner.get(node, h)
                            owner = owner if node in model.levels.get(owner, []) else h
                            names = ["log(1+pT)", "η", self._node_phi_label(owner, node), "log(1+m)"]
                        add(f"{node} candidates", names, prep["embed_in"][node], prep["present"][node], rows)
                    if "context" in hp:
                        c = hp["context"]
                        add(f"{h} context: {spec.hypotheses[h].context} and the candidates' jets", ["log(1+pT)", "η", "log(1+m)", "flag (2 = candidate jet, 1 = context jet)"], c["feats"], ~c["mask"], rows)
                for key, r in prep["relations"].items():
                    add(f"relation {key.replace('|', ' - ')}", ["log(1+M)", "ΔR", "Δφ"], r, None, rows)
        return {k: (names, np.concatenate(v), np.concatenate(l), np.concatenate(w)) for k, (names, v, l, w) in acc.items()}

    def section_inputs(self) -> str:
        idx = self.selected_subset(self.args.max_events)
        data = self.collect_network_inputs(idx)
        parts = [note(f"Distributions of the quantities the network embeds, computed from the input file by the loader and the network's own preparation step, over {len(idx):,} events" + (" (a random subsample; --max-events)" if len(idx) < len(self.ds) else "") + f" and all present candidates; weighted by the event weight (clipped at 0), each class normalised to unit area. Jet features: log(1+pT), η" + (" (reflected so that the leading jet of the first role has η ≥ 0)" if self.spec.training.reflect_eta else "") + ", Δφ to the parent candidate, log(1+m).")]
        for key, (names, vals, lab, w) in data.items():
            fig, axes = grid_figure(len(names), per_row=6)
            for ax, j, name in zip(axes, range(len(names)), names):
                series = self.class_series(vals[:, j], lab, w)
                hist_panel(ax, series, shared_bins([v for _, v, _, _ in series]), name)
            axes[0].legend(frameon=False)
            parts.append(figure(fig, f"{key}: {', '.join(names)}."))
        return "".join(parts)

    # ---- raw inputs
    def section_raw(self) -> str:
        ds, spec = self.ds, self.spec
        idx = self.selected_subset(self.args.max_events)
        lab, w = self.labels[idx], self.weights[idx]
        parts = [note(f"The features of the input file for the members each role selected (per slot), the single objects and the event scalars, before any transformation; {len(idx):,} events" + (" (random subsample)" if len(idx) < len(ds) else "") + ", weighted, each class normalised to unit area. A column read for the selection only (e.g. btag) is shown for the selected jets as well.")]
        extra_cols = {}
        with h5py.File(self.input, "r") as f:
            for role in spec.roles.values():
                coll = spec.collections[role.collection]
                cols = [c for c in role.columns if c not in coll.features]
                if cols:
                    group = f["INPUTS"][resolve_group(f, coll.name)]
                    extra_cols[role.name] = {c: np.atleast_2d(group[c][...].astype(np.float32).T).T for c in cols if c in group}
        for role in spec.roles.values():
            coll = spec.collections[role.collection]
            x = ds.inputs[role.name][idx].numpy()  # (n, F+1, slots)
            present = x[:, -1] > 0
            feats = list(coll.features)
            extra = extra_cols.get(role.name, {})
            n_slots = x.shape[2]
            names = feats + list(extra)
            fig, axes = plt.subplots(len(names), n_slots, figsize=(2.3 * n_slots, 1.9 * len(names)), squeeze=False)
            for i, name in enumerate(names):
                if name in feats:
                    col = x[:, feats.index(name)]
                else:
                    slots = ds.selected[role.name][idx]
                    col = np.where(slots >= 0, extra[name][idx[:, None], np.clip(slots, 0, None)], np.nan)
                bins = shared_bins([col[present]])
                for s in range(n_slots):
                    ax = axes[i, s]
                    ok = present[:, s]
                    series = self.class_series(col[ok, s], lab[ok], w[ok])
                    hist_panel(ax, series, bins, f"{name} [{role.name} slot {s}]")
            axes[0, 0].legend(frameon=False)
            sel_text = f"{role.collection} {role.selector} {role.n}" + (" lowest" if role.ascending else "")
            parts.append(figure(fig, f"Role {role.name} ({sel_text}): rows = features, columns = slots in selection order; empty slots are left out.", dpi=90))
        singles = [c for c in spec.collections.values() if c.max == 1 and c.name in ds.inputs]
        for c in singles:
            x = ds.inputs[c.name][idx].numpy()
            present = x[:, -1, 0] > 0
            fig, axes = grid_figure(len(c.features), per_row=6)
            for ax, j, name in zip(axes, range(len(c.features)), c.features):
                series = self.class_series(x[present, j, 0], lab[present], w[present])
                hist_panel(ax, series, shared_bins([v for _, v, _, _ in series]), f"{name} [{c.name}]")
            axes[0].legend(frameon=False)
            parts.append(figure(fig, f"Single object {c.name}: {', '.join(c.features)}."))
        if spec.event:
            ev = ds.inputs["event"][idx].numpy()
            fig, axes = grid_figure(len(spec.event), per_row=6)
            for ax, j, name in zip(axes, range(len(spec.event)), spec.event):
                kind, where = event_source(spec, name)
                series = self.class_series(ev[:, j], lab, w)
                hist_panel(ax, series, shared_bins([v for _, v, _, _ in series]), f"{name} ({'computed from ' + where if kind == 'computed' else where})")
            axes[0].legend(frameon=False)
            parts.append(figure(fig, "Event scalars as read or computed by the loader."))
        if not spec.roles and not singles and not spec.event:
            parts.append(note("The configuration declares no roles, single objects or event scalars."))
        return "".join(parts)

    # ---- weights
    def section_weights(self) -> str:
        lab, w = self.labels, self.weights
        parts = []
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.8, 3.2))
        series = []
        for i, c in enumerate(self.classes):
            sel = lab == i
            if sel.any():
                series.append((c, w[sel], np.ones(int(sel.sum())), {"color": class_color(i), "lw": 1.3}))
        bins = shared_bins([v for _, v, _, _ in series], 60)
        for lab_c, v, ww, st in series:
            a.hist(v, bins=bins, histtype="step", label=lab_c, **st)
        a.set_yscale("log")
        a.set_xlabel("event weight (as in the file)")
        a.set_ylabel("events")
        a.legend(frameon=False)
        a.set_title("(a) raw weights", loc="left")
        normed = w.copy()
        for i in range(len(self.classes)):
            sel = lab == i
            tot = np.clip(w[sel], 0, None).sum()
            if tot > 0:
                normed[sel] = np.clip(w[sel], 0, None) / tot
        series = [(c, normed[lab == i], np.ones(int((lab == i).sum())), {"color": class_color(i), "lw": 1.3}) for i, c in enumerate(self.classes) if (lab == i).any()]
        bins = shared_bins([v for _, v, _, _ in series], 60)
        for lab_c, v, ww, st in series:
            b.hist(v, bins=bins, histtype="step", label=lab_c, **st)
        b.set_yscale("log")
        b.set_xlabel("class-normalised weight (clipped at 0, class sum = 1)")
        b.set_ylabel("events")
        b.set_title("(b) as training uses them", loc="left")
        parts.append(figure(fig, "Event weights per class over the whole input file: (a) as stored; (b) clipped at 0 and divided by the class sum, which is how the training loss weights an event (the training normalises over its own training events of every fold)."))
        rows = []
        for i, c in enumerate(self.classes):
            sel = lab == i
            if not sel.any():
                continue
            ws = w[sel]
            s1, s2 = ws.sum(), (ws**2).sum()
            neff = s1**2 / s2 if s2 > 0 else 0.0
            rows.append([c, int(sel.sum()), float(s1), float(ws.min()), float(ws.max()), float(ws.mean()), float(ws.std()), fmt(neff, 1), fmt(neff / sel.sum(), 4), int((ws < 0).sum()), fmt(abs(ws[ws < 0].sum()) / s1 if s1 > 0 else np.nan)])
        parts.append(table(["class", "events", "sum", "min", "max", "mean", "sd", "effective events (sum w)^2 / sum w^2", "effective / events", "negative weights", "|negative sum| / sum"], rows, "Weight statistics per class (whole file)"))
        folds = sorted(np.unique(self.fold_of).tolist())
        rows = []
        for k in folds:
            fsel = self.fold_of == k
            row = [k, int(fsel.sum())]
            for i, c in enumerate(self.classes):
                sel = fsel & (lab == i)
                tot = w[lab == i].sum()
                row += [int(sel.sum()), float(w[sel].sum()), fmt(w[sel].sum() / tot if tot else np.nan, 3), int((w[sel] < 0).sum())]
            rows.append(row)
        header = ["fold", "events"] + [f"{c}: {q}" for c in self.classes for q in ("events", "sum", "share", "negative")]
        parts.append(table(header, rows, "Events and weights per fold and class (closure: every fold should carry about 1/n_folds of each class)"))
        rows = []
        for fd in self.folds:
            k = fd.fold
            excl = fd.exclude_folds or []
            train = ~np.isin(self.fold_of, [k, *excl])
            row = [k, esc(excl) if excl else "-"]
            for i, c in enumerate(self.classes):
                sel = train & (lab == i)
                tot = np.clip(w[sel], 0, None).sum()
                row += [int(sel.sum()), float(tot), int((w[sel] < 0).sum())]
            rows.append(row)
        header = ["held-out fold", "validation folds"] + [f"{c}: {q}" for c in self.classes for q in ("training events", "training sum", "negative (count as 0)")]
        parts.append(table(header, rows, "Training set of every fold: the weights of every class are normalised to 1 over these events"))
        return "".join(parts)

    # ---- other checks
    def section_other(self) -> str:
        ds, spec = self.ds, self.spec
        parts = []
        idx = self.selected_subset(self.args.max_events)
        lab, w = self.labels[idx], self.weights[idx]
        profiles = []
        first_role = next(iter(spec.roles.values()), None)
        if first_role is not None:
            x = ds.inputs[first_role.name][idx].numpy()
            feats = list(spec.collections[first_role.collection].features)
            if "pt" in feats:
                pt = x[:, feats.index("pt")]
                present = x[:, -1] > 0
                for s in (0, first_role.n - 1):
                    if s < pt.shape[1]:
                        profiles.append((f"pT of {first_role.name} slot {s} [GeV]", np.where(present[:, s], pt[:, s], np.nan)))
        for j, name in enumerate(spec.event):
            if "nSelJets" in name:
                profiles.append((name, ds.inputs["event"][idx].numpy()[:, j]))
        if not profiles:
            parts.append(note("No jet role or jet multiplicity in the configuration: no weight profiles."))
            return "".join(parts)
        fig, axes = grid_figure(len(profiles), per_row=3, w=3.2, h=2.6)
        for ax, (name, x) in zip(axes, profiles):
            ok = np.isfinite(x)
            bins = shared_bins([x[ok]], 20)
            centres = 0.5 * (bins[1:] + bins[:-1])
            for i, c in enumerate(self.classes):
                sel = ok & (lab == i)
                if not sel.any():
                    continue
                which = np.clip(np.digitize(x[sel], bins) - 1, 0, len(bins) - 2)
                means = np.array([w[sel][which == b].mean() if (which == b).any() else np.nan for b in range(len(bins) - 1)])
                sems = np.array([w[sel][which == b].std() / np.sqrt((which == b).sum()) if (which == b).sum() > 1 else np.nan for b in range(len(bins) - 1)])
                ax.errorbar(centres, means, sems, fmt="o-", color=class_color(i), ms=3, lw=1.0, capsize=2, label=c)
            ax.set_xlabel(name)
            ax.set_ylabel("mean event weight")
            ax.grid(alpha=0.25, lw=0.5)
        axes[0].legend(frameon=False)
        parts.append(figure(fig, "Mean event weight (as stored) per class in bins of the leading and the last candidate jet pT of the first role and of the jet multiplicity; error bars are the standard error of the mean."))
        return "".join(parts)

    def section_gaps(self) -> str:
        if not self.gaps:
            return "<p>Everything requested could be produced for this run.</p>"
        items = "".join(f"<li>{esc(g)}</li>" for g in dict.fromkeys(self.gaps))
        return f"<ul>{items}</ul>"

    # ---------------------------------------------------------------- page
    def render(self) -> str:
        bodies = {}
        order = [s for s in SECTIONS if s[0] != "gaps"]
        for key, title in order:
            try:
                bodies[key] = getattr(self, f"section_{key}")()
            except Exception as e:  # a section that fails is reported, not fatal
                logging.error(f"section '{title}' failed:\n{traceback.format_exc()}")
                self.gaps.append(f"section '{title}' could not be produced: {type(e).__name__}: {e}")
                bodies[key] = f'<p class="warn">This section could not be produced: {esc(type(e).__name__)}: {esc(e)}</p>'
            plt.close("all")
        bodies["gaps"] = self.section_gaps()
        toc = "".join(f'<li><a href="#{k}">{i + 1}. {esc(t)}</a></li>' for i, (k, t) in enumerate(SECTIONS))
        sections = "".join(f'<section id="{k}"><h2>{i + 1}. {esc(t)}</h2>{bodies[k]}</section>' for i, (k, t) in enumerate(SECTIONS))
        cmd = esc(" ".join(sys.argv))
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>CoReNet report: {esc(self.run_dir)}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;font-size:14px;color:#222;max-width:1300px;margin:0 auto;padding:16px;line-height:1.4}}
h1{{font-size:22px}} h2{{font-size:18px;border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:36px}}
table{{border-collapse:collapse;margin:10px 0;font-size:12.5px}} th,td{{border:1px solid #ccc;padding:3px 7px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
caption{{caption-side:top;text-align:left;font-weight:600;padding:4px 0}}
figure{{margin:12px 0}} figure img{{max-width:100%;height:auto;border:1px solid #eee}} figcaption{{font-size:12px;color:#555;margin-top:4px}}
details{{margin:8px 0}} pre{{background:#f6f6f6;padding:8px;overflow:auto;font-size:12px}}
.note{{color:#444;background:#f3f6fb;border-left:3px solid #8aa;padding:6px 10px}} .warn{{color:#7a1f1f;background:#fbf0f0;border-left:3px solid #c66;padding:6px 10px}}
nav ol{{columns:2;font-size:13px}} .meta{{color:#555;font-size:12.5px}}
</style></head><body>
<h1>CoReNet report</h1>
<p class="meta">run directory {esc(self.run_dir)} | input {esc(self.input)} | generated {esc(self.generated)} | corenet {esc(__version__)}<br>command: {cmd}</p>
<nav><ol>{toc}</ol></nav>
{sections}
<footer class="meta"><p>Weighted quantities use the event weights of the input file clipped at 0, as the training does. Mean ± sd are over folds with ddof = 1.</p></footer>
</body></html>
"""


_GIT_HASH: str | None = None


def git_hash() -> str:
    """Short git hash of the package checkout, '' when not in a git repository."""
    global _GIT_HASH
    if _GIT_HASH is None:
        try:
            out = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            _GIT_HASH = out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            _GIT_HASH = ""
    return _GIT_HASH


# ----------------------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corenet-report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="directory written by corenet-train (and corenet-evaluate)")
    p.add_argument("--input", required=True, help="the HDF5 file the run was trained on")
    p.add_argument("--output", default=None, help="HTML file to write [<run-dir>/report.html]")
    p.add_argument("--config", default=None, help="configuration file [the one stored in the checkpoint]")
    p.add_argument("--folds", type=int, nargs="*", default=None, help="folds to include [every fold found in the run directory]")
    p.add_argument("--max-events", type=int, default=500000, help="events of the input file used for the input and weight distributions (random subsample above this) [500000]; 0 = all")
    p.add_argument("--n-folds", type=int, default=None, help="fold = squares(event) %% K instead of EVENT/fold [the run's rule]")
    p.add_argument("--fold-seed", type=int, default=None, help="seed of that rule [the run's]")
    p.add_argument("--rest-order", choices=["pt", "selector"], default=None, help="ordering of the 'rest' selector [the run's]")
    return p


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = Report(args)
    report.load()
    page = report.render()
    out = Path(args.output) if args.output else report.run_dir / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    logging.info(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(report.gaps)} items not produced)")
    for g in dict.fromkeys(report.gaps):
        logging.info(f"not produced: {g}")
    return out


def cli():
    main()


if __name__ == "__main__":
    cli()
