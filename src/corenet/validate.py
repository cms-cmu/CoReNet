"""
corenet-validate: check a configuration, and optionally an HDF5 file against it, before training.

    corenet-validate --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5

1. Configuration: parsed with the same code as corenet-train (``parse_config``: required sections,
   unknown keys, types and ranges, every role / node / attention edge / relation / output /
   combine entry refers to a declared name; the ``trainer:`` recipe section), then the network is
   built (candidate enumeration, relation restrictions) and summarised.
2. Data (with ``--input``): every dataset the loader reads is looked up with the loader's own rules
   (``corenet.data.h5``: group aliases, used collections, computed event scalars, label keys) and its
   dtype and shape are checked; the labels must lie in 0..len(outputs), the weights must be finite,
   the folds must exist for training (``EVENT/fold``, or ``--n-folds K`` for fold = event % K from
   the event numbers; the report gives the fold source and the events per fold); the present members
   must have finite features. Finally the actual loader (``H5Dataset``) reads the file and the network
   runs a forward pass on the first events, so a file that passes here also loads in corenet-train.

Every finding is printed as OK / WARN / FAIL with the dataset or key it concerns; the exit status is 1
when anything FAILed. ``--json`` also writes the report as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import h5 as loader
from .models.config import (
    REST,
    ConfigError,
    NetworkSpec,
    parse_config,
    read_config_dict,
)
from .models.network import CoReNet


class Report:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []  # (level, section, message)

    def ok(self, section, msg):
        self.items.append(("OK", section, msg))

    def warn(self, section, msg):
        self.items.append(("WARN", section, msg))

    def fail(self, section, msg):
        self.items.append(("FAIL", section, msg))

    @property
    def errors(self):
        return [m for lvl, _, m in self.items if lvl == "FAIL"]

    @property
    def warnings(self):
        return [m for lvl, _, m in self.items if lvl == "WARN"]

    def print(self, out=sys.stdout):
        section = None
        for lvl, sec, msg in self.items:
            if sec != section:
                print(f"\n== {sec}", file=out)
                section = sec
            print(f"  [{lvl:>4}] {msg}", file=out)
        n_ok = sum(lvl == "OK" for lvl, _, _ in self.items)
        print(
            f"\nSUMMARY: {n_ok} checks passed, {len(self.warnings)} warnings, {len(self.errors)} errors"
            + (" -> NOT VALID" if self.errors else " -> valid"),
            file=out,
        )

    def to_dict(self):
        return {
            "valid": not self.errors,
            "items": [
                {"level": a, "section": b, "message": c} for a, b, c in self.items
            ],
        }


# ----------------------------------------------------------------------------------- configuration
def check_config(path, report: Report) -> NetworkSpec | None:
    sec = f"configuration {path}"
    try:
        cfg = read_config_dict(path)
    except FileNotFoundError:
        report.fail(sec, f"file not found: {path}")
        return None
    except Exception as e:  # YAML / JSON syntax
        report.fail(sec, f"cannot read {path}: {type(e).__name__}: {e}")
        return None
    try:
        spec = parse_config(cfg)
    except ConfigError as e:
        report.fail(sec, f"invalid configuration: {e}")
        return None
    report.ok(sec, "sections, names, types and ranges (parse_config)")
    try:
        from .train import build_parser, resolve_recipe

        args = resolve_recipe(
            build_parser().parse_args(["--config", str(path), "--input", "-"]), cfg
        )
        if cfg.get("trainer"):
            report.ok(
                sec,
                "trainer: "
                + ", ".join(f"{k}={getattr(args, k)}" for k in cfg["trainer"]),
            )
    except ConfigError as e:
        report.fail(sec, f"invalid trainer section: {e}")
    try:
        model = CoReNet(spec, device="cpu")
    except ConfigError as e:
        report.fail(sec, f"network cannot be built: {e}")
        return None

    for c in spec.collections.values():
        report.ok(
            sec,
            f"collection `{c.name}`: max {c.max}, features {c.features} (the network inputs)",
        )
    for r in spec.roles.values():
        how = (
            ""
            if r.selector == REST
            else (" lowest first" if r.ascending else " highest first")
        )
        report.ok(
            sec,
            f"role `{r.name}`: {r.n} x `{r.collection}` selected by `{r.selector}`{how} "
            f"(reads {', '.join(r.columns)} for the selection)",
        )
    for h, T in model.tables.items():
        nodes = ", ".join(f"{n} ({T.nodes[n].n})" for n in model.levels[h])
        ctx = spec.hypotheses[h].context
        report.ok(
            sec,
            f"hypothesis `{h}`: {T.nodes[h].n} view(s); nodes (candidates): {nodes}"
            + (f"; context `{ctx}`" if ctx else ""),
        )
    for key in model.edge_nodes:
        kind = (
            "attention"
            if key in model.attention
            else "merge (one admissible key per query)"
        )
        report.ok(sec, f"edge `{key.replace('>', ': ')}` built as {kind}")
    for key in model.relation_nodes:
        report.ok(sec, f"relation `[{key.replace('|', ', ')}]`")
    report.ok(
        sec,
        f"classes {spec.classes}; main output `{spec.main.name}` on `{spec.main.hypothesis}`; "
        f"auxiliary heads {[o.name for o in spec.outputs[1:]]}; combine {spec.combine}",
    )
    report.ok(
        sec, f"network built: {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return spec


# ----------------------------------------------------------------------------------- data
def _numeric(ds) -> bool:
    return ds.dtype.kind in "fiub"


def check_data(
    path,
    spec: NetworkSpec,
    report: Report,
    rest_order: str = "pt",
    n_forward: int = 512,
    n_folds: int | None = None,
) -> None:
    sec = f"data {path}"
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        report.fail(sec, f"cannot open {path}: {type(e).__name__}: {e}")
        return
    with f:
        n_events = None

        def length(name, ds):
            nonlocal n_events
            if n_events is None:
                n_events = ds.shape[0]
                report.ok(sec, f"{n_events:,} events (first dimension of {name})")
                return True
            if ds.shape[0] != n_events:
                report.fail(
                    sec,
                    f"{name} has {ds.shape[0]:,} entries, expected {n_events:,} (one per event)",
                )
                return False
            return True

        # ---- collections
        if "INPUTS" not in f:
            report.fail(
                sec, f"no INPUTS group in the file (top-level groups: {sorted(f)})"
            )
            return
        for coll in loader.used_collections(spec):
            users = [
                r.name for r in spec.roles.values() if r.collection == coll.name
            ] or [coll.name]
            who = f"collection `{coll.name}` (used by {', '.join(f'`{u}`' for u in users)})"
            try:
                gname = loader.resolve_group(f, coll.name)
            except KeyError:
                alts = " or ".join(
                    f"INPUTS/{g}" for g in loader.group_candidates(coll.name)
                )
                report.fail(
                    sec,
                    f"{who} expects a group {alts}: not found (INPUTS has {sorted(f['INPUTS'])})",
                )
                continue
            group = f["INPUTS"][gname]
            slots = None
            columns = loader.columns_to_read(spec, coll)
            selection = [
                c for c in columns if c not in coll.features
            ]  # read for a selector only
            for feat in columns:
                name = f"INPUTS/{gname}/{feat}"
                readers = [
                    f"`{r.name}`"
                    for r in spec.roles.values()
                    if r.collection == coll.name and feat in r.columns
                ]
                why = (
                    f" (read for the selection by {', '.join(readers)}, not a feature)"
                    if feat in selection
                    else ""
                )
                if feat not in group or not isinstance(group[feat], h5py.Dataset):
                    report.fail(
                        sec,
                        f"{who} expects dataset {name}{why}: not found (INPUTS/{gname} has {sorted(group)})",
                    )
                    continue
                ds = group[feat]
                if not _numeric(ds):
                    report.fail(sec, f"{name}: dtype {ds.dtype} is not numeric")
                    continue
                if not length(name, ds):
                    continue
                if coll.max > 1:
                    if ds.ndim != 2:
                        report.fail(
                            sec,
                            f"{name}: shape {ds.shape}, expected (events, slots) for a collection with max {coll.max}",
                        )
                        continue
                    if ds.shape[1] < coll.max:
                        report.fail(
                            sec,
                            f"{name}: {ds.shape[1]} slots, the configuration asks for max {coll.max}",
                        )
                        continue
                    if slots is not None and ds.shape[1] != slots:
                        report.fail(
                            sec,
                            f"{name}: {ds.shape[1]} slots, other features of `{coll.name}` have {slots}",
                        )
                        continue
                    slots = ds.shape[1]
                elif not (ds.ndim == 1 or (ds.ndim == 2 and ds.shape[1] == 1)):
                    report.fail(
                        sec,
                        f"{name}: shape {ds.shape}, expected (events,) for a single-object collection (max 1)",
                    )
                    continue
            if "MASK" in group:
                m = group["MASK"]
                expect = (n_events, slots) if slots else None
                if m.dtype.kind not in "biu":
                    report.fail(
                        sec,
                        f"INPUTS/{gname}/MASK: dtype {m.dtype}, expected bool (True = present)",
                    )
                elif expect and m.shape != expect:
                    report.fail(
                        sec, f"INPUTS/{gname}/MASK: shape {m.shape}, expected {expect}"
                    )
                else:
                    report.ok(sec, f"INPUTS/{gname}/MASK {m.shape} {m.dtype}")
            elif coll.max > 1:
                report.warn(
                    sec,
                    f"INPUTS/{gname}/MASK not found: every one of the {slots} slots of `{coll.name}` counts as present",
                )
            missing_kin = [
                k for k in ("pt", "eta", "phi", "mass") if k not in coll.features
            ]
            note = f"; {', '.join(missing_kin)} taken as 0" if missing_kin else ""
            report.ok(
                sec,
                f"{who}: INPUTS/{gname} with features {coll.features}"
                + (f", {slots} slots" if slots else "")
                + note,
            )
            if selection:
                report.ok(
                    sec,
                    f"{who}: {selection} read for the selection only (not network inputs)",
                )

        # ---- event scalars
        for name in spec.event:
            kind, where = loader.event_source(spec, name)
            if kind == "computed":
                note = (
                    f" and enters as log({name} - {spec.min_jets - 1}) (min_jets {spec.min_jets})"
                    if "nSelJets" in name
                    else ""
                )
                report.ok(
                    sec,
                    f"event scalar `{name}` computed from collection `{where}`{note}",
                )
            elif where not in f:
                report.fail(
                    sec, f"event scalar `{name}` expects dataset {where}: not found"
                )
            elif not _numeric(f[where]) or f[where].ndim != 1:
                report.fail(
                    sec,
                    f"{where}: shape {f[where].shape} dtype {f[where].dtype}, expected a numeric (events,) dataset",
                )
            elif length(where, f[where]):
                report.ok(sec, f"event scalar `{name}` from {where}")

        # ---- labels, weights, folds
        key = next((k for k in loader.LABEL_KEYS if k in f), None)
        label = None
        if key is None:
            report.fail(
                sec,
                f"no labels: expected {' or '.join(loader.LABEL_KEYS)} (the loader would set every label to 0)",
            )
        elif f[key].dtype.kind not in "iub" or f[key].ndim != 1:
            report.fail(
                sec,
                f"{key}: shape {f[key].shape} dtype {f[key].dtype}, expected integer (events,)",
            )
        elif length(key, f[key]):
            label = f[key][...].astype(np.int64)
            bad = np.unique(label[(label < 0) | (label > len(spec.outputs))])
            if len(bad):
                report.fail(
                    sec,
                    f"{key}: labels {bad.tolist()} outside 0..{len(spec.outputs)} (classes {spec.classes})",
                )
            else:
                counts = {
                    c: int((label == i).sum()) for i, c in enumerate(spec.classes)
                }
                report.ok(sec, f"labels from {key}: events per class {counts}")
                for i, c in enumerate(spec.classes[1:], start=1):
                    if counts[c] == 0:
                        report.warn(
                            sec, f"class `{c}` (label {i}) has no events in this file"
                        )
        if loader.WEIGHT_KEY not in f:
            report.warn(
                sec, f"{loader.WEIGHT_KEY} not found: every event gets weight 1"
            )
        else:
            w = f[loader.WEIGHT_KEY]
            if not _numeric(w) or w.ndim != 1:
                report.fail(
                    sec,
                    f"{loader.WEIGHT_KEY}: shape {w.shape} dtype {w.dtype}, expected numeric (events,)",
                )
            elif length(loader.WEIGHT_KEY, w):
                wv = w[...].astype(np.float64)
                if not np.isfinite(wv).all():
                    report.fail(
                        sec,
                        f"{loader.WEIGHT_KEY}: {int((~np.isfinite(wv)).sum())} non-finite weights",
                    )
                else:
                    n_neg = int((wv < 0).sum())
                    report.ok(sec, f"{loader.WEIGHT_KEY}: finite")
                    if n_neg:
                        report.warn(
                            sec,
                            f"{loader.WEIGHT_KEY}: {n_neg:,} negative weights ({n_neg / len(wv):.3g} of the "
                            "events); they count as 0 in the loss and in the metrics",
                        )
        column = None  # EVENT/fold, when present and well-formed
        if loader.FOLD_KEY in f:
            fo = f[loader.FOLD_KEY]
            if fo.dtype.kind not in "iu" or fo.ndim != 1:
                report.fail(
                    sec,
                    f"{loader.FOLD_KEY}: shape {fo.shape} dtype {fo.dtype}, expected integer (events,)",
                )
            elif length(loader.FOLD_KEY, fo):
                column = fo[...].astype(np.int64)
        elif n_folds is None:
            report.fail(
                sec,
                f"{loader.FOLD_KEY} not found: corenet-train needs a fold index per event; "
                "pass --n-folds K to use fold = event % K from EVENT/event",
            )
        folds, source = None, None
        if n_folds is not None:
            source = f"event % {n_folds} (--n-folds)"
            if loader.EVENT_KEY not in f:
                report.fail(
                    sec,
                    f"folds = {source}: {loader.EVENT_KEY} not found",
                )
            else:
                try:
                    folds = loader.fold_of(f[loader.EVENT_KEY][...], n_folds)
                except ValueError as e:
                    report.fail(sec, f"folds = {source}: {e}")
        elif column is not None:
            folds, source = column, loader.FOLD_KEY
        if folds is not None:
            vals, cnt = np.unique(folds, return_counts=True)
            report.ok(
                sec,
                f"folds from {source}: events per fold {dict(zip(vals.tolist(), cnt.tolist()))}",
            )
            if len(vals) < 2:
                report.warn(
                    sec,
                    f"folds from {source}: a single fold; training on fold k uses the other folds",
                )
            if n_folds is not None and column is not None:
                n_diff = int((column != folds).sum())
                if n_diff:
                    report.warn(
                        sec,
                        f"{loader.FOLD_KEY} disagrees with event % {n_folds} for {n_diff:,} of "
                        f"{len(folds):,} events; corenet-train --n-folds {n_folds} ignores the column",
                    )
                else:
                    report.ok(sec, f"{loader.FOLD_KEY} equals event % {n_folds}")
        if loader.EVENT_KEY not in f:
            report.warn(
                sec,
                f"{loader.EVENT_KEY} not found: events are numbered 0..N-1 in the prediction files",
            )

    if report.errors:
        return
    # ---- the actual loader and a forward pass
    try:
        ds = loader.H5Dataset(path, spec, rest_order=rest_order, n_folds=n_folds)
    except Exception as e:
        report.fail(sec, f"the loader (H5Dataset) fails: {type(e).__name__}: {e}")
        return
    for name, x in ds.inputs.items():
        if name == "event":
            bad = ~torch.isfinite(x)
            if bad.any():
                report.fail(sec, f"event scalars: {int(bad.sum())} non-finite values")
            continue
        present = x[:, -1] > 0
        feats = x[:, :-1].transpose(1, 2)[present]
        if not torch.isfinite(feats).all():
            report.fail(
                sec,
                f"leaf `{name}`: non-finite features in {int((~torch.isfinite(feats)).any(1).sum())} present objects",
            )
        filled = float(present.all(1).float().mean())
        report.ok(
            sec,
            f"leaf `{name}`: tensor {tuple(x.shape)}, all {x.shape[2]} slot(s) filled in {filled:.1%} of the events",
        )
    if report.errors:
        return
    n = min(n_forward, len(ds))
    model = CoReNet(spec, device="cpu")
    batch = {k: v[:n] for k, v in ds.inputs.items()}
    with torch.no_grad():
        model.train()
        model.updateMeanStd(batch)
        model.initMeanStd()
        model.eval()
        model.setGhostBatches(0)
        out = model(batch)
    finite = all(
        torch.isfinite(t).all()
        for t in [out["logits"], *out["views"].values(), *out["heads"].values()]
    )
    if finite:
        report.ok(
            sec,
            f"loader + forward pass on {n} events: finite outputs, logits {tuple(out['logits'].shape)}",
        )
    else:
        report.fail(sec, f"forward pass on {n} events gives non-finite outputs")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corenet-validate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", required=True, help="network configuration (.yaml or .json)"
    )
    p.add_argument(
        "--input",
        "--data",
        dest="input",
        default=None,
        help="HDF5 file to check against the configuration",
    )
    p.add_argument(
        "--rest-order",
        choices=["pt", "selector"],
        default=None,
        help="ordering of the 'rest' selector (default: the trainer section, else pt)",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="report the fold = event %% K split of EVENT/event (as corenet-train --n-folds K uses) "
        "instead of the EVENT/fold column",
    )
    p.add_argument(
        "--json", default=None, help="also write the report to this JSON file"
    )
    return p


def validate(config, data=None, rest_order=None, n_folds=None) -> Report:
    report = Report()
    spec = check_config(config, report)
    if spec is not None and data is not None:
        if rest_order is None:
            try:
                rest_order = (read_config_dict(config).get("trainer") or {}).get(
                    "rest_order", "pt"
                )
            except Exception:
                rest_order = "pt"
        check_data(data, spec, report, rest_order=rest_order, n_folds=n_folds)
    elif data is not None:
        report.fail(f"data {data}", "not checked: the configuration is invalid")
    return report


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_folds is not None and args.n_folds < 1:
        build_parser().error(f"--n-folds must be >= 1, got {args.n_folds}")
    report = validate(args.config, args.input, args.rest_order, args.n_folds)
    report.print()
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=1))
    return 1 if report.errors else 0


def cli():
    sys.exit(main())


if __name__ == "__main__":
    cli()
