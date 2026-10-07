#!/usr/bin/env python3
"""Write a small random subsample of a CoReNet HDF5 file with the same schema.

Every dataset whose first dimension is the number of events is subsampled with the same (sorted)
event indices; other datasets and all group attributes are copied unchanged. The subsample is
stratified by class label (``EVENT/label``, else ``CLASSIFICATIONS/EVENT/signal``), so each class keeps
its fraction of events, and the weights are copied unchanged (corenet-train re-normalises them per
class over the training set by default).

    python scripts/make_small_h5.py examples/data/hh4b_all.h5 examples/data/hh4b_small.h5 --events 20000
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--events", type=int, default=20000, help="number of events to keep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-attrs", nargs="*", default=[], help="file-level attributes not to copy")
    args = p.parse_args(argv)

    with h5py.File(args.input, "r") as src:
        key = "EVENT/label" if "EVENT/label" in src else "CLASSIFICATIONS/EVENT/signal"
        label = src[key][...]
        n = len(label)
        rng = np.random.default_rng(args.seed)
        frac = min(1.0, args.events / n)
        keep = []
        for c in np.unique(label):
            idx = np.flatnonzero(label == c)
            keep.append(rng.choice(idx, size=int(round(frac * len(idx))), replace=False))
        keep = np.sort(np.concatenate(keep))

        with h5py.File(args.output, "w") as dst:
            for k, v in src.attrs.items():
                if k not in args.drop_attrs:
                    dst.attrs[k] = v
            dst.attrs["subsample_of"] = str(args.input).split("/")[-1]
            dst.attrs["subsample_seed"] = args.seed
            dst.attrs["subsample_events"] = len(keep)

            def copy(name, obj):
                if isinstance(obj, h5py.Group):
                    g = dst.require_group(name)
                    for k, v in obj.attrs.items():
                        g.attrs[k] = v
                    return
                data = obj[...]
                if data.ndim and data.shape[0] == n:
                    data = data[keep]
                d = dst.create_dataset(name, data=data, dtype=obj.dtype)
                for k, v in obj.attrs.items():
                    d.attrs[k] = v

            src.visititems(copy)
    counts = {int(c): int((label[keep] == c).sum()) for c in np.unique(label)}
    print(f"{args.output}: {len(keep):,} of {n:,} events, per class {counts}")


if __name__ == "__main__":
    main()
