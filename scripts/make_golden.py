#!/usr/bin/env python3
"""Regenerate the golden files of the training-mode regression test (tests/golden/<case>.npz).

    python scripts/make_golden.py          # both cases: tests/golden/hh4b.npz, tests/golden/bbww.npz
    python scripts/make_golden.py hh4b     # one case

Run it on the CPU, only after an intentional change of the training-mode numerics (or of
examples/data/<case>_small.h5), and commit the files with that change. Per case it prints the change
against the file it replaces (per stage, the largest relative error as defined in
tests/training_regression.py) and two noise probes: another thread count, and the initial weights moved by
one float32 rounding. It writes nothing when a probe exceeds a tenth of the test tolerance, when a value is
not finite, or when the golden files would exceed 2 MB in total.
"""

from __future__ import annotations

import argparse
import datetime
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import training_regression as tr  # noqa: E402

PROBE_THREADS = 4
MAX_BYTES = 2_000_000


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _by_stage(errors: dict) -> dict:
    out = {}
    for k, e in errors.items():
        s = k.split(":", 1)[0]
        out[s] = max(out.get(s, 0.0), e)
    return out


def _fmt(by_stage: dict) -> str:
    return "  ".join(f"{s} {e:.1e}" for s, e in by_stage.items())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "cases", nargs="*", help=f"cases to regenerate {list(tr.CASES)} [all]"
    )
    args = p.parse_args(argv)
    cases = args.cases or list(tr.CASES)
    unknown = [c for c in cases if c not in tr.CASES]
    if unknown:
        p.error(f"unknown case(s) {unknown}; choose from {list(tr.CASES)}")
    commit = _git("rev-parse", "--short", "HEAD")
    src_dirty = _git("status", "--porcelain", "--", "src") != ""
    blobs, ok = {}, True
    for case in cases:
        arrays = tr.compute(case)  # initial state from torch.manual_seed(tr.SEED)
        init = tr.stage(arrays, "init")
        nonfinite = [
            k
            for k, v in arrays.items()
            if np.issubdtype(v.dtype, np.floating) and not np.all(np.isfinite(v))
        ]
        if nonfinite:
            print(f"{case}: non-finite values in {nonfinite[:10]}")
            ok = False
        probes = {
            f"threads={PROBE_THREADS}": tr.compute(
                case, init=init, threads=PROBE_THREADS
            ),
            "init+1ulp": tr.compute(case, init=tr.perturb(case, init)),
        }
        for name, other in probes.items():
            by_stage = _by_stage(tr.compare(other, arrays)[0])
            print(f"{case}: probe {name}: {_fmt(by_stage)}")
            if max(by_stage.values()) > tr.TOL / 10:
                print(f"{case}: probe {name} exceeds TOL/10 = {tr.TOL / 10:g}")
                ok = False
        old = tr.GOLDEN_DIR / f"{case}.npz"
        if old.exists():
            g_old, m_old = tr.load(old)
            errors, one_sided = tr.compare(arrays, g_old)
            print(
                f"{case}: change against {old.relative_to(REPO)} (made at {m_old.get('commit')}): "
                f"{_fmt(_by_stage(errors))}; arrays added/removed: {len(one_sided)} {one_sided[:10]}"
            )
        else:
            print(f"{case}: no previous golden file")
        info = tr.meta(case) | {
            "commit": commit,
            "src_dirty": src_dirty,
            "date": datetime.date.today().isoformat(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
        buf = io.BytesIO()
        tr.save(buf, arrays, info)
        blobs[case] = buf.getvalue()
        print(
            f"{case}: {len(arrays)} arrays, {len(blobs[case]):,} bytes; "
            f"losses {np.round(arrays['loss:sgd'], 6).tolist()}"
        )
    others = (
        [f for f in tr.GOLDEN_DIR.glob("*.npz") if f.stem not in blobs]
        if tr.GOLDEN_DIR.exists()
        else []
    )
    total = sum(len(b) for b in blobs.values()) + sum(f.stat().st_size for f in others)
    print(f"golden files in total: {total:,} bytes (limit {MAX_BYTES:,})")
    if total > MAX_BYTES:
        ok = False
    if not ok:
        print("NOT WRITTEN")
        return 1
    tr.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for case, blob in blobs.items():
        path = tr.GOLDEN_DIR / f"{case}.npz"
        path.write_bytes(blob)
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
