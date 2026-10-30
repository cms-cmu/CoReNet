"""Regression test of CoReNet's training-mode numerics against golden files (tests/golden/<case>.npz).

All slots of one leaf role in one node are embedded in one call and share their ghost-batch statistics, so
training-mode outputs, gradients and optimiser steps no longer match the research code bit for bit
(tests/reference compares eval mode only). This test pins them to the stored values of this code: the
HH -> 4b and HH -> bbWW example configurations on 1,024 events of the shipped small example files, CPU,
one thread. What is stored and the tolerance: tests/training_regression.py. After an intentional change of
the numerics: python scripts/make_golden.py (CONTRIBUTING.md).
"""

from __future__ import annotations

import numpy as np
import pytest
import training_regression as tr

CASES = list(tr.CASES)
REGEN = "regenerate with `python scripts/make_golden.py` (CONTRIBUTING.md)"


def _golden(case):
    path = tr.GOLDEN_DIR / f"{case}.npz"
    data = tr.CASES[case][1]
    if not data.exists():
        pytest.skip(f"{data} not found")
    assert path.exists(), f"{path} missing: {REGEN}"
    golden, info = tr.load(path)
    current = tr.meta(case)
    for key in ("input_md5", *tr.RECIPE):
        assert info.get(key) == current[key], (
            f"{case}: {key} is {current[key]!r}, the golden file was made with "
            f"{info.get(key)!r}: {REGEN}"
        )
    return golden, info


@pytest.mark.parametrize("case", CASES)
def test_seeded_initialisation(case):
    golden, _ = _golden(case)
    init, actual = tr.stage(golden, "init"), tr.initial_state(case)
    assert actual.keys() == init.keys(), (
        f"{case}: state-dict keys differ from the golden file"
    )
    for k in init:
        assert np.array_equal(actual[k], init[k]), (
            f"{case}: {k} after torch.manual_seed({tr.SEED}) differs from the golden file"
        )


@pytest.mark.parametrize("case", CASES)
def test_training_numerics(case):
    golden, info = _golden(case)
    actual = tr.compute(case, init=tr.stage(golden, "init"))
    errors, one_sided = tr.compare(actual, golden)
    assert not one_sided, f"{case}: arrays present on one side only: {one_sided[:10]}"
    worst = sorted(((e, k) for k, e in errors.items()), reverse=True)
    bad = [(e, k) for e, k in worst if not e <= tr.TOL]
    assert not bad, (
        f"{case}: {len(bad)} of {len(errors)} arrays differ from the golden file (made at "
        f"{info.get('commit')}) by more than {tr.TOL:g} (relative); largest: "
        + "; ".join(f"{k} {e:.2e}" for e, k in bad[:10])
    )
    print(
        f"    {case}: {len(errors)} arrays within {tr.TOL:g}; largest relative error "
        f"{worst[0][0]:.1e} ({worst[0][1]})"
    )
