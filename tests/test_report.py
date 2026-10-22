"""corenet-report on 2-epoch synthetic trainings (CPU): the HTML file exists, is small, holds every section,
renders the pairing accuracy when the file has TARGETS and the overtraining check when corenet-evaluate
predictions of every event exist, and leaves those figures out when they do not."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
from synthetic import write_synthetic
from test_train import BBWW, HH4B, _train

from corenet import evaluate, report

HEADINGS = [f"{i + 1}. {t}" for i, (_, t) in enumerate(report.SECTIONS)]
MAX_BYTES = 4_000_000


def _add_targets(path: Path) -> None:
    """TARGETS/h1/{b1,b2} and TARGETS/h2/{b1,b2}: four distinct present jet slots for signal events, -1 otherwise."""
    rng = np.random.default_rng(1)
    with h5py.File(path, "a") as f:
        label = f["EVENT/label"][...]
        n_present = f["INPUTS/Source/MASK"][...].sum(1)
        cols = {
            k: np.full(len(label), -1, dtype=np.int64)
            for k in ("h1/b1", "h1/b2", "h2/b1", "h2/b2")
        }
        for i in np.flatnonzero(label == 1):
            slots = rng.choice(int(n_present[i]), 4, replace=False)
            for k, s in zip(cols, slots):
                cols[k][i] = s
        for k, v in cols.items():
            f.create_dataset(f"TARGETS/{k}", data=v)


def _evaluate_all(out_dir: Path, data: Path) -> None:
    evaluate.main(
        [
            "--checkpoint",
            str(out_dir / "model_fold0.pkl"),
            "--input",
            str(data),
            "--output",
            str(out_dir / "eval_fold0.h5"),
            "--fold",
            "-1",
        ]
    )


def _report(out_dir: Path, data: Path, *extra: str) -> str:
    out = report.main(
        [
            "--run-dir",
            str(out_dir),
            "--input",
            str(data),
            "--output",
            str(out_dir / "report.html"),
            *extra,
        ]
    )
    assert out.is_file(), out
    assert out.stat().st_size < MAX_BYTES, out.stat().st_size
    page = out.read_text(encoding="utf-8")
    assert len(HEADINGS) == 8, HEADINGS
    assert re.findall(r"<h2>(.*?)</h2>", page) == HEADINGS, (
        "the page must hold exactly the eight sections in order"
    )
    low = page.lower()
    assert "not produced" not in low, "the 'Not produced' list is back"
    assert "wall time" not in low, "the wall-time column / note is back"
    assert "could not be produced" not in page, "a section failed"
    assert "data:image/png;base64," in page
    return page


def test_report_with_targets_and_all_events() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = write_synthetic(tmp / "hh4b_all.h5", "hh4b")
        _add_targets(data)
        out_dir = tmp / "run"
        _train(data, out_dir)
        _evaluate_all(out_dir, data)
        page = _report(out_dir, data)
        assert "Pairing accuracy of the argmax view of HH" in page
        assert re.search(
            r"<td>mean ± sd</td><td>[0-9.]+</td><td>[0-9.]+</td><td>[0-9.]+</td>", page
        )
        assert "<td>test, train, validation</td></tr>" in page
        assert "overtraining check" in page
        assert "AUC of the main score" in page
        assert "H1 ← b" in page and "HH context" in page and "event scalars" in page
        assert "Role b (jets btag 4)" in page and "Single object" not in page


def test_report_without_targets_and_train_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = write_synthetic(tmp / "hh4b_all.h5", "hh4b")
        out_dir = tmp / "run"
        _train(data, out_dir)
        page = _report(out_dir, data, "--max-events", "500")
        # hypotheses: view probabilities and argmax shares, but no pairing accuracy (no TARGETS)
        assert "Highest view probability per event and class on the test folds" in page
        assert "Weighted share of the argmax view, test folds: HH" in page
        assert "Pairing accuracy" not in page and "against TARGETS" not in page
        # classification: AUC on the test fold only, no train / validation curves, no overtraining overlay
        assert re.search(
            r"<td>mean ± sd</td><td>[0-9.]+</td><td>-</td><td>-</td>", page
        )
        assert "overtraining check" not in page
        assert "<td>test</td></tr>" in page and "test, train, validation" not in page
        low_page = page.lower()
        assert "pairing accuracy" not in low_page
        assert "No predictions on the" not in page
        assert "random subsample" in page


def test_report_bbww_multiclass() -> None:
    from corenet import train

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = write_synthetic(tmp / "bbww_all.h5", "bbww")
        out_dir = tmp / "run"
        train.main(
            [
                "--config",
                str(BBWW),
                "--input",
                str(data),
                "--output-dir",
                str(out_dir),
                "--fold",
                "0",
                "--exclude-folds",
                "1",
                "--epochs",
                "2",
                "--batch-size-init",
                "128",
                "--freeze-epoch",
                "1",
                "--device",
                "cpu",
            ]
        )
        _evaluate_all(out_dir, data)
        page = _report(out_dir, data)
        assert "Confusion matrix" in page
        assert "head ttbar" in page
        assert "Weighted share of the argmax view, test folds: tt" in page
        assert "Single object lepton" in page and "Single object met" in page
        assert "relation lepton - met" in page


def test_report_needs_a_run_directory() -> None:
    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(SystemExit):
            report.main(
                [
                    "--run-dir",
                    tmp,
                    "--input",
                    str(HH4B),
                    "--output",
                    str(Path(tmp) / "r.html"),
                ]
            )
