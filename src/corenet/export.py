"""
corenet-export: export a trained CoReNet to TorchScript and ONNX and check the exported models.

    corenet-export --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \\
        --output-dir runs/hh4b/export

Writes ``<name>.pt`` (TorchScript, ``torch.jit.trace``), ``<name>.onnx`` and ``<name>.json`` (input /
output names and shapes, classes). The exported graph starts after the role selection: its inputs are
the role tensors and the event scalars exactly as ``corenet.data.H5Dataset`` produces them (one
positional input per leaf in the order listed in the JSON, then ``event``), with a dynamic batch
dimension. Its outputs are probabilities:

    class_probabilities       (N, nClasses)   softmax of the main head, classes ["rest", <outputs>...]
    views_<hypothesis>        (N, V)          softmax over the candidates of every multi-candidate hypothesis
    head_<output>             (N,)            sigmoid of every auxiliary head

After writing, both files are reloaded (TorchScript with ``torch.jit.load``, ONNX with onnxruntime) and
evaluated on ``--n-events`` events of ``--input`` (and on a second, different batch size, to check the
dynamic batch dimension); the maximum absolute difference to the PyTorch model is reported and must be
below ``--atol`` (1e-4; float32 rounding differences are ~1e-5).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import load_checkpoint, n_folds_of
from .data.h5 import H5Dataset


class ExportWrapper(nn.Module):
    """Positional-tensor front end of a CoReNet (eval mode) returning a flat tuple of probabilities."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_names = list(model.leaf_order) + (["event"] if model.dA else [])
        self.view_names = [h for h in model.spec.hypotheses if h in model.selector]
        self.head_names = [o.name for o in model.spec.outputs[1:]]
        self.output_names = (
            ["class_probabilities"]
            + [f"views_{h}" for h in self.view_names]
            + [f"head_{o}" for o in self.head_names]
        )

    def forward(self, *tensors):
        out = self.model(dict(zip(self.input_names, tensors)))
        res = [F.softmax(out["logits"], dim=1)]
        res += [F.softmax(out["views"][h], dim=1) for h in self.view_names]
        res += [torch.sigmoid(out["heads"][o]) for o in self.head_names]
        return tuple(res)


def _batch(ds: H5Dataset, names, start, n):
    return tuple(ds.inputs[k][start : start + n].contiguous() for k in names)


def _max_diff(ref, got) -> float:
    return max(
        float((torch.as_tensor(np.asarray(g)) - r).abs().max())
        for r, g in zip(ref, got)
    )


def export_torchscript(wrapper, example, path: Path):
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False, strict=False)
    traced.save(str(path))
    return path


def _register_onnx_symbolics(opset: int):
    """The TorchScript-based ONNX exporter has no mapping for aten::sinh / aten::asinh although ONNX has
    native Sinh / Asinh operators (opset >= 9); registered so that a graph using them still exports."""
    from torch.onnx import register_custom_op_symbolic

    register_custom_op_symbolic("aten::sinh", lambda g, x: g.op("Sinh", x), opset)
    register_custom_op_symbolic("aten::asinh", lambda g, x: g.op("Asinh", x), opset)


def export_onnx(wrapper, example, path: Path, opset: int):
    _register_onnx_symbolics(opset)
    dynamic_axes = {k: {0: "batch"} for k in wrapper.input_names + wrapper.output_names}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            str(path),
            input_names=wrapper.input_names,
            output_names=wrapper.output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corenet-export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True, help="model_fold{k}.pkl written by corenet-train"
    )
    p.add_argument(
        "--input",
        required=True,
        help="HDF5 file providing the example / validation events",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--name",
        default=None,
        help="basename of the exported files (default: the checkpoint's)",
    )
    p.add_argument(
        "--config",
        default=None,
        help="configuration file (default: the one stored in the checkpoint)",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["torchscript", "onnx"],
        default=["torchscript", "onnx"],
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument(
        "--n-events",
        type=int,
        default=1024,
        help="events used to check the exported models",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="maximum allowed |exported - PyTorch| of any output probability (float32 rounding is ~1e-5)",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    model, spec, ckpt = load_checkpoint(args.checkpoint, args.config, "cpu")
    wrapper = ExportWrapper(model).eval()
    meta = ckpt.get("metadata", {})
    ds = H5Dataset(
        args.input,
        spec,
        rest_order=meta.get("rest_order", "pt"),
        n_folds=n_folds_of(meta),
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.checkpoint).stem

    n = min(args.n_events, len(ds))
    n2 = max(1, min(n // 2 + 3, len(ds) - n)) if len(ds) > n else max(1, n // 2 + 3)
    example = _batch(ds, wrapper.input_names, 0, n)
    checks = [example, _batch(ds, wrapper.input_names, n if len(ds) > n else 0, n2)]
    with torch.no_grad():
        refs = [wrapper(*x) for x in checks]

    meta = {
        "checkpoint": str(args.checkpoint),
        "classes": spec.classes,
        "inputs": {k: [None, *ds.inputs[k].shape[1:]] for k in wrapper.input_names},
        "outputs": wrapper.output_names,
        "files": {},
        "max_abs_diff": {},
    }
    status = {}
    if "torchscript" in args.formats:
        path = export_torchscript(wrapper, example, out_dir / f"{name}.pt")
        loaded = torch.jit.load(str(path))
        with torch.no_grad():
            diff = max(
                _max_diff(r, [t.numpy() for t in loaded(*x)])
                for r, x in zip(refs, checks)
            )
        meta["files"]["torchscript"], meta["max_abs_diff"]["torchscript"] = (
            path.name,
            diff,
        )
        status["torchscript"] = diff <= args.atol
        logging.info(
            f"TorchScript: {path} | max |diff| to PyTorch = {diff:.3e} on batches of {n} and {n2} events"
        )
    if "onnx" in args.formats:
        import onnx
        import onnxruntime as ort

        path = export_onnx(wrapper, example, out_dir / f"{name}.onnx", args.opset)
        onnx.checker.check_model(onnx.load(str(path)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = (
            torch.get_num_threads()
        )  # not every core of a shared node
        sess = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        feed_names = [i.name for i in sess.get_inputs()]
        diffs = []
        for r, x in zip(refs, checks):
            feeds = {
                k: v.numpy() for k, v in zip(wrapper.input_names, x) if k in feed_names
            }
            diffs.append(_max_diff(r, sess.run(wrapper.output_names, feeds)))
        diff = max(diffs)
        meta["files"]["onnx"], meta["max_abs_diff"]["onnx"] = path.name, diff
        meta["onnx_opset"] = args.opset
        status["onnx"] = diff <= args.atol
        logging.info(
            f"ONNX (opset {args.opset}): {path} | max |diff| to PyTorch = {diff:.3e} on batches of {n} and {n2} events"
        )
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))
    failed = [k for k, ok in status.items() if not ok]
    if failed:
        raise SystemExit(
            f"exported {failed} differ from PyTorch by more than --atol={args.atol}: {meta['max_abs_diff']}"
        )
    logging.info(f"wrote {out_dir / (name + '.json')}")
    return meta


def cli():
    """Console entry point (exit status 0; ``main`` returns the result for programmatic use)."""
    main()


if __name__ == "__main__":
    cli()
