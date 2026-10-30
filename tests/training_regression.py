"""Training-mode numerics of CoReNet on a fixed batch of a shipped example file: the computation behind
the golden-file regression test (tests/test_training_regression.py) and its generator
(scripts/make_golden.py), kept in one place so that both compute exactly the same thing.

One case = one example configuration on its small example file, on the CPU with THREADS intra-op threads.
Stages (array names "<stage>:<name>"):

  init:     state dict after torch.manual_seed(SEED); the test loads the stored one, so the stages below
            do not depend on the random-number stream (test_seeded_initialisation checks the stream)
  stats:    every GBN mean / std buffer (*.m, *.s) after updateMeanStd + initMeanStd on the batch
  train1:   training-mode outputs with GHOST_BATCHES ghost batches (logits, views, heads, event pixel)
  train2:   the same outputs from a second call
  grad:     gradients of the first of STEPS SGD steps (cross entropy + squared auxiliary heads)
  loss:     the loss of every step
  trained:  every floating-point state-dict entry after the STEPS steps
  eval:     eval-mode outputs after the steps

Stored as tests/golden/<case>.npz: one array per name plus "__meta__" (a JSON string).
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from corenet.data import H5Dataset
from corenet.models.config import load_config
from corenet.models.network import CoReNet

REPO = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO / "tests" / "golden"
CASES = {
    "hh4b": (REPO / "examples/hh4b/hh4b.yaml", REPO / "examples/data/hh4b_small.h5"),
    "bbww": (REPO / "examples/bbww/bbww.yaml", REPO / "examples/data/bbww_small.h5"),
}
N_EVENTS = 1024  # every (len // N_EVENTS)-th event of the file
GHOST_BATCHES = 4  # 256 events per ghost batch
SEED = 3
STEPS = 5
# plain SGD: linear in the gradient. Adam's first step is lr * sign(gradient), which turns the rounding
# noise of a gradient that cancels into a full step that differs between machines.
LR = 0.1
THREADS = 1
TOL = 1e-4  # largest relative error allowed per array (see compare)
FLOOR = 1e-3  # scale floor of an array: FLOOR x the largest |value| of its stage
RECIPE = ("n_events", "ghost_batches", "seed", "steps", "optimizer", "threads")


@contextmanager
def fixed_threads(n):
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta(case) -> dict:
    cfg, data = CASES[case]
    return {
        "case": case,
        "config": cfg.relative_to(REPO).as_posix(),
        "input": data.relative_to(REPO).as_posix(),
        "input_md5": file_md5(data),
        "n_events": N_EVENTS,
        "ghost_batches": GHOST_BATCHES,
        "seed": SEED,
        "steps": STEPS,
        "optimizer": f"SGD lr={LR}",
        "threads": THREADS,
        "tol": TOL,
    }


def save(file, arrays: dict, info: dict) -> None:
    np.savez_compressed(
        file, __meta__=np.array(json.dumps(info, sort_keys=True)), **arrays
    )


def load(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
        info = json.loads(str(z["__meta__"]))
    return arrays, info


def stage(arrays: dict, name: str) -> dict:
    p = name + ":"
    return {k[len(p) :]: v for k, v in arrays.items() if k.startswith(p)}


def _model(case) -> CoReNet:
    torch.manual_seed(SEED)
    return CoReNet(load_config(CASES[case][0]), "cpu")


def _numpy(t) -> np.ndarray:
    return t.detach().cpu().numpy().copy()


def initial_state(case) -> dict:
    return {k: _numpy(v) for k, v in _model(case).state_dict().items()}


def perturb(case, init: dict) -> dict:
    """``init`` with every parameter (not buffer) multiplied by 1 +- 2^-23 (random signs, seed 0): one
    float32 rounding of the starting point, the noise probe of scripts/make_golden.py."""
    names = {k for k, _ in _model(case).named_parameters()}
    rng = np.random.default_rng(0)
    out = {}
    for k, v in init.items():
        if k in names:
            sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=v.shape)
            v = (v * (np.float32(1) + sign * np.float32(2.0**-23))).astype(v.dtype)
        out[k] = v
    return out


def _batch(case):
    cfg, data = CASES[case]
    ds = H5Dataset(data, load_config(cfg))
    idx = torch.arange(N_EVENTS) * (len(ds) // N_EVENTS)
    return {k: v[idx].clone() for k, v in ds.inputs.items()}, ds.labels[idx].clone()


def _outputs(name, out, arrays):
    def walk(path, x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(f"{path}.{k}" if path else str(k), v)
        else:
            arrays[f"{name}:{path}"] = _numpy(x)

    walk("", out)


def compute(case, init: dict | None = None, threads: int = THREADS) -> dict:
    """The arrays of one case (module docstring). ``init``: initial state dict to load (numpy arrays as
    stored under ``init:``); None = the state after torch.manual_seed(SEED)."""
    x, label = _batch(case)

    def inputs():
        return {k: v.clone() for k, v in x.items()}

    arrays = {}
    with fixed_threads(threads):
        model = _model(case)
        if init is not None:
            model.load_state_dict(
                {k: torch.from_numpy(np.array(v)) for k, v in init.items()}, strict=True
            )
        for k, v in model.state_dict().items():
            arrays[f"init:{k}"] = _numpy(v)
        model.train()
        model.updateMeanStd(inputs())
        model.initMeanStd()
        for k, v in model.state_dict().items():
            if k.endswith((".m", ".s")) and v.is_floating_point():
                arrays[f"stats:{k}"] = _numpy(v)
        model.setGhostBatches(GHOST_BATCHES)
        for name in ("train1", "train2"):
            with torch.no_grad():
                _outputs(name, model(inputs()), arrays)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        losses = []
        for step in range(STEPS):
            out = model(inputs())
            loss = torch.nn.functional.cross_entropy(out["logits"], label)
            for h in out["heads"].values():
                loss = loss + h.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            if step == 0:
                for k, p in model.named_parameters():
                    if p.grad is not None:
                        arrays[f"grad:{k}"] = _numpy(p.grad)
            opt.step()
            losses.append(loss.item())
        arrays["loss:sgd"] = np.array(losses)
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                arrays[f"trained:{k}"] = _numpy(v)
        model.eval()
        model.setGhostBatches(0)
        with torch.no_grad():
            _outputs("eval", model(inputs()), arrays)
    return arrays


def compare(actual: dict, golden: dict):
    """Relative error of every golden array: max |actual - golden| / scale, scale = max |golden| of that
    array, floored at FLOOR x the largest |golden| of its stage (an array that is zero up to rounding, such
    as a gradient that cancels, is not measured against its own noise). Integer arrays must be equal
    (error 0 or inf); a shape mismatch or a non-finite value is inf.
    Returns ({name: error}, names present on one side only)."""
    stage_max = {}
    for k, g in golden.items():
        if np.issubdtype(g.dtype, np.floating) and g.size:
            s = k.split(":", 1)[0]
            stage_max[s] = max(stage_max.get(s, 0.0), float(np.abs(g).max()))
    errors = {}
    for k, g in golden.items():
        if k not in actual:
            continue
        a = actual[k]
        if a.shape != g.shape:
            errors[k] = np.inf
        elif not np.issubdtype(g.dtype, np.floating):
            errors[k] = 0.0 if np.array_equal(a, g) else np.inf
        elif g.size == 0:
            errors[k] = 0.0
        elif not np.all(np.isfinite(a)):
            errors[k] = np.inf
        else:
            floor = FLOOR * stage_max[k.split(":", 1)[0]]
            scale = max(float(np.abs(g).max()), floor, 1e-30)
            diff = np.abs(a.astype(np.float64) - g.astype(np.float64)).max()
            errors[k] = float(diff) / scale
    return errors, sorted(set(golden) ^ set(actual))
