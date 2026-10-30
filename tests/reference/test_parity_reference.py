"""
Parity of CoReNet with the research code it was extracted from (the ``hcr`` package of the reference
repository, in which the network was called ``HCRGeneric``).

Skipped unless the environment variable ``CORENET_REFERENCE_REPO`` points to a checkout of that
repository (its ``src/`` is put on ``sys.path``; its data files ``data/h5/{hh4b,bbww}_all.h5`` and
checkpoints under ``models/`` are used when present). Nothing is written into the reference checkout.

    CORENET_REFERENCE_REPO=/path/to/reference pytest tests/reference -s

Checks, for the HH -> 4b and HH -> bbWW example configurations:
  * the example YAML files describe the same network as the reference configurations (sections other
    than ``training`` as written, ``training`` after the defaults are filled in); for HH -> bbWW the
    reference is the eta-reflection configuration ``configs/hcr/dscan/bbww_reflect_d8.yaml`` and its
    trained checkpoint ``models/bbww_d8_e20bestreflect_r3_best`` (the paper's model);
  * with the same seed both code bases initialise identical state dicts (same module creation order);
    parameter counts 1,508 and 3,832;
  * the two loaders produce identical input tensors from the same HDF5 file;
  * after updateMeanStd / initMeanStd on identical batches the frozen statistics are identical, except
    those of the leaf embedders, which agree to float32 rounding (CoReNet accumulates all slots of one
    leaf role in one node in one call); with the reference statistics copied, eval-mode logits, view
    logits, auxiliary heads and event pixels are bit-identical (torch.equal). Training-mode outputs,
    gradients and optimiser steps differ by design (the slots of one leaf role share the ghost-batch
    statistics) and are pinned by the golden-file test tests/test_training_regression.py;
  * the trained reference checkpoints load strictly into CoReNet and give torch.equal outputs;
  * the primitives (GBN, conv1d, MinimalAttention, ReinforceLayer, ResNetBlock) are bit-identical;
  * the two trainers, run for 2 epochs with the same arguments, write checkpoints with the same
    state-dict keys, shapes and integer buffers, predict the same events in the same order and follow
    the same learning-rate and batch-size schedule (the trained weights differ by design, see above);
  * CoReNet with the 4b configuration reproduces the reference repository's original hard-coded 4b
    network (``hcr.models.HCR.HCR``) loaded from its trained checkpoint, torch.equal.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REF = os.environ.get("CORENET_REFERENCE_REPO")
if not REF:
    pytest.skip("CORENET_REFERENCE_REPO is not set", allow_module_level=True)
REF = Path(REF)
sys.path.insert(0, str(REF / "src"))

from hcr.data.h5_generic import H5GenericDataset as RefDataset
from hcr.models import blocks as ref_blocks
from hcr.models import combinatorics as ref_comb
from hcr.models.config import parse_config as ref_parse
from hcr.models.hcr_generic import HCRGeneric

from corenet.data import H5Dataset
from corenet.models import blocks, combinatorics
from corenet.models.config import parse_config
from corenet.models.network import CoReNet

repo = Path(__file__).resolve().parents[2]
TOPOLOGIES = {
    "hh4b": dict(
        config=repo / "examples/hh4b/hh4b.yaml",
        ref_config=REF / "configs/hcr/hh4b.yaml",
        data=REF / "data/h5/hh4b_all.h5",
        params=1508,
        ckpt=REF / "models/hh4b_generic_check/model_fold0.pkl",
    ),
    "bbww": dict(
        config=repo / "examples/bbww/bbww.yaml",
        # the eta-reflection configuration of the paper (configs/hcr/bbww.yaml has reflect_eta: false)
        ref_config=REF / "configs/hcr/dscan/bbww_reflect_d8.yaml",
        data=REF / "data/h5/bbww_all.h5",
        params=3832,
        # the paper's bbWW model (best-validation epoch, fold 0), trained with bbww_reflect_d8.yaml
        ckpt=REF / "models/bbww_d8_e20bestreflect_r3_best/model_fold0.pkl",
    ),
}
N_EVENTS = 4096


def _cfg(topology):
    t = TOPOLOGIES[topology]
    return yaml.safe_load(t["config"].read_text()), yaml.safe_load(
        t["ref_config"].read_text()
    )


def _models(topology, seed=0):
    cfg, ref_cfg = _cfg(topology)
    torch.manual_seed(seed)
    ref = HCRGeneric(ref_parse(ref_cfg), device="cpu")
    torch.manual_seed(seed)
    new = CoReNet(parse_config(cfg), device="cpu")
    return ref, new


_DATA = {}


def _data(topology):
    """Identical input batches for both models: N_EVENTS random events of the full file, both loaders."""
    if topology not in _DATA:
        t = TOPOLOGIES[topology]
        if not t["data"].exists():
            pytest.skip(f"{t['data']} not found")
        cfg, ref_cfg = _cfg(topology)
        ref_ds = RefDataset(t["data"], ref_parse(ref_cfg))
        new_ds = H5Dataset(t["data"], parse_config(cfg))
        idx = torch.as_tensor(
            np.sort(
                np.random.default_rng(1).choice(len(new_ds), N_EVENTS, replace=False)
            )
        )
        _DATA[topology] = (ref_ds, new_ds, idx)
    return _DATA[topology]


def _batch(ds, idx):
    return {k: v[idx].clone() for k, v in ds.inputs.items()}


def _same_tree(a, b, what):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{what}: keys {sorted(a)} vs {sorted(b)}"
        for k in a:
            _same_tree(a[k], b[k], f"{what}.{k}")
        return
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    assert torch.equal(a, b), f"{what}: max |diff| = {(a - b).abs().max().item():.3e}"


def _same_state(ref, new, what):
    rs, ns = ref.state_dict(), new.state_dict()
    assert list(rs) == list(ns), f"{what}: state-dict keys differ"
    for k in rs:
        assert torch.equal(rs[k], ns[k]), f"{what}: {k} differs"


# ----------------------------------------------------------------------------------- configuration
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_example_config_matches_reference(topology):
    cfg, ref_cfg = _cfg(topology)

    def rest(c):
        return {k: v for k, v in c.items() if k != "training"}

    assert rest(cfg) == rest(ref_cfg), (
        f"{topology}: example configuration differs from the reference"
    )
    # training options after the defaults are filled in: an omitted key equal to its default passes
    new_t = dataclasses.asdict(parse_config(cfg).training)
    ref_t = dataclasses.asdict(ref_parse(ref_cfg).training)
    assert new_t == ref_t, f"{topology}: training options {new_t} vs {ref_t}"


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_initialisation_and_parameter_count(topology):
    ref, new = _models(topology)
    n_ref = sum(p.numel() for p in ref.parameters())
    n_new = sum(p.numel() for p in new.parameters())
    assert n_ref == n_new == TOPOLOGIES[topology]["params"], (n_ref, n_new)
    _same_state(
        ref, new, f"{topology} initial state"
    )  # same seed -> same weights, same key names
    print(
        f"    {topology}: {n_new} parameters in both, identical initial state dict ({len(new.state_dict())} tensors)"
    )


# ----------------------------------------------------------------------------------- loader
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_loader_parity(topology):
    ref_ds, new_ds, _ = _data(topology)
    _same_tree(ref_ds.inputs, new_ds.inputs, f"{topology} inputs")
    for a in ("labels", "weights", "folds", "events"):
        assert torch.equal(getattr(ref_ds, a), getattr(new_ds, a)), a
    for order in ("pt", "selector"):
        cfg, ref_cfg = _cfg(topology)
        r = RefDataset(
            TOPOLOGIES[topology]["data"], ref_parse(ref_cfg), rest_order=order
        )
        n = H5Dataset(TOPOLOGIES[topology]["data"], parse_config(cfg), rest_order=order)
        _same_tree(r.inputs, n.inputs, f"{topology} inputs rest_order={order}")
    print(
        f"    {topology}: loader tensors identical for {len(new_ds):,} events (rest_order pt and selector)"
    )


# ----------------------------------------------------------------------------------- forward (eval mode)
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_eval_forward_parity(topology):
    """Frozen statistics and eval-mode outputs. Training-mode outputs, gradients and optimiser steps are
    not compared: CoReNet embeds all slots of one leaf role in one node in one call (they share the
    ghost-batch statistics), the research code embedded every slot on its own;
    tests/test_training_regression.py pins them instead."""
    ref_ds, new_ds, idx = _data(topology)
    ref, new = _models(topology, seed=3)
    new.load_state_dict(
        ref.state_dict(), strict=True
    )  # explicit copy (no key renaming needed)
    ref.train()
    new.train()
    ref.updateMeanStd(_batch(ref_ds, idx))
    new.updateMeanStd(_batch(new_ds, idx))
    ref.initMeanStd()
    new.initMeanStd()
    # frozen statistics: identical, except those of the leaf embedders, which CoReNet accumulates over all
    # slots of a role in one call (the same pixels summed in another order: float32 rounding only)
    rs, ns = ref.state_dict(), new.state_dict()
    assert list(rs) == list(ns), f"{topology}: state-dict keys differ"
    rounded = []
    for k in rs:
        if torch.equal(rs[k], ns[k]):
            continue
        assert k.startswith("leaf_embed.") and k.endswith((".m", ".s")), (
            f"{topology} after initMeanStd: {k} differs"
        )
        torch.testing.assert_close(
            ns[k], rs[k], rtol=1e-5, atol=1e-6, msg=f"{topology} after initMeanStd: {k}"
        )
        rounded.append(k)
    new.load_state_dict(
        rs, strict=True
    )  # the reference statistics in both from here on
    ref.setGhostBatches(0)
    new.setGhostBatches(0)
    ref.eval()
    new.eval()
    for _ in range(2):
        with torch.no_grad():
            _same_tree(
                ref(_batch(ref_ds, idx)),
                new(_batch(new_ds, idx)),
                f"{topology} [eval] outputs",
            )
        _same_state(ref, new, f"{topology} [eval] buffers")
    print(
        f"    {topology}: {N_EVENTS} events: frozen statistics identical except {rounded} (float32 "
        f"rounding); eval-mode logits, views, heads, event pixel and buffers torch.equal"
    )


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_trained_checkpoint_parity(topology):
    ck_path = TOPOLOGIES[topology]["ckpt"]
    if ck_path is None or not ck_path.exists():
        pytest.skip(
            f"no trained {topology} checkpoint for this network"
            if ck_path is None
            else f"{ck_path} not found"
        )
    ref_ds, new_ds, idx = _data(topology)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ref, new = _models(topology)
    ref.load_state_dict(ck["model"], strict=True)
    new.load_state_dict(ck["model"], strict=True)
    ref.eval()
    new.eval()
    ref.setGhostBatches(0)
    new.setGhostBatches(0)
    with torch.no_grad():
        o_ref, o_new = ref(_batch(ref_ds, idx)), new(_batch(new_ds, idx))
    _same_tree(o_ref, o_new, f"{topology} trained outputs")
    p = torch.softmax(o_new["logits"], 1)[:, 1]
    print(
        f"    {topology}: {ck_path.relative_to(REF)} loads strictly; outputs torch.equal "
        f"(mean P(signal) = {p.mean().item():.4f})"
    )


# ----------------------------------------------------------------------------------- primitives
def _modes(ref, new, call, what):
    for mode in ("train", "train", "eval"):
        getattr(ref, mode)()
        getattr(new, mode)()
        with torch.no_grad():
            a, b = call(ref), call(new)
        a, b = (a, b) if isinstance(a, tuple) else ((a,), (b,))
        for i, (x, y) in enumerate(zip(a, b)):
            assert torch.equal(x, y), (
                f"{what} [{mode}] output {i}: max |diff| {(x - y).abs().max().item():.3e}"
            )
        _same_state(ref, new, f"{what} [{mode}]")


def _pair(ref_cls, new_cls, kw, new_kw=None, seed=0):
    torch.manual_seed(seed)
    ref = ref_cls(**kw)
    torch.manual_seed(seed)
    new = new_cls(**(new_kw or kw))
    _same_state(ref, new, f"{new_cls.__name__} init")
    return ref, new


def test_primitives_parity():
    BS = 256
    g = torch.Generator().manual_seed(1)
    x = torch.randn(BS, 8, 6, generator=g)
    mask = torch.rand(BS, 6, generator=g) < 0.3
    for kw in (
        dict(features=8, number_of_ghost_batches=4, device="cpu"),
        dict(
            features=8,
            features_out=8,
            conv=True,
            number_of_ghost_batches=4,
            device="cpu",
        ),
        dict(
            features=8,
            features_out=8,
            conv=True,
            phase_symmetric=True,
            number_of_ghost_batches=4,
            device="cpu",
        ),
        dict(
            features=8,
            stride=2,
            conv=True,
            features_out=8,
            number_of_ghost_batches=4,
            device="cpu",
        ),
    ):
        for m in (None, mask):
            if m is not None and kw.get("stride", 1) != 1:
                continue
            ref, new = _pair(ref_blocks.GhostBatchNorm1d, blocks.GhostBatchNorm1d, kw)
            _modes(ref, new, lambda mod, m=m: mod(x, m), f"GBN {kw}")
    ref, new = _pair(
        ref_blocks.GhostBatchNorm1d,
        blocks.GhostBatchNorm1d,
        dict(features=8, number_of_ghost_batches=4, device="cpu"),
    )
    for mod in (ref, new):
        mod.updateMeanStd(3 * x + 1, mask)
        mod.updateMeanStd(x, mask)
        mod.initMeanStd()
    _modes(ref, new, lambda mod: mod(x, mask), "GBN after initMeanStd")
    for kw in (
        dict(in_channels=8, out_channels=8),
        dict(in_channels=8, out_channels=8, phase_symmetric=True),
        dict(in_channels=8, out_channels=8, kernel_size=2, stride=2),
    ):
        ref, new = _pair(ref_blocks.conv1d, blocks.conv1d, kw)
        _modes(ref, new, lambda mod: mod(x), f"conv1d {kw}")
    q, v = torch.randn(BS, 8, 3, generator=g), torch.randn(BS, 8, 5, generator=g)
    qv = torch.randn(BS, 8, 3, 5, generator=g)
    amask = torch.rand(BS, 3, 5, generator=g) < 0.25
    for gate, kv in ((True, True), (False, False), (True, False)):
        kw = dict(
            dim=8,
            heads=2,
            phase_symmetric=True,
            use_attention_gate=gate,
            use_kv_proj=kv,
            device="cpu",
        )
        ref, new = _pair(ref_blocks.MinimalAttention, blocks.MinimalAttention, kw)
        for m in (amask, None):
            _modes(
                ref,
                new,
                lambda mod, m=m: mod(q, v, m, q.clone(), qv),
                f"MinimalAttention gate={gate} kv={kv}",
            )
        _modes(
            ref,
            new,
            lambda mod: mod(q, v, amask, q.clone(), None),
            f"MinimalAttention gate={gate} kv={kv} qv=None",
        )
    d6, q3 = torch.randn(BS, 8, 6, generator=g), torch.randn(BS, 8, 3, generator=g)
    for idx, sym in (
        ([[0, 1], [2, 3], [4, 5]], True),
        ([[0, 1], [2, 3], [4, 5]], False),
        (None, False),
    ):
        kw = dict(
            d=8, child_index=idx, phase_symmetric=True, symmetric_pair=sym, device="cpu"
        )
        ref, new = _pair(ref_comb.ReinforceLayer, combinatorics.ReinforceLayer, kw)
        _modes(ref, new, lambda mod: mod(d6, q3), f"ReinforceLayer {idx} sym={sym}")
    for upd in (True, False):
        kw = dict(
            d=8,
            child_index=[[0, 1], [2, 3], [4, 5]],
            phase_symmetric=True,
            symmetric_pair=True,
            device="cpu",
            prefix="p ",
            nLayers=2,
            xx0Update=upd,
        )
        ref, new = _pair(ref_comb.ResNetBlock, combinatorics.ResNetBlock, kw)
        _modes(
            ref,
            new,
            lambda mod: mod(d6, q3, d6.clone(), q3.clone()),
            f"ResNetBlock xx0Update={upd}",
        )
    print(
        "    GBN, conv1d, MinimalAttention, ReinforceLayer, ResNetBlock: torch.equal (train x2, eval), identical buffers"
    )


# ----------------------------------------------------------------------------------- trainer
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_trainer_parity(topology, tmp_path):
    import h5py
    from hcr.train import train_generic as ref_train

    from corenet import train as new_train

    small = repo / f"examples/data/{topology}_small.h5"
    if not small.exists():
        pytest.skip(f"{small} not found")
    args = [
        "--input",
        str(small),
        "--fold",
        "0",
        "--exclude-folds",
        "1",
        "--epochs",
        "2",
        "--batch-size-init",
        "256",
        "--freeze-epoch",
        "1",
        "--device",
        "cpu",
        "--seed",
        "7",
    ]
    ref_train.main(
        [
            "--config",
            str(TOPOLOGIES[topology]["ref_config"]),
            "--output-dir",
            str(tmp_path / "ref"),
            *args,
        ]
    )
    new_train.main(
        [
            "--config",
            str(TOPOLOGIES[topology]["config"]),
            "--output-dir",
            str(tmp_path / "new"),
            *args,
        ]
    )
    r = torch.load(
        tmp_path / "ref/model_fold0.pkl", map_location="cpu", weights_only=False
    )["model"]
    n = torch.load(
        tmp_path / "new/model_fold0.pkl", map_location="cpu", weights_only=False
    )["model"]
    # trained weights, predictions and losses differ by design (grouped leaf embedding, see
    # test_eval_forward_parity); tests/test_training_regression.py pins the training numerics
    assert list(r) == list(n)
    for k in r:
        assert r[k].shape == n[k].shape and r[k].dtype == n[k].dtype, (
            f"{topology}: trained {k}: shape/dtype {tuple(r[k].shape)} {r[k].dtype} vs "
            f"{tuple(n[k].shape)} {n[k].dtype}"
        )
        if not torch.is_floating_point(r[k]):
            assert torch.equal(r[k], n[k]), f"{topology}: integer buffer {k} differs"
    with (
        h5py.File(tmp_path / "ref/predictions_fold0.h5") as fr,
        h5py.File(tmp_path / "new/predictions_fold0.h5") as fn,
    ):
        assert (
            fr["CLASSIFICATIONS/EVENT/signal"].shape == fn["class_probabilities"].shape
        )
        assert np.array_equal(fr["EVENT/event"][...], fn["EVENT/event"][...])
    import json

    hr = json.loads((tmp_path / "ref/history_fold0.json").read_text())["history"]
    hn = json.loads((tmp_path / "new/history_fold0.json").read_text())["history"]
    assert len(hr) == len(hn) == 2, (len(hr), len(hn))
    for a, b in zip(hr, hn):
        for key in ("lr", "batch_size"):
            assert a[key] == b[key], (key, a[key], b[key])
    print(
        f"    {topology}: 2-epoch training on {small.name}: same state-dict keys, shapes and integer "
        f"buffers, same predicted events, same lr / batch-size schedule; train loss (differs by design) "
        f"reference {[round(h['train_loss'], 6) for h in hr]}, CoReNet {[round(h['train_loss'], 6) for h in hn]}"
    )


# ----------------------------------------------------------------------------------- original 4b network
NAME_MAP = [
    ("inputEmbed.ancillaryEmbed.", "event_embed."),
    ("inputEmbed.jetEmbed.", "leaf_embed.b.embed."),
    ("inputEmbed.jetConv.", "leaf_embed.b.conv."),
    ("inputEmbed.dijetEmbed.", "node_embed.H1.embed."),
    ("inputEmbed.dijetConv.", "node_embed.H1.conv."),
    ("inputEmbed.quadjetEmbed.", "node_embed.HH.embed."),
    ("inputEmbed.quadjetConv.", "node_embed.HH.conv."),
    ("inputEmbed.othJetEmbed.", "context.HH.embed.embed."),
    ("inputEmbed.othJetConv.", "context.HH.embed.conv."),
    ("inputEmbed.MdPhi_embed.", "context.HH.pair.embed."),
    ("inputEmbed.MdPhi_conv.", "context.HH.pair.conv."),
    ("attention_oo.", "context.HH.self_attention."),
    ("attention_do.", "context.HH.attention."),
    ("dijetResNetBlock.", "resnet.H1."),
    ("quadjetResNetBlock.", "resnet.HH."),
    ("select_q.", "selector.HH."),
    ("out.", "head_main."),
]


def test_original_4b_network():
    ck_path, data = (
        REF / "models/hh4b_d8/model_fold0.pkl",
        REF / "data/h5/hh4b_fold0_test.h5",
    )
    if not (ck_path.exists() and data.exists()):
        pytest.skip("reference 4b checkpoint or test file not found")
    from hcr.data.h5_dataset import H5HCRDataset
    from hcr.models.HCR import HCR

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ref = HCR(
        dijetFeatures=8,
        quadjetFeatures=8,
        ancillaryFeatures=("nSelJets", "HT"),
        canJetFeatures=("pt", "eta", "phi", "mass"),
        useOthJets="attention",
        device="cpu",
        nClasses=2,
        use_attention_gate=True,
        use_kv_proj=True,
    )
    ref.load_state_dict(ck["model"], strict=True)
    mapped = {}
    for k, v in ck["model"].items():
        src, dst = next((s, d) for s, d in NAME_MAP if k.startswith(s))
        mapped[dst + k[len(src) :]] = v
    new = CoReNet(parse_config(_cfg("hh4b")[0]), device="cpu")
    new.load_state_dict(mapped, strict=True)
    ds = H5HCRDataset(data, candidates="btag")
    can, oth, anc = ds.can_jet[:256], ds.not_can_jet[:256], ds.ancillary[:256]
    x = {
        "b": torch.cat([can, torch.ones_like(can[:, :1])], dim=1),
        "other": oth.clone(),
        "event": anc.clone(),
    }
    assert (
        sum(p.numel() for p in ref.parameters())
        == sum(p.numel() for p in new.parameters())
        == 1508
    )
    for mode in ("eval", "train"):
        getattr(ref, mode)()
        getattr(new, mode)()
        with torch.no_grad():
            c_ref, q_ref = ref(can.clone(), oth.clone(), anc.clone())
            out = new({k: v.clone() for k, v in x.items()})
        assert torch.equal(c_ref, out["logits"]), f"[{mode}] class logits differ"
        assert torch.equal(q_ref, out["views"]["HH"]), f"[{mode}] view logits differ"
    print(
        "    original hard-coded 4b network (trained checkpoint, 256 test events): class and view logits torch.equal"
    )
