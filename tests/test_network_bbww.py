"""Structure and forward/backward tests of CoReNet on the HH -> bbWW example configuration
(examples/bbww/bbww.yaml).

Random inputs in the loader contract: role tensors (n, len(features) + 1, slots) with the presence flag
last, event scalars (n, 2). Checks shapes, finiteness (also with padded leaves), the inferred
constituent-exclusivity mask of `t_had: t_lep`, the relations pixels, the lepton + MET node, statistics
initialisation and a backward pass.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from corenet.models.config import ConfigError, load_config, parse_config
from corenet.models.fourvectors import addFourVectors
from corenet.models.network import CoReNet

repo = Path(__file__).resolve().parent.parent
CFG = repo / "examples" / "bbww" / "bbww.yaml"


def make_inputs(n=64, seed=0, pad_q=False):
    g = torch.Generator().manual_seed(seed)

    def jets(k):
        pt = 30 + 120 * torch.rand(n, k, generator=g)
        eta = 4.8 * torch.rand(n, k, generator=g) - 2.4
        phi = math.tau * torch.rand(n, k, generator=g) - math.pi
        m = 5 + 20 * torch.rand(n, k, generator=g)
        flag = torch.ones(n, k)
        return torch.stack(
            [pt, eta, phi, m, flag], dim=1
        )  # (n, 4+1, k): the features + flag

    b, q = jets(2), jets(2)
    if pad_q:  # second q jet missing
        q[:, :, 1] = -1.0
    lpt = 25 + 80 * torch.rand(n, 1, generator=g)
    lep = torch.stack(
        [
            lpt,
            4.8 * torch.rand(n, 1, generator=g) - 2.4,
            math.tau * torch.rand(n, 1, generator=g) - math.pi,
            torch.full((n, 1), 0.105),
            torch.zeros(n, 1),
            torch.ones(n, 1),
            torch.ones(n, 1),
        ],
        dim=1,
    )  # (n, 6+1, 1)
    met = torch.stack(
        [
            20 + 100 * torch.rand(n, 1, generator=g),
            math.tau * torch.rand(n, 1, generator=g) - math.pi,
            torch.ones(n, 1),
        ],
        dim=1,
    )  # (n, 2+1, 1)
    event = torch.stack(
        [
            torch.randint(4, 9, (n,), generator=g).float(),
            200 + 400 * torch.rand(n, generator=g),
        ],
        dim=1,
    )
    return {"b": b, "q": q, "lepton": lep, "met": met, "event": event}


def build():
    spec = load_config(CFG)
    model = CoReNet(spec, device="cpu")
    return spec, model


def test_structure():
    spec, model = build()
    # candidates: 2 exclusive tt views, one candidate for every HH node
    assert (
        model.tables["tt"].nodes["tt"].n == 2
        and model.tables["tt"].nodes["t_had"].n == 2
    )
    assert all(
        model.tables["HH"].nodes[x].n == 1
        for x in ("H_bb", "W_had", "W_lep", "H_WW", "HH")
    )
    # W_lep is an ordinary two-body composite of two single-object leaves: the same 4 embedding channels as W_had
    assert spec.nodes["W_lep"].children == ["lepton", "met"]
    assert (
        model.node_embed["W_lep"].embed.features
        == 4
        == model.node_embed["W_had"].embed.features
    )
    # leaf embedders per role, with their own widths (lepton carries isE/isM, MET nothing extra)
    assert (
        model.leaf_embed["lepton"].embed.features == 6
        and model.leaf_embed["met"].embed.features == 4
    )
    assert (
        model.leaf_embed["b"].embed.features == 4
        and model.leaf_embed["q"].embed.features == 4
    )
    # edges: both bbWW entries have exactly one admissible key per query -> registered as merges, not attention
    assert (
        list(model.merge) == ["W_lep>W_had", "t_had>t_lep"]
        and len(model.attention) == 0
    )
    assert list(model.attention_pair) == ["W_lep>W_had", "t_had>t_lep"]
    # t_had[0] (b0) pairs with t_lep[0] (b1), t_had[1] (b1) with t_lep[1] (b0)
    assert model._partner("t_had>t_lep").tolist() == [0, 1] and model._partner(
        "W_lep>W_had"
    ).tolist() == [0]
    assert list(model.relation_embed) == ["lepton|met", "H_bb|met", "lepton|W_had"]
    assert (
        model.head_reduce is not None and model.head_reduce.stride == 2 + 3
    )  # combine [HH, tt] + 3 relations
    assert set(model.aux_heads) == {"ttbar"} and model.nC == 3
    # exclusivity mask of t_had -> t_lep: the same-b pairs are masked, the others open
    same = model._same("t_had>t_lep")[0]
    tH = model.tables["tt"].nodes["t_had"].candidates
    tL = model.tables["tt"].nodes["t_lep"].candidates
    for i, ch in enumerate(tH):
        for j, cl in enumerate(tL):
            assert bool(same[i, j]) == bool(set(ch) & set(cl)), (ch, cl)
    assert same.sum() == 2 and (~same).sum() == 2
    n_par = sum(p.numel() for p in model.parameters())
    assert n_par == 3832, n_par


def test_forward_shapes_and_finiteness():
    spec, model = build()
    x = make_inputs()
    model.updateMeanStd(x)
    model.initMeanStd()
    assert all(m.initialized for m in model._input_embedders()), (
        "statistics not initialised"
    )
    model.eval()
    out = model(x)
    n = x["b"].shape[0]
    assert out["logits"].shape == (n, 3) and torch.isfinite(out["logits"]).all()
    assert out["views"]["tt"].shape == (n, 2) and "HH" not in out["views"]
    assert (
        out["heads"]["ttbar"].shape == (n,)
        and torch.isfinite(out["heads"]["ttbar"]).all()
    )
    assert out["event"].shape == (n, spec.training.features, 1)
    # attention biases and masks exist for both entries
    prep = model._prepare(x)
    assert prep["attention"]["W_lep>W_had"]["qv"].shape == (n, 3, 1, 1)
    assert prep["attention"]["t_had>t_lep"]["qv"].shape == (n, 3, 2, 2)
    assert torch.equal(
        prep["attention"]["t_had>t_lep"]["mask"][0], model._same("t_had>t_lep")[0]
    )
    assert torch.isfinite(prep["attention"]["t_had>t_lep"]["qv"]).all()
    assert all(
        v.shape == (n, 3, 1) and torch.isfinite(v).all()
        for v in prep["relations"].values()
    )
    # the leptonic W is the plain four-vector sum of the lepton and the massless MET (eta = m = 0, so pz comes
    # from the lepton only): no W-mass constraint, no extra features
    lep, met = prep["leaf_kin"]["lepton"], prep["leaf_kin"]["met"]
    assert torch.equal(met[:, 1], torch.zeros_like(met[:, 1])) and torch.equal(
        met[:, 3], torch.zeros_like(met[:, 3])
    )
    assert torch.equal(prep["kin"]["W_lep"], addFourVectors(lep, met)[0])
    assert prep["hyp"]["HH"]["embed_in"]["W_lep"].shape[1] == 4
    # training mode forward + backward; ghost batches of 8 events as the training loop would set them
    # (with the default one-event ghost batches, single-pixel tensors have no variance)
    model.setGhostBatches(n // 8)
    model.train()
    out = model(x)
    loss = (
        torch.nn.functional.cross_entropy(out["logits"], torch.randint(0, 3, (n,)))
        + out["heads"]["ttbar"].pow(2).mean()
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_padded_leaves_stay_finite():
    spec, model = build()
    x = make_inputs(pad_q=True)
    model.updateMeanStd(x)
    model.initMeanStd()
    model.eval()
    prep = model._prepare(x)
    assert not prep["leaf_present"]["q"][:, 1].any()
    # W_had (q0 + q1) is absent for every event -> so are H_WW, HH, t_had, tt
    for node in ("W_had", "H_WW", "HH", "t_had", "tt"):
        assert not prep["present"][node].any(), node
    out = model(x)
    assert (
        torch.isfinite(out["logits"]).all() and torch.isfinite(out["views"]["tt"]).all()
    )
    assert torch.isfinite(out["heads"]["ttbar"]).all()


def test_relation_needs_unique_candidates():
    cfg = {
        "objects": {
            "jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12},
            "b": "jets btag 2",
            "q": "jets rest 2",
            "lepton": {"features": ["pt", "eta", "phi", "mass"]},
            "met": {"features": ["pt", "phi"]},
        },
        "hypotheses": {
            "HH": {
                "H_bb": ["b", "b"],
                "H_WW": {"W_had": ["q", "q"], "W_lep": ["lepton", "met"]},
            },
            "tt": {"t_had": ["b", "W_had"], "t_lep": ["b", "W_lep"]},
        },
        "relations": [["t_had", "met"]],
        "outputs": {"signal": "HH", "ttbar": "tt"},
        "combine": ["HH", "tt"],
    }
    try:
        CoReNet(parse_config(cfg), device="cpu")
    except ConfigError as e:
        assert "t_had has 2 candidates" in str(e), e
    else:
        raise AssertionError("expected a ConfigError for a multi-candidate relation")


def _bbww_dict():
    import yaml

    return yaml.safe_load(open(CFG))


def test_shared_node_phi_independent_of_hypothesis_order():
    """W_had / W_lep (declared under HH) take dphi to H_WW whether HH or tt is listed first."""
    cfg = _bbww_dict()
    swapped = dict(cfg)
    swapped["hypotheses"] = {
        "tt": cfg["hypotheses"]["tt"],
        "HH": cfg["hypotheses"]["HH"],
    }
    x = make_inputs(n=32, seed=3)
    m1, m2 = CoReNet(parse_config(cfg), "cpu"), CoReNet(parse_config(swapped), "cpu")
    assert (
        m1.spec.owner["W_had"] == "HH"
        and m2.spec.owner["W_had"] == "HH"
        and m2.spec.owner["t_had"] == "tt"
    )
    p1, p2 = m1._prepare(x), m2._prepare(x)
    for node in ("W_had", "W_lep", "H_bb", "t_had", "tt", "HH"):
        assert torch.equal(p1["embed_in"][node], p2["embed_in"][node]), node
    # and the chosen copy is the HH one (dphi to H_WW), not the tt one (phi kept)
    assert torch.equal(p1["embed_in"]["W_had"], p1["hyp"]["HH"]["embed_in"]["W_had"])
    assert not torch.equal(
        p1["hyp"]["HH"]["embed_in"]["W_had"], p1["hyp"]["tt"]["embed_in"]["W_had"]
    )


def test_context_includes_lepton_and_met():
    """With `context: other` the object list holds every leaf of the hypothesis, lepton and MET included."""
    cfg = _bbww_dict()
    cfg["objects"]["other"] = "jets rest 8"  # b 2 + q 2 + other 8 = 12 = jets.max
    cfg["hypotheses"]["HH"]["context"] = "other"
    m = CoReNet(parse_config(cfg), "cpu")
    x = make_inputs(n=32, seed=4)
    x["other"] = torch.cat([x["q"][:, :, :1].repeat(1, 1, 8)], dim=2).clone()
    x["other"][:, :, 3:] = -1.0  # three real "other" jets, five padded
    ctx = m._prepare(x)["hyp"]["HH"]["context"]
    assert ctx["S"] == 2 + 2 + 1 + 1 + 8, ctx["S"]  # b, q, lepton, met, other
    assert ctx["P"] == 1  # first-level node (H_bb) has a single candidate
    flags = ctx["feats"][
        0, -1
    ]  # last channel = flag: 2 for own leaves, +-1 for context jets
    assert (
        (flags[:6] == 2).all() and (flags[6:9] == 1).all() and (flags[9:] == -1).all()
    )
    m.updateMeanStd(x)
    m.initMeanStd()
    m.eval()
    out = m(x)
    assert torch.isfinite(out["logits"]).all()


def test_self_attention_masks_diagonal_only():
    """`tt: tt` masks a view against itself only, not against the other view sharing the same jets
    (with 2 views that leaves one admissible key -> a merge with partner [1, 0])."""
    cfg = _bbww_dict()
    cfg["attention"] = [{"W_lep": "W_had"}, {"tt": "tt"}]
    m = CoReNet(parse_config(cfg), "cpu")
    assert torch.equal(m._same("tt>tt")[0], torch.eye(2, dtype=torch.bool))
    assert "tt>tt" in m.merge and m._partner("tt>tt").tolist() == [1, 0]
    x = make_inputs(n=32, seed=5)
    m.updateMeanStd(x)
    m.initMeanStd()
    m.eval()
    assert torch.isfinite(m(x)["logits"]).all()


def test_edge_with_many_keys_is_attention():
    """An edge whose queries have several admissible keys stays real attention: HH -> 4b dijets attending to
    the other dijets (6 candidates, diagonal masked -> 5 keys each)."""
    # btag listed as a feature here: it is one more input channel (the 5th of the 6 stacked below)
    cfg = {
        "objects": {
            "jets": {"features": ["pt", "eta", "phi", "mass", "btag"], "max": 12},
            "b": "jets btag 4",
            "other": "jets rest 8",
        },
        "event": ["nSelJets", "HT"],
        "hypotheses": {"HH": {"H1": ["b", "b"], "H2": ["b", "b"], "context": "other"}},
        "attention": [{"H1": "H1"}],
        "outputs": {"signal": "HH"},
        "combine": ["HH"],
    }
    m = CoReNet(parse_config(cfg), "cpu")
    assert "H1>H1" in m.attention and not m.merge
    same = m._same("H1>H1")[0]
    assert same.shape == (6, 6) and torch.equal(same, torch.eye(6, dtype=torch.bool))
    n = 16
    g = torch.Generator().manual_seed(7)
    jets = torch.stack(
        [
            30 + 100 * torch.rand(n, 12, generator=g),
            4 * torch.rand(n, 12, generator=g) - 2,
            math.tau * torch.rand(n, 12, generator=g) - math.pi,
            5 + 10 * torch.rand(n, 12, generator=g),
            torch.rand(n, 12, generator=g),
            torch.ones(n, 12),
        ],
        dim=1,
    )
    x = {
        "b": jets[:, :, :4].clone(),
        "other": jets[:, :, 4:].clone(),
        "event": torch.stack(
            [torch.full((n,), 6.0), 300 + 200 * torch.rand(n, generator=g)], dim=1
        ),
    }
    x["other"][:, :, 5:] = -1.0
    m.updateMeanStd(x)
    m.initMeanStd()
    m.eval()
    assert torch.isfinite(m(x)["logits"]).all()


def test_merge_partner_semantics():
    """A merge updates each query from its unique partner; a query whose partner is absent is left unchanged."""
    spec, model = build()
    x = make_inputs(n=32, seed=8)
    model.updateMeanStd(x)
    model.initMeanStd()
    model.eval()
    prep = model._prepare(x)
    att = prep["attention"]["t_had>t_lep"]
    assert torch.equal(att["mask"][0], torch.tensor([[False, True], [True, False]]))
    q = torch.randn(32, model.d, 2)
    q0 = q.clone()
    k = torch.randn(32, model.d, 2)
    qv = torch.randn(32, model.d, 2, 2)
    q_new, q0_new = model._apply_merge("t_had>t_lep", q, q0, k, qv, att["mask"])
    assert q_new.shape == (32, model.d, 2) and not torch.equal(q0_new, q0)
    all_masked = torch.ones_like(att["mask"])
    q_same, q0_same = model._apply_merge("t_had>t_lep", q, q0, k, qv, all_masked)
    assert torch.equal(q0_same, q0)


def test_three_jet_event_scalars_stay_finite():
    """nSelJets = 3 gives log(0) = -inf before the guard; the network must stay finite."""
    spec, model = build()
    x = make_inputs(n=32, seed=6)
    x["event"][:, 0] = 3.0
    model.updateMeanStd(x)
    model.initMeanStd()
    model.eval()
    prep = model._prepare(x)
    assert torch.isfinite(prep["event"]).all()
    assert torch.isfinite(model(x)["logits"]).all()


def test_predict_and_predict_proba():
    spec, model = build()
    x = make_inputs(n=16, seed=12)
    model.updateMeanStd(x)
    model.initMeanStd()
    probs = model.predict_proba(x)
    preds = model.predict(x)
    assert probs.shape == (16, len(spec.classes))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert preds.shape == (16,)
    assert (preds == probs.argmax(dim=-1)).all()

    # Also test with numpy inputs
    x_numpy = {k: v.numpy() for k, v in x.items()}
    probs_np = model.predict_proba(x_numpy)
    assert torch.allclose(probs_np, probs, atol=1e-6)
