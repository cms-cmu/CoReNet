"""
CoReNet: the Combinatorial Residual Network, built from a ``NetworkSpec`` (docs/EventInfo.md).

Structure of one forward pass:

    inputs  role tensors (n, len(features) + 1, slots) in the collection's feature order, plus a
            presence flag (+1 present / -1 padded) as the last channel; event scalars (n, dA)
    1. kinematics    node 4-vectors bottom-up from the candidate tables (four-vector sum over the
                     children; a massless MET leaf enters with eta = m = 0)
    2. features      log(1+pt), eta, dphi-to-parent, log(1+m) for every child slot; the top node
                     drops phi; event scalars whose name contains ``nSelJets`` enter as
                     log(nSelJets - 3)
    3. embedding     GBN embed (+ event scalars for leaves) -> conv, per role / node
    4. reinforce     bottom-up ResNetBlock per composite node (a node reused by several
                     hypotheses is computed once); ``context`` attention after the first level;
                     ``attention`` entries as soon as both nodes exist, with the pair
                     (log M, dR, dPhi) bias and the constituent-exclusivity mask
    5. select        masked softmax over the top node's views -> selected candidate pixel
    6. heads         main head on the concatenated ``combine`` + ``relations`` pixels (reduced by a
                     k*d -> d convolution when there is more than one); one head per further output

The role tensors are produced by ``corenet.data.H5Dataset``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import GhostBatchNorm1d, MinimalAttention, linear
from .combinatorics import ResNetBlock, build_hypothesis_table
from .config import KINEMATIC, ConfigError, NetworkSpec
from .fourvectors import NonLU, PxPyPzE, addFourVectors, diObjectMass

TAG_FEATURES = (
    "btag",
)  # selection tags: used by the role selectors, never fed to the network
MASKED_LOGIT = (
    -1e4
)  # selector logit of an absent view (finite, so an all-absent event stays finite)


# ------------------------------------------------------------------------------- small helpers
def calcDeltaPhi(v1, v2):
    """Minimal |dphi| between two (n, 4, P) tensors in (pt, eta, phi, m) representation."""
    dPhi12 = (v1[:, 2:3] - v2[:, 2:3]) % math.tau
    dPhi21 = (v2[:, 2:3] - v1[:, 2:3]) % math.tau
    return torch.min(dPhi12, dPhi21)


def matrixMdPhi(v1, v2, v1PxPyPzE=None, v2PxPyPzE=None):
    """(n, 2, N, M): log(1 + M(a,b)) and dphi(a,b) for all pairs of v1 (n,4,N) and v2 (n,4,M)."""
    if v1PxPyPzE is None:
        v1PxPyPzE = PxPyPzE(v1)
    if v2PxPyPzE is None:
        v2PxPyPzE = PxPyPzE(v2)
    M = torch.log(1 + diObjectMass(v1PxPyPzE.unsqueeze(-1), v2PxPyPzE.unsqueeze(-2)))
    dPhi = calcDeltaPhi(v1.unsqueeze(-1), v2.unsqueeze(-2))
    return torch.cat((M, dPhi), 1)


def matrixMdRdPhi(v1, v2, v1PxPyPzE=None, v2PxPyPzE=None):
    """(n, 3, N, M): log(1 + M(a,b)), dR(a,b), dphi(a,b) -- the pair features of ``attention`` biases
    and ``relations``."""
    MdPhi = matrixMdPhi(v1, v2, v1PxPyPzE, v2PxPyPzE)
    dEta = v1[:, 1:2].unsqueeze(-1) - v2[:, 1:2].unsqueeze(-2)
    dR = torch.sqrt(dEta**2 + MdPhi[:, 1:2] ** 2)
    return torch.cat((MdPhi[:, 0:1], dR, MdPhi[:, 1:2]), 1)


def log_kinematics(v):
    """(pt, eta, phi, m) -> (log(1+pt), eta, phi, log(1+m))."""
    return torch.cat(
        [torch.log(1 + v[:, 0:1]), v[:, 1:3], torch.log(1 + v[:, 3:4])], dim=1
    )


def finite(x, fill=-1.0):
    """Non-finite entries (log of a padded pt = -1, mass of a garbage sum) -> ``fill``; identity otherwise."""
    return torch.where(torch.isfinite(x), x, torch.full_like(x, fill))


class _Embed(nn.Module):
    """GBN embedding (in -> d) followed by a GBN convolution (d -> d)."""

    def __init__(self, n_in, d, phase_symmetric, name, device, bias=True):
        super().__init__()
        self.embed = GhostBatchNorm1d(
            n_in,
            features_out=d,
            phase_symmetric=phase_symmetric,
            conv=True,
            bias=bias,
            name=f"{name} embedder",
            device=device,
        )
        self.conv = GhostBatchNorm1d(
            d,
            phase_symmetric=phase_symmetric,
            conv=True,
            name=f"{name} convolution",
            device=device,
        )

    def forward(self, x, mask=None, add=None):
        x = self.embed(x, mask)
        if add is not None:
            x = x + add
        return self.conv(NonLU(x), mask)

    def setGhostBatches(self, n):
        self.embed.setGhostBatches(n)
        self.conv.setGhostBatches(n)


class _Context(nn.Module):
    """``context: <role>``: the objects self-attend, then the first-level candidates attend to them."""

    def __init__(self, d, heads, phase_symmetric, gate, kv, device):
        super().__init__()
        self.embed = _Embed(
            4, d, phase_symmetric, "attention jet", device
        )  # log pt, eta, log m, flag
        self.pair = _Embed(
            3, d, phase_symmetric, "M(a,b), dPhi(a,b)", device
        )  # log(1+M), dphi, pair-type flag
        kw = dict(
            heads=heads,
            phase_symmetric=phase_symmetric,
            use_attention_gate=gate,
            use_kv_proj=kv,
            device=device,
        )
        self.self_attention = MinimalAttention(d, **kw)
        self.attention = MinimalAttention(d, **kw)

    def setGhostBatches(self, n):
        self.embed.setGhostBatches(n)
        self.pair.setGhostBatches(n)
        self.self_attention.setGhostBatches(n)
        self.attention.setGhostBatches(n)


# ------------------------------------------------------------------------------- the network
# Symbols used below, in physics terms:
#   n, d        events in the batch; width of every pixel (``training.features``)
#   h, T        a hypothesis (named by its top node, e.g. HH) and its HypothesisTable: for every node of the decay
#               tree, the ordered candidates (which jets / objects are assigned to its decay products)
#   nt          the table of one node: nt.children (its decay products), nt.n candidates, nt.child_index (P, k)
#               = for each candidate the index of the candidate of each child in that child's table
#   P           the candidate axis of a tensor (n, features, P): one pixel per candidate of a node (for HH -> 4b:
#               6 dijets, 3 pairings; 1 for a node with a unique assignment). V = candidates of a top node = views
#   k           the number of children (decay products) of a node, or (in ``attention``) the keys node of an edge
#   kin, kinP   candidate four-vectors as (pt, eta, phi, m) and as (px, py, pz, E); present = all constituents are
#               present in the event (a candidate built on a padded slot is absent)
#   x, xx       child pixels and parent pixels of a reinforce block; x0, xx0 their residual (pre-activation) copies
class CoReNet(nn.Module):
    """The network of a ``NetworkSpec``. ``forward(inputs)`` takes the dict produced by the loader
    (one tensor per leaf plus ``"event"``) and returns ``{"logits": (n, nClasses), "views": {hypothesis:
    (n, V)}, "heads": {output: (n,)}, "event": (n, d, 1)}``; views are given for hypotheses with more
    than one candidate, heads for every output after the first."""

    def __init__(self, spec: NetworkSpec, device="cpu"):
        super().__init__()
        self.spec = spec
        self.device = device
        tr = spec.training
        self.d, ps = tr.features, tr.phase_symmetric
        self.nC = len(spec.classes)
        self.leaves = spec.leaves()
        self.leaf_order = [r for r in spec.roles] + [
            c.name
            for c in spec.collections.values()
            if c.max == 1 and c.name not in spec.roles
        ]

        # candidate tables, one per hypothesis
        self.tables = {
            h: build_hypothesis_table(
                h, {n: spec.nodes[n] for n in spec.nodes_in(h)}, self.leaves
            )
            for h in spec.hypotheses
        }
        # canonical composite nodes in bottom-up order, per hypothesis
        self.levels: dict[str, list[str]] = {
            h: self._bottom_up(h) for h in spec.hypotheses
        }
        self.tops = set(spec.hypotheses)
        # canonical name of every declared node (H2 -> H1)
        self.canonical = {}
        for T in self.tables.values():
            self.canonical.update({n: n for n in T.nodes})
            self.canonical.update(T.aliases)

        # ---- embeddings
        self.dA = len(spec.event)
        self.event_embed = (
            GhostBatchNorm1d(
                self.dA,
                features_out=self.d,
                phase_symmetric=ps,
                conv=True,
                bias=False,
                name="ancillary embedder",
                device=device,
            )
            if self.dA
            else None
        )
        self.leaf_embed = nn.ModuleDict()
        for r in self.leaf_order:
            if any(
                r in T.nodes[n].children
                for T in self.tables.values()
                for n in T.nodes
                if n not in self.leaves
            ):
                self.leaf_embed[r] = _Embed(
                    4 + len(self._extras(r)), self.d, ps, r, device
                )
        self.node_embed = nn.ModuleDict()
        self.resnet = nn.ModuleDict()
        for h, T in self.tables.items():
            for node in self.levels[h]:
                if node in self.node_embed:
                    continue
                nt = T.nodes[node]  # candidate table of this node
                k = len(nt.children)  # decay products per candidate
                n_in = (
                    3 if node in self.tops else 4
                )  # node features: log pt, eta, log m (+ dphi to its parent)
                self.node_embed[node] = _Embed(n_in, self.d, ps, node, device)
                children_composite = all(c not in self.leaves for c in nt.children)
                groups = [
                    [p * k + j for j in range(k)] for p in range(nt.n)
                ]  # per-slot child pixels
                self.resnet[node] = ResNetBlock(
                    self.d,
                    groups,
                    phase_symmetric=ps,
                    symmetric_pair=(
                        nt.symmetric_children and children_composite and k == 2
                    ),
                    device=device,
                    prefix=f"{node} ",
                    nLayers=2,
                    xx0Update=node not in self.tops,
                )

        # ---- attention
        self.context = nn.ModuleDict(
            {
                h: _Context(
                    self.d,
                    tr.attention_heads,
                    ps,
                    tr.attention_gate,
                    tr.kv_projection,
                    device,
                )
                for h, hs in spec.hypotheses.items()
                if hs.context
            }
        )
        # every `attention` entry is an edge query -> keys with the constituent-exclusivity mask.
        # An edge whose queries have at most ONE admissible key each is not attention (a softmax over one
        # entry is 1): it is registered as a fixed pairwise MERGE of query, partner and pair features.
        self.attention = nn.ModuleDict()
        self.merge = nn.ModuleDict()
        self.attention_pair = (
            nn.ModuleDict()
        )  # (log(1+M), dR, dphi) embedder, both kinds
        self.attention_nodes: dict[str, tuple[str, str]] = {}
        self.merge_nodes: dict[str, tuple[str, str]] = {}
        for a in spec.attention:
            q, k = (
                self.canonical[a.query],
                self.canonical[a.keys],
            )  # query node and keys node of the edge
            key = f"{a.query}>{a.keys}"
            cq, ck = self._candidates(q), self._candidates(k)
            if (
                q == k
            ):  # self-attention: the views of one node share members by construction -> mask the diagonal only
                same = torch.eye(len(cq), dtype=torch.bool)
            else:  # a query candidate never attends to a key candidate sharing a member
                same = torch.tensor(
                    [[bool(set(x) & set(y)) for y in ck] for x in cq], dtype=torch.bool
                )
            self.register_buffer(
                f"same_{key.replace('>', '_')}",
                same.view(1, len(cq), len(ck)),
                persistent=False,
            )
            self.attention_pair[key] = _Embed(3, self.d, ps, f"{key} pair", device)
            admissible = (~same).sum(1)  # admissible keys per query candidate
            if int(admissible.max()) > 1:
                self.attention_nodes[key] = (q, k)
                self.attention[key] = MinimalAttention(
                    self.d,
                    heads=tr.attention_heads,
                    phase_symmetric=ps,
                    use_attention_gate=tr.attention_gate,
                    use_kv_proj=tr.kv_projection,
                    device=device,
                )
            else:
                self.merge_nodes[key] = (q, k)
                partner = torch.full((len(cq),), -1, dtype=torch.long)
                for i in range(len(cq)):
                    js = torch.nonzero(~same[i]).flatten()
                    if js.numel():
                        partner[i] = int(js[0])
                self.register_buffer(
                    f"partner_{key.replace('>', '_')}", partner, persistent=False
                )
                self.merge[key] = GhostBatchNorm1d(
                    self.d,
                    features_out=self.d,
                    stride=3,
                    conv=True,
                    name=f"{key} merge",
                    device=device,
                )
        self.edge_nodes = {**self.attention_nodes, **self.merge_nodes}

        # ---- relations
        self.relation_embed = nn.ModuleDict()
        self.relation_nodes: dict[str, tuple[str, str]] = {}
        for rel in spec.relations:
            for name in (rel.a, rel.b):
                P = self._n_candidates(name)
                if P != 1:
                    raise ConfigError(
                        f"relation [{rel.a}, {rel.b}]: {name} has {P} candidates; relations need unique candidates"
                    )
            key = f"{rel.a}|{rel.b}"
            self.relation_nodes[key] = (rel.a, rel.b)
            self.relation_embed[key] = _Embed(3, self.d, ps, f"{key} relation", device)

        # ---- selectors and heads
        self.selector = nn.ModuleDict(
            {
                h: GhostBatchNorm1d(
                    self.d,
                    features_out=1,
                    conv=True,
                    bias=False,
                    name=f"{h} selector",
                    device=device,
                )
                for h, T in self.tables.items()
                if T.nodes[h].n > 1
            }
        )
        k = len(spec.combine) + len(spec.relations)
        # k = 1: the head reads the selected pixel directly; otherwise the k pixels are first
        # reduced by a k*d -> d convolution
        self.head_reduce = (
            GhostBatchNorm1d(
                self.d,
                features_out=self.d,
                stride=k,
                conv=True,
                name="combine",
                device=device,
            )
            if k > 1
            else None
        )
        self.head_main = GhostBatchNorm1d(
            self.d, features_out=self.nC, conv=True, name="out", device=device
        )
        self.aux_heads = nn.ModuleDict()
        for o in spec.outputs[1:]:
            if o.hypothesis:
                self.aux_heads[o.name] = GhostBatchNorm1d(
                    self.d,
                    features_out=1,
                    conv=True,
                    name=f"{o.name} head",
                    device=device,
                )
            else:
                self.aux_heads[o.name] = linear(self.d, 1, name=f"{o.name} head")

    # ------------------------------------------------------------------ structure helpers
    def _bottom_up(self, h):
        T = self.tables[h]
        depth = {}

        def dep(n):
            if n in self.leaves:
                return 0
            if n not in depth:
                depth[n] = 1 + max(dep(c) for c in T.nodes[n].children)
            return depth[n]

        nodes = [n for n in T.nodes if n not in self.leaves]
        return sorted(nodes, key=dep)

    def _extras(self, role):
        feats = self.spec.collection_of(role).features
        return [f for f in feats if f not in KINEMATIC and f not in TAG_FEATURES]

    def _kinematic_channels(self, role):
        feats = self.spec.collection_of(role).features
        return [feats.index(k) if k in feats else None for k in KINEMATIC]

    def _table_of(self, name):
        """(hypothesis, NodeTable) of a canonical node, from the first hypothesis that contains it."""
        for h, T in self.tables.items():
            if name in T.nodes:
                return h, T.nodes[name]
        raise KeyError(name)

    def _candidates(self, name):
        """Assignments (tuples of (role, slot)) of a node or leaf."""
        if name in self.leaves:
            return [((name, i),) for i in range(self.leaves[name].n)]
        return self._table_of(self.canonical[name])[1].candidates

    def _n_candidates(self, name):
        return len(self._candidates(name))

    def _same(self, key):
        return getattr(self, f"same_{key.replace('>', '_')}")

    def _partner(self, key):
        return getattr(self, f"partner_{key.replace('>', '_')}")

    # ------------------------------------------------------------------ preparation
    def _prepare(self, inputs):
        """Kinematics, features and masks for every hypothesis. Returns a dict of intermediate tensors."""
        n = next(iter(inputs.values())).shape[0]
        # per leaf (role or single object): four-vectors (n, 4, slots), its non-kinematic features (isE / isM; btag
        # is never used), and which slots hold a real object
        leaf_kin, leaf_extra, leaf_present, etaSign = {}, {}, {}, None
        for r in self.leaf_order:
            if r not in inputs:
                continue
            x = inputs[r].clone()
            chans = self._kinematic_channels(r)
            S = x.shape[2]
            kin = torch.zeros(n, 4, S, dtype=x.dtype, device=x.device)
            for i, c in enumerate(chans):
                if c is not None:
                    kin[:, i] = x[:, c]
            feats = self.spec.collection_of(r).features
            extras = [feats.index(f) for f in self._extras(r)]
            leaf_extra[r] = x[:, extras, :] if extras else None
            leaf_present[r] = x[:, -1, :] != -1  # presence flag is the last channel
            if self.spec.training.reflect_eta:
                if etaSign is None:  # leading member of the first role
                    etaSign = 1 - 2 * (kin[:, 1, 0:1] < 0).float()
                kin[:, 1, :] = etaSign * kin[:, 1, :]
            leaf_kin[r] = kin

        # event scalars; a jet multiplicity (name containing ``nSelJets``) enters as log(nSelJets - 3)
        a = None
        if self.dA:
            a = inputs["event"].view(n, self.dA, 1)
            for idx, name in enumerate(self.spec.event):
                if "nSelJets" in name:
                    a = torch.cat(
                        [
                            a[:, :idx],
                            torch.log(a[:, idx : idx + 1] - 3),
                            a[:, idx + 1 :],
                        ],
                        dim=1,
                    )
            a = finite(a)  # nSelJets <= 3 would give -inf / NaN

        out = {
            "n": n,
            "leaf_kin": leaf_kin,
            "leaf_extra": leaf_extra,
            "leaf_present": leaf_present,
            "event": a,
            "hyp": {},
        }
        # node kinematics are shared across hypotheses (a reused node is one node)
        kin, kinP, present = {}, {}, {}
        for h, T in self.tables.items():
            out["hyp"][h] = self._prepare_hypothesis(
                h, T, leaf_kin, leaf_extra, leaf_present, kin, kinP, present
            )
        out["kin"], out["kinP"], out["present"] = kin, kinP, present
        # a node reused by several hypotheses is embedded once; its dphi reference is the parent in the
        # hypothesis that DECLARES it, independent of the order of the hypotheses in the file
        embed_in = {}
        for h in self.tables:
            for node, f in out["hyp"][h]["embed_in"].items():
                if self.spec.owner.get(node, h) == h:
                    embed_in[node] = f
        for h in self.tables:  # safety net: a node whose owner table lacks it
            for node, f in out["hyp"][h]["embed_in"].items():
                embed_in.setdefault(node, f)
        out["embed_in"] = embed_in

        # attention biases and masks
        def vec(name):
            if name in self.leaves:
                return leaf_kin[name], None, leaf_present[name]
            c = self.canonical[name]
            return kin[c], kinP[c], present[c]

        out["attention"] = {}
        for key, (q, k) in self.edge_nodes.items():
            vq, vqP, pq = vec(q)
            vk, vkP, pk = vec(k)
            qv = finite(matrixMdRdPhi(vq, vk, vqP, vkP))  # (n,3,Pq,Pk)
            mask = ~pq.unsqueeze(2) | ~pk.unsqueeze(1) | self._same(key)  # (n,Pq,Pk)
            out["attention"][key] = {"qv": qv, "mask": mask}
        # relations: one (log M, dR, dphi) pixel per pair
        out["relations"] = {}
        for key, (ra, rb) in self.relation_nodes.items():
            va, vaP, _ = vec(ra)
            vb, vbP, _ = vec(rb)
            out["relations"][key] = finite(matrixMdRdPhi(va, vb, vaP, vbP)).view(
                n, 3, 1
            )
        return out

    def _prepare_hypothesis(
        self, h, T, leaf_kin, leaf_extra, leaf_present, kin, kinP, present
    ):
        # kin / kinP / present are shared across hypotheses: a reused node is computed once
        for node in self.levels[h]:
            if node in kin:
                continue
            nt = T.nodes[node]
            vecs, vecsP, pres = [], [], None
            for j, c in enumerate(nt.children):
                idx = nt.child_index[:, j]
                if c in self.leaves:
                    vecs.append(leaf_kin[c][:, :, idx])
                    vecsP.append(None)
                    p = leaf_present[c][:, idx]
                else:
                    vecs.append(kin[c][:, :, idx])
                    vecsP.append(kinP[c][:, :, idx])
                    p = present[c][:, idx]
                pres = p if pres is None else (pres & p)
            v, vP = addFourVectors(
                vecs[0], vecs[1], v1PxPyPzE=vecsP[0], v2PxPyPzE=vecsP[1]
            )
            for extra_v, extra_P in zip(vecs[2:], vecsP[2:]):
                v, vP = addFourVectors(v, extra_v, v1PxPyPzE=vP, v2PxPyPzE=extra_P)
            kin[node], kinP[node], present[node] = v, vP, pres

        # log features
        lf = {c: log_kinematics(k) for c, k in leaf_kin.items()}
        nf = {node: log_kinematics(kin[node]) for node in self.levels[h]}
        # unique-parent map: node -> (parent, slot) if every candidate sits in exactly one parent slot
        parent_of = {}
        for node in self.levels[h]:
            nt = T.nodes[node]
            for j, c in enumerate(nt.children):
                if c in self.leaves:
                    continue
                parent_of.setdefault(c, []).append((node, j))
        # per-slot child features of every composite node, one (n,F_j,P) tensor per child (children may
        # have different widths: a lepton carries isE/isM, MET nothing), and each node's own embedding input
        slot_feats, slot_present, embed_in = {}, {}, {}
        for node in self.levels[h]:
            nt = T.nodes[node]
            cols, pres = [], []
            for j, c in enumerate(nt.children):
                idx = nt.child_index[:, j]
                f = (lf[c] if c in self.leaves else nf[c])[:, :, idx]  # (n,4,P)
                dphi = calcDeltaPhi(nf[node], f)
                f = torch.cat([f[:, 0:2], dphi, f[:, 3:4]], dim=1)
                if c in self.leaves and leaf_extra[c] is not None:
                    f = torch.cat([f, leaf_extra[c][:, :, idx]], dim=1)
                cols.append(finite(f) if c in self.leaves else f)
                pres.append(
                    leaf_present[c][:, idx] if c in self.leaves else present[c][:, idx]
                )
            slot_feats[node], slot_present[node] = cols, pres
        for node in self.levels[h]:
            f = nf[node]
            if node in self.tops:
                f = torch.cat((f[:, :2], f[:, 3:]), 1)  # no phi
            else:
                # dphi to the parent when every candidate sits in exactly one slot of one parent node
                # (all slots of that parent together cover the candidates once); otherwise keep phi
                parents = parent_of.get(node, [])
                P = f.shape[2]
                pnodes = {pn for pn, _ in parents}
                dphi = None
                if len(pnodes) == 1:
                    pnode = next(iter(pnodes))
                    cols = [T.nodes[pnode].child_index[:, j] for pn, j in parents]
                    allidx = torch.cat(cols)
                    if allidx.numel() == P and allidx.unique().numel() == P:
                        dphi = torch.empty_like(f[:, 2:3])
                        for idx in cols:
                            dphi[:, :, idx] = calcDeltaPhi(nf[pnode], f[:, :, idx])
                if dphi is not None:
                    f = torch.cat([f[:, 0:2], dphi, f[:, 3:4]], dim=1)
            embed_in[node] = finite(f)
        first = self.levels[h][0]
        prep = {
            "slot_feats": slot_feats,
            "slot_present": slot_present,
            "embed_in": embed_in,
            "first": first,
        }
        hs = self.spec.hypotheses[h]
        if hs.context:
            prep["context"] = self._prepare_context(
                h, T, hs.context, leaf_kin, leaf_present, kin, kinP
            )
        return prep

    def _prepare_context(self, h, T, crole, leaf_kin, leaf_present, kin, kinP):
        """Objects = this hypothesis' leaves (jet roles, lepton, MET; flag 2) + the context role (its own flag); pair
        matrices object-object (flag 0) and first-level-candidate-object (flag 1), with masks.
        ``crole`` is the context role (e.g. ``other``: the jets not assigned to any decay product); ``o`` (n, 5, S) the
        S objects the candidates may look at; ``d`` (n, 4, P) the P candidates of the first-level node."""
        first = self.levels[h][0]
        # every leaf of the hypothesis (jet roles and single-object collections such as lepton / MET), flag 2
        leaf_roles = [
            c
            for c in self.leaf_order
            if c != crole and any(c in T.nodes[n].children for n in self.levels[h])
        ]
        parts, flags = [], []
        for r in leaf_roles:
            parts.append(leaf_kin[r])
            flags.append(2 * torch.ones_like(leaf_kin[r][:, 0:1, :]))
        parts.append(leaf_kin[crole])
        flags.append(
            torch.where(leaf_present[crole], 1.0, -1.0)
            .unsqueeze(1)
            .to(leaf_kin[crole].dtype)
        )
        o = torch.cat([torch.cat(parts, 2), torch.cat(flags, 2)], 1)  # (n,5,S)
        mask = o[:, 4, :] == -1
        S = o.shape[2]
        oP = PxPyPzE(o)
        oo = matrixMdPhi(o, o, v1PxPyPzE=oP, v2PxPyPzE=oP)
        oo = torch.cat([oo, torch.zeros_like(oo[:, 0:1])], 1)
        # float identity cast to bool (same mask; ONNX Runtime has no bool EyeLike kernel)
        eye = (torch.eye(S, device=o.device) > 0).unsqueeze(0)
        mask_oo = mask.unsqueeze(1) | mask.unsqueeze(2) | eye
        d = kin[first]
        do = matrixMdPhi(d, o, v1PxPyPzE=kinP[first], v2PxPyPzE=oP)
        do = torch.cat([do, torch.ones_like(do[:, 0:1])], 1)
        P = d.shape[2]
        mask_do = mask.unsqueeze(1).repeat(1, P, 1)
        # a candidate must not attend to its own constituents
        same = torch.zeros(1, P, S, dtype=torch.bool, device=o.device)
        offset = {}
        pos = 0
        for r in leaf_roles:
            offset[r] = pos
            pos += leaf_kin[r].shape[2]
        for p, cand in enumerate(T.nodes[first].candidates):
            for role, slot in cand:
                if role in offset:
                    same[0, p, offset[role] + slot] = True
        mask_do = mask_do | same
        o_pt = torch.log(1 + o[:, 0:1])
        o_m = torch.log(1 + o[:, 3:4])
        of = torch.cat([o_pt, o[:, 1:3], o_m, o[:, 4:]], dim=1)
        of = torch.where(
            torch.isinf(of) | torch.isnan(of), torch.full_like(of, -1.0), of
        )
        of = torch.cat((of[:, :2], of[:, 3:]), 1)  # log pt, eta, log m, flag
        return {
            "feats": of,
            "mask": mask,
            "oo": oo,
            "mask_oo": mask_oo,
            "do": do,
            "mask_do": mask_do,
            "S": S,
            "P": P,
        }

    # ------------------------------------------------------------------ forward
    def forward(self, inputs):
        prep = self._prepare(inputs)
        n = prep["n"]
        a = self.event_embed(prep["event"]) if self.event_embed is not None else None

        selected, view_logits = {}, {}
        pixels, pixels0 = {}, {}  # canonical node -> (n,d,P), shared by hypotheses
        applied = set()  # attention entries already applied
        for h, T in self.tables.items():
            hp = prep["hyp"][h]
            for node in self.levels[h]:
                if node in pixels:
                    continue  # reused node: computed once
                nt = T.nodes[node]
                k = len(nt.children)
                # child slot pixels: leaves are embedded per slot (padded slots zeroed); composites are gathered
                cols, cols0 = [], []
                for j, c in enumerate(nt.children):
                    idx = nt.child_index[:, j]
                    if c in self.leaves:
                        x = self.leaf_embed[c](hp["slot_feats"][node][j], add=a)
                        x0 = x.clone()
                        x = NonLU(x)
                        keep = hp["slot_present"][node][j].unsqueeze(1).to(x.dtype)
                        x, x0 = x * keep, x0 * keep
                    else:
                        x, x0 = pixels[c][:, :, idx], pixels0[c][:, :, idx]
                    cols.append(x)
                    cols0.append(x0)
                x = torch.stack(cols, dim=3).reshape(n, self.d, nt.n * k)
                x0 = torch.stack(cols0, dim=3).reshape(n, self.d, nt.n * k)
                xx = self.node_embed[node](prep["embed_in"][node])
                xx0 = xx.clone()
                xx = NonLU(xx)
                if node in self.tops:
                    xx = self.resnet[node](x, xx, x0, xx0)
                    pixels[node], pixels0[node] = (
                        xx,
                        xx,
                    )  # a top has no residual; keep a usable q0
                else:
                    xx, xx0 = self.resnet[node](x, xx, x0, xx0)
                    pixels[node], pixels0[node] = xx, xx0
                if node == hp["first"] and h in self.context:
                    pixels[node], pixels0[node] = self._apply_context(
                        h, hp["context"], a, pixels[node], pixels0[node]
                    )
                # node-to-node edges as soon as both sides exist
                for key, (q, kk) in self.edge_nodes.items():
                    if key in applied or q not in pixels or kk not in pixels:
                        continue
                    att = prep["attention"][key]
                    qv = NonLU(
                        self.attention_pair[key](
                            att["qv"].flatten(2), att["mask"].flatten(1)
                        )
                    )
                    qv = qv.view(n, self.d, att["qv"].shape[2], att["qv"].shape[3])
                    if key in self.attention:
                        pixels[q], pixels0[q], _ = self.attention[key](
                            pixels[q], pixels[kk], att["mask"], pixels0[q], qv
                        )
                    else:
                        pixels[q], pixels0[q] = self._apply_merge(
                            key, pixels[q], pixels0[q], pixels[kk], qv, att["mask"]
                        )
                    applied.add(key)
            top = pixels[h]
            if h in self.selector:
                logits = self.selector[h](top)  # (n,1,V)
                logits = logits.masked_fill(
                    ~prep["present"][h].unsqueeze(1), MASKED_LOGIT
                )
                score = F.softmax(logits, dim=-1)
                selected[h] = torch.matmul(top, score.transpose(1, 2))  # (n,d,1)
                view_logits[h] = logits.view(n, -1)
            else:
                selected[h] = top

        pix = [selected[h] for h in self.spec.combine]
        pix += [
            NonLU(self.relation_embed[key](prep["relations"][key]))
            for key in self.relation_nodes
        ]
        e = torch.cat(pix, dim=2)  # (n,d,k)
        if self.head_reduce is not None:
            e = NonLU(self.head_reduce(e))  # (n,d,1)
        logits = self.head_main(e).view(n, self.nC)
        heads = {}
        for o in self.spec.outputs[1:]:
            if o.hypothesis:
                heads[o.name] = self.aux_heads[o.name](selected[o.hypothesis]).view(n)
            else:  # `~` output: scored from the event pixel
                heads[o.name] = self.aux_heads[o.name](e.view(n, self.d)).view(n)
        return {"logits": logits, "views": view_logits, "heads": heads, "event": e}

    def _apply_merge(self, key, q, q0, kpix, qv, mask):
        """Fixed pairwise merge for an edge with <= 1 admissible key per query: every query candidate i is
        updated from its partner j = partner[i] and the pair features (i, j) through a 3d -> d convolution,
        with the same residual convention as MinimalAttention (q0 += delta, q = NonLU(q0)). Queries without an
        admissible (present) partner are left unchanged.
        ``q`` / ``q0`` (n, d, Pq): pixels of the query node's candidates and their residual copies; ``kpix`` (n, d, Pk):
        pixels of the keys node; ``qv`` (n, d, Pq, Pk): embedded pair features (log M, dR, dphi); ``mask`` (n, Pq, Pk)."""
        n = q.shape[0]
        partner = self._partner(key)  # (Pq,) long, -1 = none
        has = partner >= 0
        j = partner.clamp(min=0)
        kj = kpix[:, :, j]  # (n,d,Pq) partner pixels
        pair = qv[
            :, :, torch.arange(q.shape[2], device=q.device), j
        ]  # (n,d,Pq) pair embedding (i, j)
        x = torch.stack([q, kj, pair], dim=3).reshape(n, self.d, 3 * q.shape[2])
        delta = self.merge[key](x)  # (n,d,Pq)
        valid = has.view(1, 1, -1) & ~mask[
            :, torch.arange(q.shape[2], device=q.device), j
        ].unsqueeze(1)
        delta = NonLU(delta) * valid.to(delta.dtype)
        q0 = q0 + delta
        return NonLU(q0), q0

    def _apply_context(self, h, ctx, a, d, d0):
        # a: embedded event scalars (n, d, 1); d / d0: pixels of the first-level candidates and their residual copies
        C = self.context[h]
        S, P = ctx["S"], ctx["P"]
        o = C.embed(ctx["feats"], ctx["mask"], add=a)
        MdPhi = torch.cat((ctx["oo"].flatten(2), ctx["do"].flatten(2)), dim=2)
        mask_MdPhi = torch.cat(
            (ctx["mask_oo"].flatten(1), ctx["mask_do"].flatten(1)), dim=1
        )
        MdPhi = C.pair(MdPhi, mask_MdPhi)
        oo = MdPhi[:, :, : S * S].view(-1, self.d, S, S)
        do = MdPhi[:, :, S * S :].view(-1, self.d, P, S)
        o0 = o.clone()
        o = NonLU(o)
        oo = NonLU(oo)
        do = NonLU(do)
        o, o0, _ = C.self_attention(o, o, ctx["mask_oo"], o0, oo)
        d, d0, _ = C.attention(d, o, ctx["mask_do"], d0, do)
        return d, d0

    # ------------------------------------------------------------------ statistics / ghost batches
    def _input_embedders(self):
        """Every GBN that sees raw features and needs frozen mean/std statistics."""
        out = [self.event_embed] if self.event_embed is not None else []
        out += [m.embed for m in self.leaf_embed.values()] + [
            m.embed for m in self.node_embed.values()
        ]
        for C in self.context.values():
            out += [C.embed.embed, C.pair.embed]
        out += [m.embed for m in self.attention_pair.values()] + [
            m.embed for m in self.relation_embed.values()
        ]
        return out

    @torch.no_grad()
    def updateMeanStd(self, inputs):
        prep = self._prepare(inputs)
        if self.event_embed is not None:
            self.event_embed.updateMeanStd(prep["event"])
        done = set()
        for h, T in self.tables.items():
            hp = prep["hyp"][h]
            for node in self.levels[h]:
                if node in done:
                    continue
                done.add(node)
                nt = T.nodes[node]
                for j, c in enumerate(nt.children):
                    if c in self.leaves:
                        self.leaf_embed[c].embed.updateMeanStd(
                            hp["slot_feats"][node][j], ~hp["slot_present"][node][j]
                        )
                self.node_embed[node].embed.updateMeanStd(prep["embed_in"][node])
            if "context" in hp:
                ctx = hp["context"]
                self.context[h].embed.embed.updateMeanStd(ctx["feats"], ctx["mask"])
                MdPhi = torch.cat((ctx["oo"].flatten(2), ctx["do"].flatten(2)), dim=2)
                mask = torch.cat(
                    (ctx["mask_oo"].flatten(1), ctx["mask_do"].flatten(1)), dim=1
                )
                self.context[h].pair.embed.updateMeanStd(MdPhi, mask)
        for key, att in prep["attention"].items():
            self.attention_pair[key].embed.updateMeanStd(
                att["qv"].flatten(2), att["mask"].flatten(1)
            )
        for key, feats in prep["relations"].items():
            self.relation_embed[key].embed.updateMeanStd(feats)

    @torch.no_grad()
    def initMeanStd(self):
        """Freeze the accumulated statistics into every input embedder."""
        for m in self._input_embedders():
            m.initMeanStd()

    def setGhostBatches(self, nGhostBatches):
        """Number of ghost batches of every GBN (0 = use the running statistics also in training mode)."""
        if self.event_embed is not None:
            self.event_embed.setGhostBatches(nGhostBatches)
        for m in (
            list(self.leaf_embed.values())
            + list(self.node_embed.values())
            + list(self.context.values())
        ):
            m.setGhostBatches(nGhostBatches)
        for m in list(self.attention_pair.values()) + list(
            self.relation_embed.values()
        ):
            m.setGhostBatches(nGhostBatches)
        for m in (
            list(self.resnet.values())
            + list(self.attention.values())
            + list(self.merge.values())
        ):
            m.setGhostBatches(nGhostBatches)
        for m in (
            list(self.selector.values())
            + list(self.aux_heads.values())
            + [self.head_main, self.head_reduce]
        ):
            if isinstance(m, GhostBatchNorm1d):
                m.setGhostBatches(nGhostBatches)


__all__ = ["CoReNet", "calcDeltaPhi", "log_kinematics", "matrixMdPhi", "matrixMdRdPhi"]
