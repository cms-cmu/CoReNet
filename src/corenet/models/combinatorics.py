"""
Candidate enumeration from a decay tree, and the index-gather reinforce layers.

1. ``HypothesisTable`` -- for every node of a hypothesis, the ordered list of its candidates (as
   assignments of role slots), the ``child_index`` table mapping each candidate to the candidates of
   its children, the top node's *views*, and the inferred symmetries. A role slot is used at most
   once per candidate (exclusivity), enforced at enumeration.

   Ordering rule: views of the top node are enumerated in canonical lexicographic order; the
   candidates of every node are listed in order of first appearance across those views. For
   ``HH: {H1: [b, b], H2: [b, b]}`` with four b this yields the dijets
   (0,1),(2,3),(0,2),(1,3),(0,3),(1,2) and the views {01|23},{02|13},{03|12}.

2. ``ReinforceLayer`` -- gathers the child pixels of every parent candidate, interleaves them with the
   parent pixel ``[c_1 .. c_k, parent]`` and applies a kernel = stride = k+1 GBN convolution. With
   ``symmetric_pair=True`` (k = 2, interchangeable composite children) the pair is replaced by its
   fixed symmetric / antisymmetric combinations. ``ResNetBlock`` alternates these reinforce layers with
   pixel-wise convolutions of the children.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .blocks import GhostBatchNorm1d
from .fourvectors import NonLU


# ----------------------------------------------------------------------------- tree description
@dataclass(frozen=True)
class Role:
    """A fixed number of slots taken from an input collection (``b: jets btag 4`` -> Role('b', 4))."""
    name: str
    n: int


@dataclass
class Node:
    """A composite: ``children`` are role names or node names; ``recipe`` names a reconstruction
    (e.g. ``W_lep: lepton + met``) instead of a plain four-vector sum."""
    name: str
    children: list[str]
    recipe: str | None = None


# a candidate's assignment: tuple of (role, slot) pairs, sorted
Assignment = tuple[tuple[str, int], ...]


def _signature(name: str, nodes: dict[str, Node], roles: dict[str, Role]) -> tuple:
    """Structural signature; children with equal signatures are interchangeable."""
    if name in roles:
        return ("role", name)
    node = nodes[name]
    return ("node", node.recipe or "", tuple(sorted(_signature(c, nodes, roles) for c in node.children)))


@dataclass
class NodeTable:
    name: str
    candidates: list[Assignment]                         # ordered
    children: list[str]                                  # (canonical) child names in declaration order
    child_index: torch.Tensor | None = None              # (P, k) long: candidate index in each child's table
    symmetric_children: bool = False                     # children interchangeable (same signature)

    @property
    def n(self) -> int:
        return len(self.candidates)


@dataclass
class HypothesisTable:
    top: str
    nodes: dict[str, NodeTable]                          # keyed by canonical node name
    views: list[tuple]                                   # per view: the children assignments of the top node
    view_index: torch.Tensor = field(default=None)       # (V,) long: index into nodes[top].candidates
    aliases: dict[str, str] = field(default_factory=dict)  # declared name -> canonical name (H2 -> H1)

    def __getitem__(self, name: str) -> NodeTable:
        return self.nodes[name]


# an entry is a candidate with its full structure: (assignment, children-entries)
Entry = tuple


def _merge(assignments) -> Assignment:
    return tuple(sorted(itertools.chain.from_iterable(assignments)))


def _exclusive(assignments) -> bool:
    used = list(itertools.chain.from_iterable(assignments))
    return len(used) == len(set(used))


def _enumerate(name: str, nodes: dict[str, Node], roles: dict[str, Role], cache: dict) -> list[Entry]:
    """All candidates of ``name`` as entries ``(assignment, children_entries)`` in canonical
    lexicographic order. A role slot may be used at most once within a candidate (exclusivity).
    Two candidates are distinct when their child structure differs, even if they use the same slots."""
    if name in cache:
        return cache[name]
    if name in roles:
        out = [(((name, i),), ()) for i in range(roles[name].n)]
        cache[name] = out
        return out
    node = nodes[name]
    child_lists = [_enumerate(c, nodes, roles, cache) for c in node.children]
    sigs = [_signature(c, nodes, roles) for c in node.children]
    symmetric = len(node.children) >= 2 and all(s == sigs[0] for s in sigs)
    if symmetric:  # interchangeable children: unordered selection of distinct child candidates
        pool = child_lists[0]
        combos = [tuple(pool[i] for i in idx) for idx in itertools.combinations(range(len(pool)), len(node.children))]
    else:
        combos = list(itertools.product(*child_lists))
    combos = [c for c in combos if _exclusive(e[0] for e in c)]  # a slot is used once per candidate
    out = [(_merge(e[0] for e in combo), combo) for combo in combos]
    out.sort(key=lambda entry: (entry[0], entry[1]))
    cache[name] = out
    return out


def build_hypothesis_table(top: str, nodes: dict[str, Node], roles: dict[str, Role]) -> HypothesisTable:
    """Enumerate the views of ``top`` and derive every node's ordered candidate list and child_index."""
    cache: dict = {}
    top_entries = _enumerate(top, nodes, roles, cache)

    # interchangeable sibling nodes (equal signature, e.g. H1/H2) share one table under the first name
    aliases: dict[str, str] = {}
    for node in nodes.values():
        groups: dict[tuple, str] = {}
        for c in node.children:
            if c in roles:
                continue
            aliases[c] = groups.setdefault(_signature(c, nodes, roles), c)
    canon = lambda n: aliases.get(n, n)  # noqa: E731

    # order: views in canonical order; each node's candidates by first appearance across the views
    ordered: dict[str, list[Entry]] = {}

    def visit(name: str, entry: Entry):
        cname = canon(name)
        ordered.setdefault(cname, [])
        if entry not in ordered[cname]:
            ordered[cname].append(entry)
        if name in roles:
            return
        for child, sub in zip(nodes[name].children, entry[1]):
            visit(child, sub)

    for entry in top_entries:
        visit(top, entry)

    tables: dict[str, NodeTable] = {}
    for name, entries in ordered.items():
        cands = [a for a, _ in entries]
        if name in roles:
            tables[name] = NodeTable(name, cands, [])
            continue
        node = nodes[name]
        sigs = [_signature(c, nodes, roles) for c in node.children]
        sym = len(node.children) >= 2 and all(s == sigs[0] for s in sigs)
        children = [canon(c) for c in node.children]
        idx = torch.tensor(
            [[ordered[c].index(sub) for c, sub in zip(children, entry[1])] for entry in entries], dtype=torch.long
        )
        tables[name] = NodeTable(name, cands, children, idx, sym)

    view_index = torch.tensor([ordered[top].index(entry) for entry in top_entries], dtype=torch.long)
    views = [tuple(e[0] for e in entry[1]) for entry in top_entries]
    return HypothesisTable(top, tables, views, view_index, aliases)


# ----------------------------------------------------------------------------- torch modules
class ReinforceLayer(nn.Module):
    """Update parent-candidate pixels from their children.

    parent ``xx``: (bs, d, P); children ``x``: (bs, d, C) indexed by ``child_index`` (P, k). The k child
    pixels and the parent pixel are interleaved ``[c_1, .., c_k, xx]`` per parent and convolved with
    kernel = stride = k+1. ``child_index=None`` (k = 0) is a plain pixel-wise convolution of the parent.

    ``symmetric_pair`` (k = 2 only): feed ``(c_a + c_b)/2`` and ``|c_a - c_b|/2`` instead of the two
    children, so the update is invariant under exchanging them.
    """

    def __init__(self, d, child_index, phase_symmetric=False, symmetric_pair=False, name="reinforce", device=None):
        super().__init__()
        self.d = d
        self.name = name
        self.device = device
        idx = torch.as_tensor(child_index, dtype=torch.long) if child_index is not None else torch.zeros(1, 0, dtype=torch.long)
        if idx.dim() == 1:
            idx = idx.view(-1, 1)
        self.register_buffer("child_index", idx, persistent=False)  # not a weight: keep out of the state_dict
        self.k = int(idx.shape[1])
        self.symmetric_pair = symmetric_pair
        if symmetric_pair:
            assert self.k == 2, "symmetric_pair needs exactly two children"
            self.sym = nn.Conv1d(d, d, 2, stride=2, bias=False, groups=d)
            self.sym.weight.data.fill_(0.5)
            self.sym.weight.requires_grad = False
            self.antisym = nn.Conv1d(d, d, 2, stride=2, bias=False, groups=d)
            self.antisym.weight.data.fill_(0.5)
            self.antisym.weight.data[:, :, 1] *= -1
            self.antisym.weight.requires_grad = False
        self.conv = GhostBatchNorm1d(d, phase_symmetric=phase_symmetric, stride=self.k + 1, conv=True, name=name,
                                     device=device)

    def forward(self, x, xx):
        bs, d, P = xx.shape
        if self.k == 0:
            return self.conv(xx)
        gathered = x[:, :, self.child_index.reshape(-1)].view(bs, d, P, self.k)  # (bs, d, P, k)
        if self.symmetric_pair:
            pair = gathered.reshape(bs, d, P * 2)
            s, a = self.sym(pair), self.antisym(pair).abs()                        # (bs, d, P) each
            gathered = torch.stack([s, a], dim=3)                                   # (bs, d, P, 2)
        merged = torch.cat([gathered, xx.unsqueeze(3)], dim=3).reshape(bs, d, P * (self.k + 1))
        return self.conv(merged)


class ResNetBlock(nn.Module):
    """``nLayers`` reinforce steps of parent pixels ``xx`` from child pixels ``x``, with pixel-wise
    convolutions refining ``x`` in between and residuals ``x0`` / ``xx0``. With ``xx0Update=False``
    (top node) only the updated parent pixels are returned, otherwise also their new residual."""

    def __init__(self, d, child_index, phase_symmetric=True, symmetric_pair=False, device="cuda", prefix="",
                 nLayers=2, xx0Update=True):
        super().__init__()
        self.d = d
        self.device = device
        self.xx0Update = xx0Update
        reinforce, conv = [], []
        for i in range(1, nLayers + 1):
            if i != nLayers:  # the final x update is never used
                conv.append(GhostBatchNorm1d(d, phase_symmetric=phase_symmetric, conv=True,
                                             name="%sjet convolution" % prefix, device=device))
            reinforce.append(ReinforceLayer(d, child_index, phase_symmetric=phase_symmetric,
                                            symmetric_pair=symmetric_pair, name="%sreinforce" % prefix, device=device))
        self.reinforce = nn.ModuleList(reinforce)
        self.conv = nn.ModuleList(conv)

    def setGhostBatches(self, nGhostBatches):
        for r in self.reinforce:
            r.conv.setGhostBatches(nGhostBatches)
        for c in self.conv:
            c.setGhostBatches(nGhostBatches)

    def forward(self, x, xx, x0, xx0):
        for i, conv in enumerate(self.conv):
            xx = self.reinforce[i](x, xx)
            x = conv(x)
            xx = xx + xx0
            x = x + x0
            xx = NonLU(xx)
            x = NonLU(x)
        xx = self.reinforce[-1](x, xx)
        xx = xx + xx0
        if self.xx0Update:
            xx0 = xx.clone()
            xx = NonLU(xx)
            return xx, xx0
        return NonLU(xx)


__all__ = ["Role", "Node", "NodeTable", "HypothesisTable", "build_hypothesis_table", "ReinforceLayer", "ResNetBlock"]
