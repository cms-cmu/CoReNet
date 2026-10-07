"""
Parse and validate a CoReNet configuration (YAML or JSON; grammar in docs/configuration.md) into a
``NetworkSpec``.

    spec = load_config("examples/hh4b/hh4b.yaml")     # or parse_config(dict)

The file grammar::

    objects:     jets: {features: [pt, eta, phi, mass, btag], max: 12}   # collection
                 b: jets btag 4                                          # role: <collection> <selector> <n>
    event:       [nSelJets, HT]
    hypotheses:  HH: {H1: [b, b], H2: [b, b], context: other}
                 tt: {t_had: [b, W_had], t_lep: [b, W_lep]}               # W_had / W_lep reused by name
                 ... W_lep: lepton + met                                  # recipe
    attention:   [{W_lep: W_had}, {t_had: t_lep}]                        # query: keys (exclusivity mask inferred)
    relations:   [[lepton, met], [H_bb, met]]   # named pairs -> (log M, dR, dPhi) pixels into the main head
    outputs:     {signal: HH, ttbar: tt}        # first = main score; names are the classes (labels 1, 2, ...)
    combine:     [HH, tt]                       # must contain the main hypothesis
    training:    {features: 8, attention_heads: 2, ...}                  # network-wide options
    trainer:     {epochs: 20, lr: 0.01, ...}    # optional training-recipe defaults, read by corenet-train only

Only ``objects``, ``hypotheses``, ``outputs`` and ``combine`` are required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .combinatorics import Node, Role

KINEMATIC = ("pt", "eta", "phi", "mass")
SELECTORS = ("btag", "pt", "rest")
RECIPES = ("lepton+met",)          # registered reconstructions: sorted *collection* names of the children joined by '+'
SECTIONS = ("objects", "event", "hypotheses", "attention", "relations", "outputs", "combine", "training", "trainer")


class ConfigError(ValueError):
    pass


@dataclass
class Collection:
    name: str
    features: list[str]
    max: int = 1


@dataclass
class RoleSpec:
    name: str
    collection: str
    selector: str          # btag | pt | rest
    n: int


@dataclass
class HypothesisSpec:
    name: str              # also the name of its top node
    context: str | None = None


@dataclass
class AttentionSpec:
    query: str
    keys: str


@dataclass
class RelationSpec:
    a: str                 # node or leaf name
    b: str


@dataclass
class OutputSpec:
    name: str
    hypothesis: str | None     # None (~): scored from the event pixel


@dataclass
class TrainingSpec:
    """Network-wide options (the ``training:`` section)."""
    features: int = 8                    # embedding width d of every pixel
    attention_heads: int = 2
    attention_gate: bool = True
    kv_projection: bool = True
    phase_symmetric: bool = True
    reflect_eta: bool = True             # flip all eta so the leading member of the first role has eta >= 0
    head_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class NetworkSpec:
    collections: dict[str, Collection]
    roles: dict[str, RoleSpec]           # in file order (matters for the `rest` selector)
    event: list[str]
    nodes: dict[str, Node]               # every composite node, by name
    hypotheses: dict[str, HypothesisSpec]
    attention: list[AttentionSpec]
    outputs: list[OutputSpec]
    combine: list[str]
    training: TrainingSpec
    relations: list[RelationSpec] = field(default_factory=list)
    owner: dict[str, str] = field(default_factory=dict)   # node -> hypothesis in which it is declared

    # ---- derived
    @property
    def classes(self) -> list[str]:
        """Label convention: 0 = rest, then the outputs in order."""
        return ["rest"] + [o.name for o in self.outputs]

    @property
    def main(self) -> OutputSpec:
        return self.outputs[0]

    def leaves(self) -> dict[str, Role]:
        """Role objects for the combinatorics: declared roles plus max-1 collections used as leaves."""
        out = {r.name: Role(r.name, r.n) for r in self.roles.values()}
        for c in self.collections.values():
            if c.max == 1 and c.name not in out:
                out[c.name] = Role(c.name, 1)
        return out

    def collection_of(self, leaf: str) -> Collection:
        return self.collections[self.roles[leaf].collection] if leaf in self.roles else self.collections[leaf]

    def nodes_in(self, hypothesis: str) -> set[str]:
        """Composite nodes reachable from a hypothesis' top node."""
        seen, stack = set(), [hypothesis]
        while stack:
            n = stack.pop()
            if n in self.nodes and n not in seen:
                seen.add(n)
                stack.extend(self.nodes[n].children)
        return seen


# --------------------------------------------------------------------------------------- parsing
COLLECTION_KEYS = ("features", "max")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _str_list(value, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ConfigError(f"{where}: expected a list of names, got {value!r}")
    dup = sorted({x for x in value if value.count(x) > 1})
    if dup:
        raise ConfigError(f"{where}: duplicate entries {dup}")
    return list(value)


def _parse_collection(name: str, body: dict) -> Collection:
    if not isinstance(body, dict) or "features" not in body:
        raise ConfigError(f"objects.{name}: a collection needs {{features: [...], max: n}}")
    unknown = [k for k in body if k not in COLLECTION_KEYS]
    if unknown:
        raise ConfigError(f"objects.{name}: unknown keys {unknown} (a collection has {', '.join(COLLECTION_KEYS)})")
    feats = _str_list(body["features"], f"objects.{name}.features")
    if not feats:
        raise ConfigError(f"objects.{name}.features: at least one feature is needed")
    mx = body.get("max", 1)
    if not _is_int(mx) or mx < 1:
        raise ConfigError(f"objects.{name}: max must be an integer >= 1, got {mx!r}")
    return Collection(name, feats, mx)


def _check_training(tr: dict) -> None:
    """Types and ranges of the ``training:`` section."""
    for key in ("features", "attention_heads"):
        if key in tr and (not _is_int(tr[key]) or tr[key] < 1):
            raise ConfigError(f"training.{key} must be an integer >= 1, got {tr[key]!r}")
    for key in ("attention_gate", "kv_projection", "phase_symmetric", "reflect_eta"):
        if key in tr and not isinstance(tr[key], bool):
            raise ConfigError(f"training.{key} must be true or false, got {tr[key]!r}")
    hw = tr.get("head_weights", {})
    if hw is not None and (not isinstance(hw, dict) or not all(_is_number(v) and v >= 0 for v in hw.values())):
        raise ConfigError(f"training.head_weights must map output names to numbers >= 0, got {hw!r}")


def _parse_role(name: str, text: str, collections: dict[str, Collection]) -> RoleSpec:
    m = re.fullmatch(r"\s*(\w+)\s+(\w+)\s+(\d+)\s*", text)
    if not m:
        raise ConfigError(f"objects.{name}: expected '<collection> <selector> <n>', got {text!r}")
    coll, sel, n = m.group(1), m.group(2), int(m.group(3))
    if coll not in collections:
        raise ConfigError(f"objects.{name}: unknown collection {coll!r}")
    if sel not in SELECTORS:
        raise ConfigError(f"objects.{name}: selector must be one of {SELECTORS}, got {sel!r}")
    if sel == "btag" and "btag" not in collections[coll].features:
        raise ConfigError(f"objects.{name}: selector 'btag' needs a 'btag' feature on {coll}")
    if n < 1 or n > collections[coll].max:
        raise ConfigError(f"objects.{name}: n={n} must be within 1..max={collections[coll].max} of {coll}")
    return RoleSpec(name, coll, sel, n)


def _parse_tree(name: str, body, nodes: dict[str, Node], leaves: set[str], defined_here: set[str]):
    """Register composite node ``name`` (and its sub-nodes) from ``body``; returns the child names."""
    if isinstance(body, str):
        parts = [p.strip() for p in body.split("+")]
        if len(parts) < 2 or not all(parts):
            raise ConfigError(f"{name}: a recipe is written 'A + B', got {body!r}")
        return _register(name, parts, "+".join(sorted(parts)), nodes, leaves, defined_here)
    if isinstance(body, list):
        if len(body) < 2:
            raise ConfigError(f"{name}: a composite needs at least two children")
        for c in body:
            if not isinstance(c, str):
                raise ConfigError(f"{name}: children must be names; nest a mapping to define sub-nodes")
        return _register(name, list(body), None, nodes, leaves, defined_here)
    if isinstance(body, dict):
        children = []
        for cname, cbody in body.items():
            if cname == "context":
                continue
            _parse_tree(cname, cbody, nodes, leaves, defined_here)
            children.append(cname)
        if len(children) < 2:
            raise ConfigError(f"{name}: a composite needs at least two children")
        return _register(name, children, None, nodes, leaves, defined_here)
    raise ConfigError(f"{name}: cannot interpret {body!r}")


def _register(name, children, recipe, nodes, leaves, defined_here):
    node = Node(name, list(children), recipe)
    if name in leaves:
        raise ConfigError(f"{name}: a node cannot have the same name as a role or collection")
    if name in nodes:
        if nodes[name].children != node.children or nodes[name].recipe != node.recipe:
            raise ConfigError(f"{name}: defined twice with different children")
        return children
    nodes[name] = node
    defined_here.add(name)
    return children


def parse_config(cfg: dict) -> NetworkSpec:
    if not isinstance(cfg, dict):
        raise ConfigError("configuration must be a mapping")
    for key in ("objects", "hypotheses", "outputs", "combine"):
        if key not in cfg:
            raise ConfigError(f"missing required section '{key}'")
    unknown = [k for k in cfg if k not in SECTIONS]
    if unknown:
        raise ConfigError(f"unknown top-level sections {unknown} (known: {', '.join(SECTIONS)})")
    for key in ("objects", "hypotheses", "outputs"):
        if not isinstance(cfg[key], dict) or not cfg[key]:
            raise ConfigError(f"{key} must be a non-empty mapping")

    # objects ------------------------------------------------------------------------------
    collections: dict[str, Collection] = {}
    roles: dict[str, RoleSpec] = {}
    for name, body in cfg["objects"].items():
        if isinstance(body, dict):
            collections[name] = _parse_collection(name, body)
    for name, body in cfg["objects"].items():
        if isinstance(body, str):
            roles[name] = _parse_role(name, body, collections)
        elif not isinstance(body, dict):
            raise ConfigError(f"objects.{name}: expected a collection mapping or a role string")
    for c in collections.values():
        if c.max > 1 and any(k not in c.features for k in ("pt", "eta", "phi")):
            raise ConfigError(f"objects.{c.name}: collections with max > 1 need pt, eta, phi features")
    # role slot budget per collection
    used: dict[str, int] = {}
    for r in roles.values():
        used[r.collection] = used.get(r.collection, 0) + r.n
        if used[r.collection] > collections[r.collection].max:
            raise ConfigError(f"roles on {r.collection} use {used[r.collection]} slots > max {collections[r.collection].max}")

    event = _str_list(cfg.get("event", []) or [], "event")

    # hypotheses ---------------------------------------------------------------------------
    leaves = set(roles) | {c.name for c in collections.values() if c.max == 1}
    nodes: dict[str, Node] = {}
    hypotheses: dict[str, HypothesisSpec] = {}
    owner: dict[str, str] = {}
    for hname, body in cfg["hypotheses"].items():
        if not isinstance(body, dict) or len([k for k in body if k != "context"]) < 2:
            raise ConfigError(f"hypotheses.{hname}: a hypothesis is a mapping of at least two children")
        context = body.get("context")
        if context is not None and (not isinstance(context, str) or context not in roles):
            raise ConfigError(f"hypotheses.{hname}.context: {context!r} is not a declared role")
        defined_here: set[str] = set()
        _parse_tree(hname, body, nodes, leaves, defined_here)
        for n in defined_here:                       # first declaration wins; a reuse by name is not a declaration
            owner.setdefault(n, hname)
        hypotheses[hname] = HypothesisSpec(hname, context)
    # every referenced name must be a leaf or a node; a recipe must be registered for its children's collections
    for node in nodes.values():
        if node.recipe:
            # the registry is keyed by the children's collections, so `l: leptons pt 1` works like `lepton`
            key = "+".join(sorted(roles[c].collection if c in roles else c for c in node.children))
            if key not in RECIPES:
                raise ConfigError(f"{node.name}: no recipe for {' + '.join(node.children)} "
                                  f"(collections {key}; known: {', '.join(RECIPES)})")
            if any(c not in leaves for c in node.children):
                raise ConfigError(f"{node.name}: recipe children must be leaves")
            node.recipe = key
        for c in node.children:
            if c not in leaves and c not in nodes:
                raise ConfigError(f"{node.name}: child {c!r} is neither a role, a max-1 collection nor a node")

    # attention ----------------------------------------------------------------------------
    attention: list[AttentionSpec] = []
    for item in cfg.get("attention", []) or []:
        if not isinstance(item, dict) or len(item) != 1:
            raise ConfigError("attention entries are single-key mappings 'query: keys'")
        (q, k), = item.items()
        for n in (q, k):
            if n not in nodes:
                raise ConfigError(f"attention {q}: {k}: {n!r} is not a node")
        attention.append(AttentionSpec(q, k))

    # relations ----------------------------------------------------------------------------
    relations: list[RelationSpec] = []
    for item in cfg.get("relations", []) or []:
        if not (isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item)):
            raise ConfigError("relations entries are pairs '[A, B]' of node or leaf names")
        a, b = item
        for n in (a, b):
            if n not in nodes and n not in leaves:
                raise ConfigError(f"relations [{a}, {b}]: {n!r} is neither a node nor a leaf")
        if a == b:
            raise ConfigError(f"relations [{a}, {b}]: a pair needs two different names")
        relations.append(RelationSpec(a, b))
    # (the single-candidate restriction needs the enumeration; CoReNet checks it when it is built)

    # outputs / combine --------------------------------------------------------------------
    outs = cfg["outputs"]
    if not isinstance(outs, dict) or not outs:
        raise ConfigError("outputs must be a non-empty mapping name -> hypothesis (or ~)")
    outputs = []
    for name, hyp in outs.items():
        if hyp is not None and (not isinstance(hyp, str) or hyp not in hypotheses):
            raise ConfigError(f"outputs.{name}: {hyp!r} is not a declared hypothesis")
        outputs.append(OutputSpec(name, hyp))
    if outputs[0].hypothesis is None:
        raise ConfigError("the first output (main score) needs a hypothesis")
    combine = _str_list(cfg["combine"] or [], "combine")
    if outputs[0].hypothesis not in combine:
        raise ConfigError(f"combine must list the main hypothesis {outputs[0].hypothesis!r} explicitly")
    for h in combine:
        if h not in hypotheses:
            raise ConfigError(f"combine: {h!r} is not a declared hypothesis")
    used_h = set(combine) | {o.hypothesis for o in outputs if o.hypothesis}
    for h in hypotheses:
        if h not in used_h:
            raise ConfigError(f"hypothesis {h!r} is used by no output and not in combine")

    # training -----------------------------------------------------------------------------
    if not isinstance(cfg.get("training", {}) or {}, dict):
        raise ConfigError("training must be a mapping")
    tr = dict(cfg.get("training", {}) or {})
    unknown = set(tr) - set(TrainingSpec.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"training: unknown keys {sorted(unknown)} (known: {', '.join(TrainingSpec.__dataclass_fields__)})")
    _check_training(tr)
    if tr.get("head_weights") is None:
        tr.pop("head_weights", None)
    training = TrainingSpec(**tr)
    if training.features % training.attention_heads:
        raise ConfigError("training.features must be divisible by training.attention_heads")
    if training.phase_symmetric and training.features % 2:
        raise ConfigError("training.features must be even when training.phase_symmetric is true")
    if training.kv_projection and training.features < 2 * training.attention_heads:
        raise ConfigError("training.features must be >= 2 * attention_heads when training.kv_projection is true")
    for h in training.head_weights:
        if h not in [o.name for o in outputs[1:]]:
            raise ConfigError(f"training.head_weights: {h!r} is not a secondary output")

    return NetworkSpec(collections, roles, event, nodes, hypotheses, attention, outputs, combine, training, relations, owner)


def read_config_dict(path: str | Path) -> dict:
    """The raw configuration mapping of a ``.yaml`` / ``.yml`` (needs PyYAML) or ``.json`` file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def load_config(path: str | Path) -> NetworkSpec:
    return parse_config(read_config_dict(path))


__all__ = ["ConfigError", "Collection", "RoleSpec", "HypothesisSpec", "AttentionSpec", "RelationSpec", "OutputSpec",
           "TrainingSpec", "NetworkSpec", "parse_config", "read_config_dict", "load_config", "RECIPES", "SECTIONS"]
