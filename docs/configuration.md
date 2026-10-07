# Defining an event decay chain: the configuration file

A CoReNet network is described entirely by one YAML (or JSON) file. The same code builds the
HH → 4b network of `examples/hh4b/hh4b.yaml`, the HH → bbWW network of `examples/bbww/bbww.yaml`,
and any other final state that can be written as decay trees over reconstructed objects.

```yaml
objects:     # input collections and roles                      (required)
event:       # per-event scalars                                 (optional)
hypotheses:  # decay trees                                       (required)
attention:   # node-to-node attention edges                      (optional)
relations:   # pairwise features fed to the main head            (optional)
outputs:     # classes and score heads; the first is the main one (required)
combine:     # hypotheses feeding the main head                  (required)
training:    # network-wide options                              (optional, see hyperparameters.md)
trainer:     # training-recipe defaults for corenet-train        (optional, see training.md)
```

Any other top-level key is an error, as are unknown keys inside `training`, `trainer` and the
collection mappings. Run `corenet-validate --config <file>` to check a file (see the README).

---------------------------------------------------------------------------------------------------

## 1. `objects`: collections and roles

The block holds two kinds of entry: **collections**, read from the HDF5 file, and **roles**, which
pick a fixed number of members of a collection.

### 1.1 Collections

```yaml
objects:
  jets:   {features: [pt, eta, phi, mass, btag], max: 12}
  lepton: {features: [pt, eta, phi, mass, isE, isM]}
  met:    {features: [pt, phi]}
```

| key | type | default | meaning |
|---|---|---|---|
| `features` | list of names | required | datasets `INPUTS/<group>/<feature>` read from the file (docs/data_format.md) |
| `max` | integer ≥ 1 | 1 | number of slots the configuration uses; the file may have more slots (all of them are candidates for the role selectors) |

How features are used:

* `pt`, `eta`, `phi`, `mass` are the **kinematic** features. Collections with `max > 1` must have
  `pt`, `eta` and `phi`. A missing kinematic feature is taken as 0 (MET is given as `[pt, phi]`, so
  it enters as a massless vector with η = 0).
* `btag` is a **selection tag**: it is used by the `btag` selector only and is never fed to the network.
* Every other feature (`isE`, `isM` above) is an **extra** feature: it is appended to the embedding
  input of that collection's objects.

A collection with `max: 1` is a leaf of the decay trees by itself (`lepton`, `met`); a collection with
more slots is used through roles.

### 1.2 Roles

```yaml
  b:     jets btag 4       # the 4 highest-btag jets
  q:     jets rest 2       # the 2 leading jets not taken by an earlier role
  other: jets rest 8       # up to 8 remaining jets
```

Grammar: `<role>: <collection> <selector> <n>`, with `1 ≤ n ≤ max` and the `n` of all roles on a
collection adding up to at most its `max`. Roles are filled **in file order**, and a member is used by
at most one role.

| selector | members chosen | order of the chosen members |
|---|---|---|
| `btag` | the `n` present members with the largest `btag × 10⁶ + pT` (needs a `btag` feature) | by that key (for a 0/1 tag: tagged first, then by pT) |
| `pt` | the `n` present members with the largest pT | by pT |
| `rest` | up to `n` of the members not consumed by earlier roles | by pT (default), or with `rest_order: selector` (a `trainer` option / `--rest-order`) by the key of the previous `btag`/`pt` role on the same collection |

When fewer than `n` members are available the role is padded: the missing slots are flagged absent
and masked everywhere in the network (the event is not dropped; event selection is the job of the
file preparation).

---------------------------------------------------------------------------------------------------

## 2. `event`: event-level scalars

```yaml
event: [nSelJets, HT]
```

A list of names, default `[]`. `nSelJets` (number of present members) and `HT` (scalar pT sum of the
present members) are computed from the first collection with `max > 1`; every other name is read from
`EVENT/<name>`. The scalars are normalised, embedded to `d` features and **added to the embedding of
every leaf object and of every context object**. A scalar whose name contains `nSelJets` enters as
`log(nSelJets − 3)` (values ≤ 3 give −1).

---------------------------------------------------------------------------------------------------

## 3. `hypotheses`: decay trees

```yaml
hypotheses:
  HH:
    H1: [b, b]
    H2: [b, b]
    context: other
  tt:
    t_had: [b, W_had]
    t_lep: [b, W_lep]
```

A hypothesis is a named decay tree; its name is also the name of its top node. A tree node is one of

| form | meaning |
|---|---|
| `Node: [child, child, ...]` | composite whose four-vector is the sum of its children (2 or more roles, max-1 collections or nodes) |
| `Node: {Child: ..., Child: ...}` | composite whose children are themselves defined in place |
| `Node: A + B` | **recipe**: a registered reconstruction of leaves A and B (section 3.3) |
| `context: <role>` | (directly inside a hypothesis) the hypothesis attends to the objects of this role (section 4.2) |

### 3.1 Combinatorics (inferred, nothing to declare)

For each composite node the network enumerates every assignment of distinct role slots to its
children:

* children with identical structure are interchangeable: `[b, b]` gives unordered pairs, and the
  sibling nodes `H1: [b, b]`, `H2: [b, b]` give unordered pairs of pairs; interchangeable siblings share
  one set of weights (H2 is computed by the H1 modules);
* a role slot is used at most once per candidate (exclusivity); non-exclusive combinations are never
  built.

The candidates of a hypothesis' top node are its **views**. A hypothesis with more than one view gets
a **selector**: a learned score per view, a softmax over the views that are present, and the weighted
sum of the view pixels as the hypothesis' selected candidate. The view scores are learned only
through the classification loss.

| declaration | candidates | views |
|---|---|---|
| `HH: {H1: [b,b], H2: [b,b]}` with `b: jets btag 4` | 6 dijets | 3 pairings {01\|23}, {02\|13}, {03\|12} |
| `H_bb: [b, b]` with `b: jets btag 2` | 1 | 1 |
| `t_had: [b, W_had]` with 2 b and one W_had | 2 | – |
| `tt: {t_had: [b, W_had], t_lep: [b, W_lep]}` | 2 × 2 minus the 2 that use the same b = 2 | 2 |

### 3.2 Node features (inferred)

Every composite node is built from, automatically:

* its four-vector, the sum of its children's (or the recipe's result), and from it log(1+pT), η,
  log(1+m) and either φ, or Δφ to its parent when every candidate of the node sits in exactly one slot
  of one parent node; the top node of a hypothesis carries no φ at all;
* for every child slot: the child's log(1+pT), η, Δφ(child, node), log(1+m) (plus the extra features
  of a leaf), embedded and then refined by two **reinforce** layers (a residual block that updates the
  node pixel from its children's pixels);
* for a recipe node, the recipe's extra quantities.

Relations between two objects that are not parent and child (ΔR, pair mass, ...) are not implied by
the tree; declare them in `relations` (section 5) or obtain them as attention biases (section 4.1).

### 3.3 Recipes

`Node: A + B` uses a registered reconstruction instead of a plain four-vector sum. The registry is
keyed by the (sorted) **collection names** of the children, so a role `l: lepton pt 1` on a `lepton`
collection works like the `lepton` collection itself.

| children collections | produces |
|---|---|
| `lepton + met` | the leptonic W: neutrino pz from the W-mass constraint (m_W = 80.379 GeV), taking the smaller-\|pz\| solution (the real part when there is none); extra node features log(1 + m_T(lepton, MET)) and log(1 + off-shell score), the off-shell score being the imaginary part of the pz solution (0 when real) |

Any other combination is a configuration error. New recipes are added in code (`RECIPES` in
`src/corenet/models/network.py` and `src/corenet/models/config.py`).

### 3.4 Node reuse across hypotheses

Node names are global. `tt` above references `W_had` and `W_lep` declared under `HH`: they are the same
nodes (one set of weights, computed once). A reused node's Δφ-to-parent refers to the parent in the
hypothesis that **declares** it (`W_had` → `H_WW`), whatever the order of the hypotheses in the file.
Defining a name twice with different children is an error.

---------------------------------------------------------------------------------------------------

## 4. Attention

### 4.1 `attention`: node to node

```yaml
attention:
  - W_lep: W_had     # queries: W_lep candidates; keys / values: W_had candidates
  - t_had: t_lep
```

A list of single-key mappings `query: keys`; both sides are composite nodes. Each entry is applied
as soon as the pixels of both nodes exist during the bottom-up pass, and updates the query node's
pixels (residual). Automatically:

* **bias**: the (log(1+M), ΔR, Δφ) of every (query candidate, key candidate) pair, embedded 3 → d;
* **exclusivity mask**: a key candidate that shares a constituent (the same jet, lepton or MET) with
  the query candidate is masked; for a node attending to itself (`tt: tt`) only the diagonal is masked;
  absent candidates are masked as well.

If after the exclusivity mask every query candidate has **at most one** admissible key, a softmax over
it would be identically 1; such an edge is built as a pairwise **merge** instead:
`q_i ← q_i + conv([q_i, k_partner(i), pair_i])` (3d → d). Both bbWW edges are merges; edges with several
admissible keys (for example `H1: H1` in HH → 4b) are real attention with `training.attention_heads`
heads.

### 4.2 `context`: a hypothesis attending to a role

```yaml
  HH:
    H1: [b, b]
    H2: [b, b]
    context: other
```

The objects of the context are every leaf object of the hypothesis (flagged 2) plus the members of
the context role (flagged +1 when present, −1 when padded); their log(1+pT), η, log(1+m) and flag are
embedded (plus the event scalars). The objects first self-attend (pair bias: log(1+M), Δφ), then the
candidates of the hypothesis' first composite node in bottom-up order (the H dijets above) attend to
them; a candidate never attends to its own constituents. The context role must be a declared role.

---------------------------------------------------------------------------------------------------

## 5. `relations`: pairwise features for the main head

```yaml
relations:
  - [lepton, met]
  - [H_bb, met]
  - [lepton, W_had]
```

Optional list of `[A, B]` pairs of node or leaf names. For each pair the network computes
(log(1+M), ΔR, Δφ) of the two four-vectors, embeds them (3 → d) and appends the pixel to the inputs of
the main head (section 6). Both sides must have exactly one candidate (a max-1 collection, a role with
`n: 1`, or a node with a single candidate such as `H_bb` with two b); otherwise the network cannot be
built ("relation [t_had, met]: t_had has 2 candidates").

---------------------------------------------------------------------------------------------------

## 6. `outputs` and `combine`: classes and heads

```yaml
outputs:
  signal: HH
  ttbar:  tt
combine: [HH, tt]
```

`outputs` is an ordered mapping `name: hypothesis` (or `name: ~`). Each entry is at once

* a **class**: the class labels are `0 = rest` (every event of a process not named here) and
  `1, 2, ...` for the outputs **in file order**; the file's labels must lie in `0..len(outputs)`;
* a **head**: the first output is the **main head**, a softmax over `[rest, output_1, output_2, ...]`
  trained with cross entropy; every further output also gets its own binary **auxiliary head**
  (`label == its class`, binary cross entropy):

| output | head input |
|---|---|
| first (main) | the selected candidates of the `combine` hypotheses and the `relations` pixels; with more than one of these they are concatenated and reduced by a k·d → d convolution |
| further, with a hypothesis | that hypothesis' selected candidate |
| further, with `~` | the event pixel (the input of the main head) |

`combine` must list the main output's hypothesis; every name must be a declared hypothesis, and every
hypothesis must appear in `combine` or in `outputs`.

---------------------------------------------------------------------------------------------------

## 7. `training`: network-wide options

```yaml
training:
  features: 8
  attention_heads: 2
  attention_gate: true
  kv_projection: true
  phase_symmetric: true
  reflect_eta: true
  head_weights: {ttbar: 1.0}
```

See [hyperparameters.md](hyperparameters.md) for the meaning, defaults and constraints of each key.

---------------------------------------------------------------------------------------------------

## 8. The two examples

* [examples/hh4b/README.md](../examples/hh4b/README.md): HH → bbbb, one hypothesis with three
  pairings and context attention to the remaining jets (1,508 parameters).
* [examples/bbww/README.md](../examples/bbww/README.md): HH → bb WW → bb qq ℓν against tt̄ → bqq bℓν,
  two hypotheses sharing the W nodes, a recipe, two merge edges and three relations (3,840 parameters).
