# Event Description Format

The first step is to describe the final state of your event: which objects the detector sees, which
of them are the candidates for each decay product, and how the signal and background processes decay
into them. CoReNet reads this from a `.yaml` file and builds the network from it. The file has the
following sections; `objects`, `hypotheses`, `outputs` and `combine` are required.

```yaml
objects:     # the reconstructed objects and the candidate jets     (required)
event:       # per-event quantities                                  (optional)
hypotheses:  # the decay chains                                      (required)
attention:   # which candidates are compared with which              (optional)
relations:   # pairs of objects the classifier looks at              (optional)
outputs:     # the classes                                           (required)
combine:     # the hypotheses the classifier is built from           (required)
training:    # network settings                                      (optional, see Options.md)
trainer:     # training settings                                     (optional, see Options.md)
```

Any other key is an error. `corenet-validate --config <file>` checks a file. The complete examples
are [`examples/hh4b/hh4b.yaml`](../examples/hh4b/hh4b.yaml) ([HH4b.md](./HH4b.md)) and
[`examples/bbww/bbww.yaml`](../examples/bbww/bbww.yaml) ([BBWW.md](./BBWW.md)).

## `objects`

The first **required** section: the collections of objects in the input file, and which of their
members are the candidate jets of the decay chain.

```yaml
objects:
  jets:   {features: [pt, eta, phi, mass], max: 12}
  b:      jets btag 4
  other:  jets rest 8
  lepton: {features: [pt, eta, phi, mass, isE, isM]}
  met:    {features: [pt, phi]}
```

* `jets`: the jet collection, with up to 12 jets per event. The listed features are the inputs of the
  network: exactly these columns of the input file ([Dataset.md](./Dataset.md)) are fed to it, `pt`,
  `eta`, `phi` and `mass` as the kinematics and any other feature as an additional input.
* `b: jets btag 4`: the four jets most likely to be b jets, taken from the jet collection. The `btag`
  rule reads the `btag` column of the file for this choice only; it need not be a feature. (Listing
  `btag` under `features` would make it one more input of the network.)
* `other: jets rest 8`: up to eight of the remaining jets.
* `lepton`: a single object per event (no `max`), with two extra inputs `isE` and `isM`.
* `met`: the missing transverse momentum, a single object given by `pt` and `phi`.

A collection has `features` (required) and `max` (the number of jets per event, default 1). A
single object enters the decay chains by its own name; a collection with several jets is used
through candidate lines of the form `<name>: <collection> <column> <n>`, where `<column>` is any
per-jet column of the collection's group in the input file (`btag`, `pt`, `mass`, ...):

| line | the `n` jets taken |
|---|---|
| `<column> <n>` | the `n` jets with the highest value of the column (`jets btag 4`: the four jets most likely to be b jets; `jets pt 4`: the four leading jets); ties go to the higher `pt` |
| `<column> <n> lowest` | the `n` jets with the lowest value of the column (`jets mass 2 lowest`), ties again to the higher `pt` |
| `rest <n>` | up to `n` of the jets not taken by an earlier line, leading `pt` first ([Options.md](./Options.md): `rest_order`) |

`rest` is the only reserved word; every other word in that position names a column. The column (and
`pt`) is read from the input file for the selection only and need not be a feature; the network sees
the `features`. Nothing else may follow `<n>` but `lowest`.

Lines are filled in order and a jet is used by at most one line. An event with fewer jets than needed
is kept; the missing ones are simply absent.

## `event`

```yaml
event: [nSelJets, HT]
```

Optional per-event quantities. `nSelJets` (the number of jets) and `HT` (their scalar momentum sum)
are computed from the jet collection; any other name is read from the input file.

```yaml
min_jets: 4
```

Optional, default 4: the minimum number of jets per event in your selection. The jet multiplicity enters
the network as `log(nSelJets - (min_jets - 1))`, so that the minimum multiplicity maps to zero; an event
with fewer jets than `min_jets` would give a non-finite input.

## `hypotheses`

The second **required** section: the decay chain of each process, as a tree over the candidate
jets, the single objects and the intermediate particles.

```yaml
hypotheses:
  HH:
    H1: [b, b]             # a Higgs boson decays to two b jets
    H2: [b, b]
    context: other         # the Higgs candidates also see the other jets of the event
```

With a lepton and missing momentum in the final state, as in the HH → bb̄WW example, the chain has
decaying products defined in place, and a second hypothesis can reuse them by name:

```yaml
hypotheses:
  HH:
    H_bb: [b, b]
    H_WW:
      W_had: [q, q]        # a hadronic W decays to two light jets
      W_lep: [lepton, met] # the leptonic W, from the lepton and the missing momentum
  tt:
    t_had: [b, W_had]      # a top decays to a b jet and a W
    t_lep: [b, W_lep]
```

Each hypothesis is named after the particle at the top of its chain (`HH`, `tt`). A particle is
written as:

| form | meaning |
|---|---|
| `Particle: [product, product, ...]` | decays to the listed products (candidate jets, single objects or other particles) |
| `Particle: {Product: ..., Product: ...}` | the products are themselves decaying particles, defined in place |
| `W_lep: [lepton, met]` | the leptonic W, formed from the lepton and the missing momentum |

`context: <name>` inside a hypothesis lets its first particles (the Higgs candidates above) also see
the jets of that name, that is, the whole event.

Nothing has to be declared about symmetries or combinatorics: products of the same kind are
interchangeable (the two b of a Higgs, the two Higgs bosons of HH) and a jet is never used twice in one
candidate. With four b jets there are three ways to pair them into two Higgs bosons; the probability
of each pairing is written to the prediction files (`views/<hypothesis>`). No truth pairing is needed
in the input file.

Names are shared between hypotheses: `tt` in the second snippet uses the `W_had` and `W_lep` declared
under `HH`, and they are the same particles in both chains.

## `attention` and `relations`

```yaml
attention:
  - W_lep: W_had           # the leptonic W candidate is compared with the hadronic W candidate
  - t_had: t_lep           # each top candidate with the other one
relations:
  - [lepton, met]          # the classifier also looks at these pairs
  - [H_bb, met]
```

Both are optional. `attention` lists pairs `A: B` of particles whose candidates are compared with
each other (candidates that would share a jet never are). `relations` lists pairs of objects or
particles given to the classifier as pairs; each side must be a single object or a particle with only
one candidate.

## `outputs` and `combine`

```yaml
outputs:
  signal: HH               # class 1: the signal score
  ttbar:  tt               # class 2: also gets a tt̄ score of its own
combine: [HH, tt]
```

`outputs` is the ordered list of classes, each tied to a hypothesis. The labels of the input file
are `0` for "rest" (any process not listed), then `1, 2, ...` for the outputs in this order. The
first output is the main one: the classifier gives the probability of every class, built from the
hypotheses in `combine`. Every further output also gets its own score from its hypothesis alone
(`heads/<output>` in the prediction files). Every hypothesis must appear in `combine` or in `outputs`.

## `training` and `trainer`

```yaml
training:
  features: 8              # width of the network
trainer:
  epochs: 20
  lr: 0.01
```

The network settings and the training settings; see [Options.md](./Options.md).
