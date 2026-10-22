# HH → bb̄WW Example Guide

HH → bb̄ WW → bb̄ qq̄′ ℓν against tt̄ → b qq̄′ b ℓν. One Higgs boson decays to two b jets, the other to
a hadronic and a leptonic W. The main background, tt̄, has the same final state (two b jets, two light
jets, a lepton and missing momentum), so it is described as a second hypothesis on the same objects.
This guide builds on the [HH → bb̄bb̄ guide](./HH4b.md) and shows what is new: single objects (a
lepton, the missing momentum), the leptonic W, two hypotheses sharing particles, and a second class
with its own score.

## Data

`examples/data/bbww_small.h5` (20,000 events: 1,198 HH, 18,802 tt̄; 10.2 MB) is a random subsample of
`examples/data/bbww_all.h5` (292,285 events: 17,511 HH, 274,774 tt̄; 149 MB), which is distributed
separately on Zenodo ([doi:10.5281/zenodo.23567532](https://doi.org/10.5281/zenodo.23567532)); download it into `examples/data/`. The layout
([Dataset.md](./Dataset.md)):

| path | shape | meaning |
|---|---|---|
| `INPUTS/Jets/pt`, `eta`, `phi`, `mass`, `btag`, `MASK` | (N, 12) | all jets of the event, 12 slots |
| `INPUTS/Lepton/pt`, `eta`, `phi`, `mass`, `isE`, `isM` | (N,) | the lepton; `isE`, `isM` flag electron or muon |
| `INPUTS/Met/pt`, `phi` | (N,) | the missing transverse momentum |
| `EVENT/label` | (N,) | 1 = HH, 2 = tt̄ (the file has no class 0) |
| `EVENT/weight`, `EVENT/fold`, `EVENT/event` | (N,) | weight, fold (5 folds), event identifier |

## Event `.yaml` File

The decay-chain part of `examples/bbww/bbww.yaml`:

```yaml
objects:
  jets:   {features: [pt, eta, phi, mass], max: 12}        # INPUTS/Jets
  b:      jets btag 2                                       # the 2 highest-btag jets
  q:      jets rest 2                                       # the 2 leading remaining jets
  lepton: {features: [pt, eta, phi, mass, isE, isM]}        # INPUTS/Lepton, a single object (max 1)
  met:    {features: [pt, phi]}                             # INPUTS/Met, a single object (max 1)
event: [nSelJets, HT]

hypotheses:
  HH:
    H_bb: [b, b]
    H_WW:
      W_had: [q, q]
      W_lep: [lepton, met]   # the leptonic W, from the lepton and the missing momentum
  tt:
    t_had: [b, W_had]        # W_had and W_lep are the W candidates already defined under HH, reused by name
    t_lep: [b, W_lep]

attention:
  - W_lep: W_had
  - t_had: t_lep             # a top never pairs with the top that uses the same b jet

relations:                   # pairs the classifier looks at directly
  - [lepton, met]
  - [H_bb, met]
  - [lepton, W_had]

outputs:
  signal: HH                 # label 1: main score P(signal)
  ttbar:  tt                 # label 2: the ttbar hypothesis also gets a score of its own
combine: [HH, tt]

training:
  features: 8
  attention_heads: 2
  attention_gate: true
  kv_projection: true
  phase_symmetric: true
  reflect_eta: false
```

### `objects`

* `jets`: the jet collection, up to 12 jets per event, read from `INPUTS/Jets`; the four kinematic
  columns are its network inputs.
* `b: jets btag 2`: the two jets most likely to be b jets; the rule reads `INPUTS/Jets/btag` for this
  choice only.
* `q: jets rest 2`: the two leading remaining jets, the light-quark jets.
* `lepton`: a single object per event; `isE` and `isM` tell the network whether it is an electron or a
  muon.
* `met`: the missing transverse momentum, a single object given by `pt` and `phi`.

### `hypotheses`

* `HH → H_bb H_WW`, `H_bb → bb̄`, `H_WW → W_had W_lep`, `W_had → qq̄′`. With two b jets and two light
  jets each of these has a single candidate.
* `W_lep: [lepton, met]` is the leptonic W, formed from the lepton and the missing momentum.
* `tt → t_had t_lep`, `t_had → b W_had`, `t_lep → b W_lep`. `W_had` and `W_lep` are the same particles
  as under `HH`. Each top takes one of the two b jets, so there are two ways to form the tt̄ pair;
  their probabilities are written to `views/tt` in the prediction files.

### `attention` and `relations`

The leptonic W candidate is compared with the hadronic W candidate, and each top candidate with the
other one. The classifier also looks directly at the pairs lepton–MET, H_bb–MET and lepton–W_had.

### `outputs` and `combine`

Two outputs, hence three classes: 0 = rest, 1 = signal (HH), 2 = tt̄; the example file has only labels
1 and 2. The classifier is built from both hypotheses, and `ttbar` also gets a tt̄ score of its own
(`heads/ttbar` in the prediction files).

### `training`

All network settings at their defaults except `reflect_eta: false`.
`examples/bbww/bbww_hyperparameters.yaml` writes every setting out ([Options.md](./Options.md)).

## Training, evaluation and export

Quick test on the small file (2 epochs):

```bash
corenet-validate --config examples/bbww/bbww.yaml --input examples/data/bbww_small.h5
corenet-train --config examples/bbww/bbww.yaml --input examples/data/bbww_small.h5 \
    --fold 0 --exclude-folds 1 --epochs 2 --device cuda --output-dir runs/bbww
corenet-evaluate --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_small.h5 \
    --output runs/bbww/eval_fold0.h5
corenet-export --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_small.h5 \
    --output-dir runs/bbww/export
```

Full training (20 epochs) on the full file, on a GPU:

```bash
corenet-train --config examples/bbww/bbww.yaml --input examples/data/bbww_all.h5 \
    --fold 0 --exclude-folds 1 --device cuda --output-dir runs/bbww_full
corenet-evaluate --checkpoint runs/bbww_full/model_fold0.pkl --input examples/data/bbww_all.h5 \
    --output runs/bbww_full/eval_fold0.h5
```

The commands, their options and their output files are the same as in the
[HH → bb̄bb̄ guide](./HH4b.md). The prediction file additionally holds `views/tt` (the two tt̄
pairings) and `heads/ttbar` (the tt̄ score), and `class_probabilities` has three columns (rest,
signal, ttbar). The exported model takes `b`, `q`, `lepton`, `met` and `event`.
