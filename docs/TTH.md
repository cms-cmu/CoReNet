# ttH → bb̄ Example Guide

tt̄H with H → bb̄, tt̄ → b qq̄′ b ℓν, against semileptonic tt̄ → b qq̄′ b ℓν. The Higgs boson decays to
two b jets, one top quark decays hadronically and the other leptonically. The background, semileptonic
tt̄, has the same objects except the two b jets of the Higgs boson. This guide builds on the
[HH → bb̄WW guide](./BBWW.md) and shows what is new: four b jets shared by three particles, a
`min_jets` above the default, and W candidates that are declared under a hypothesis of their own
because two top quarks use them.

## Data

`examples/data/tth_small.h5` (20,000 events: 10,009 ttH, 9,991 tt̄; 7.9 MB) is a random subsample of
`examples/data/tth_all.h5` (423,753 events: 212,067 ttH, 211,686 tt̄; 166 MB), which is distributed
separately on Zenodo ([doi:10.5281/zenodo.23567532](https://doi.org/10.5281/zenodo.23567532)); download it into `examples/data/`.

The events are generated at 13 TeV and passed through the Delphes fast simulation of the CMS detector:

* ttH: `p p → t t̄ H` from a MadGraph5_aMC@NLO leading-order gridpack, showered and decayed with
  Pythia 8, H → bb̄ forced;
* tt̄: inclusive tt̄ production from Pythia 8.

Both samples keep the events with exactly one W boson decaying to an electron or a muon and a neutrino
and one W boson decaying to quarks (W → τν rejected), with a generator lepton of pT > 20 GeV. The
selection asks for one tight lepton, a Z-boson veto and a τ veto; jets with pT > 25 GeV, |η| < 2.4 and
ΔR > 0.4 from the lepton; at least six jets and at least two b-tagged jets.

The layout ([Dataset.md](./Dataset.md)) is that of the HH → bb̄WW file, without `TARGETS` groups:

| path | shape | meaning |
|---|---|---|
| `INPUTS/Jets/pt`, `eta`, `phi`, `mass`, `btag`, `MASK` | (N, 12) | all jets of the event, 12 slots |
| `INPUTS/Lepton/pt`, `eta`, `phi`, `mass`, `isE`, `isM` | (N,) | the lepton; `isE`, `isM` flag electron or muon |
| `INPUTS/Met/pt`, `phi` | (N,) | the missing transverse momentum |
| `EVENT/label` | (N,) | 1 = ttH, 0 = tt̄ |
| `EVENT/weight`, `EVENT/fold`, `EVENT/event` | (N,) | weight, fold (5 folds), event identifier |

## Event `.yaml` File

The decay-chain part of `examples/tth/tth.yaml`:

```yaml
objects:
  jets:   {features: [pt, eta, phi, mass], max: 12}        # INPUTS/Jets; features = network inputs
  b:      jets btag 4                                       # the 4 highest-btag jets: two from the tops, two from the Higgs boson
  q:      jets rest 2                                       # the 2 leading remaining jets: the hadronic W
  lepton: {features: [pt, eta, phi, mass, isE, isM]}        # INPUTS/Lepton, a single object (max 1)
  met:    {features: [pt, phi]}                             # INPUTS/Met, a single object (max 1)
event: [nSelJets, HT]
min_jets: 6

hypotheses:
  ttH:
    H:     [b, b]            # the Higgs boson: one of the 6 pairs of b jets
    t_had: [b, W_had]        # the hadronic top: a b jet and the hadronic W
    t_lep: [b, W_lep]        # the leptonic top: a b jet and the leptonic W; 12 assignments of the 4 b jets in all
  WW:                        # declares the two W candidates that both tops use (a shared node needs a parent;
    W_had: [q, q]            #   in bbww.yaml that parent is H_WW); WW has a single candidate
    W_lep: [lepton, met]     # the leptonic W, from the lepton and the missing momentum

attention:
  - W_lep: W_had
  - t_had: t_lep             # the two tops never use the same b jet

relations:                   # pairs the classifier looks at directly
  - [lepton, met]
  - [lepton, W_had]

outputs:
  signal: ttH                # label 1: ttH; label 0: semileptonic ttbar
combine: [ttH, WW]           # WW is listed so that its W candidates reach the output

training:
  features: 8
  attention_heads: 2
  attention_gate: true
  kv_projection: true
  phase_symmetric: true
  reflect_eta: true        # use the forward-backward symmetry of the detector
```

### `objects`

* `jets`: the jet collection, up to 12 jets per event, read from `INPUTS/Jets`; the four kinematic
  columns are its network inputs.
* `b: jets btag 4`: the four jets most likely to be b jets: two from the top quarks, two from the Higgs
  boson. The rule reads `INPUTS/Jets/btag` for this choice only.
* `q: jets rest 2`: the two leading remaining jets, the light-quark jets of the hadronic W.
* `lepton`: a single object per event; `isE` and `isM` tell the network whether it is an electron or a
  muon.
* `met`: the missing transverse momentum, a single object given by `pt` and `phi`.
* `min_jets: 6`: the selection keeps events with at least six jets, so the jet multiplicity enters the
  network as `log(nSelJets - 5)`, zero for six jets ([EventInfo.md](./EventInfo.md)).

### `hypotheses`

* `ttH → H t_had t_lep`, `H → bb̄`, `t_had → b W_had`, `t_lep → b W_lep`. The Higgs boson takes one of
  the six pairs of the four b jets, and the two remaining b jets go to the hadronic and the leptonic top
  in one of two ways: 12 assignments in all. Their probabilities are written to `views/ttH` in the
  prediction files.
* `WW → W_had W_lep`, `W_had → qq̄′`, `W_lep: [lepton, met]`: the hadronic W from the two light jets and
  the leptonic W from the lepton and the missing momentum. Both tops use these W candidates. A particle
  used by two parents is declared once, under a hypothesis of its own, and that hypothesis is listed in
  `combine`; `t_had` and `t_lep` then refer to `W_had` and `W_lep` by name. With two light jets and one
  lepton, `WW` has a single candidate.

### `attention` and `relations`

The leptonic W candidate is compared with the hadronic W candidate, and each top candidate with the
other one; a top never pairs with the top that uses the same b jet. The classifier also looks directly
at the pairs lepton–MET and lepton–W_had.

### `outputs` and `combine`

One output, `signal: ttH`, hence two classes: 0 = rest (semileptonic tt̄), 1 = signal (ttH), read from
`EVENT/label`. The classifier is built from both hypotheses in `combine`: `ttH` and `WW`.

### `training`

All network settings at their defaults. [Options.md](./Options.md)
describes every setting.

## Training, evaluation and export

Every command is run from the repository root in the pixi environment of the [README](../README.md).

1. Validate the configuration against the input file:

   ```bash
   corenet-validate --config examples/tth/tth.yaml --input examples/data/tth_small.h5
   ```

   * `--config`: the event description.
   * `--input`: the file to check against it.

   | output | content |
   |---|---|
   | report on the terminal | every check, ending with `valid` or `NOT VALID` |

2. Train a quick test on the small file (2 epochs):

   ```bash
   corenet-train --config examples/tth/tth.yaml --input examples/data/tth_small.h5 \
       --fold 0 --exclude-folds 1 --epochs 2 --device cuda --output-dir runs/tth
   ```

   * `--fold 0`: hold out fold 0 for testing.
   * `--exclude-folds 1`: keep fold 1 out of the training and use it for validation.
   * `--epochs 2`: train for 2 epochs instead of 20.
   * `--device cuda`: train on the GPU.

   | file in `runs/tth` | content |
   |---|---|
   | `model_fold0.pkl` | checkpoint; evaluation and export need no configuration file |
   | `history_fold0.json` | training settings and per-epoch history |
   | `predictions_fold0.h5` | predictions on the held-out fold 0 |

3. Evaluate the model:

   ```bash
   corenet-evaluate --checkpoint runs/tth/model_fold0.pkl --input examples/data/tth_small.h5 \
       --output runs/tth/eval_fold0.h5
   ```

   * `--fold -1`: evaluate every event of the file instead of the held-out fold 0.
   * `--device cuda`: evaluate on the GPU (default: CPU).

   | file | content |
   |---|---|
   | `eval_fold0.h5` | `class_probabilities` (rest, signal), `views/ttH` (the 12 assignments of the four b jets), `roc/`, `EVENT/` |
   | `eval_fold0.h5.json` | the metrics |

4. Export the model:

   ```bash
   corenet-export --checkpoint runs/tth/model_fold0.pkl --input examples/data/tth_small.h5 \
       --output-dir runs/tth/export
   ```

   * `--input`: events on which the exported models are checked against PyTorch.
   * `--formats torchscript` or `--formats onnx`: export only one of the two formats.

   | file in `runs/tth/export` | content |
   |---|---|
   | `model_fold0.pt` | TorchScript model |
   | `model_fold0.onnx` | ONNX model |
   | `model_fold0.json` | description of the inputs and outputs |

   The exported model takes `b` (N, 5, 4), `q` (N, 5, 2), `lepton` (N, 7, 1), `met` (N, 3, 1) and
   `event` (N, 2).

5. Write the results page of the run ([Report.md](./Report.md)):

   ```bash
   corenet-report --run-dir runs/tth --input examples/data/tth_small.h5 --output runs/tth/report.html
   ```

   * `--run-dir`: the directory written by `corenet-train` and `corenet-evaluate`.
   * `--max-events 0`: use every event of the input file for the input and weight distributions.

   | file | content |
   |---|---|
   | `report.html` | the page, figures embedded |

6. Full training (20 epochs) on the full file, on a GPU, then evaluation of every event and the report:

   ```bash
   corenet-validate --config examples/tth/tth.yaml --input examples/data/tth_all.h5
   corenet-train --config examples/tth/tth.yaml --input examples/data/tth_all.h5 \
       --fold 0 --exclude-folds 1 --device cuda --output-dir runs/tth_full
   corenet-evaluate --checkpoint runs/tth_full/model_fold0.pkl --input examples/data/tth_all.h5 \
       --output runs/tth_full/eval_fold0.h5 --fold -1
   corenet-report --run-dir runs/tth_full --input examples/data/tth_all.h5 \
       --output runs/tth_full/report.html
   ```

   * `--fold 0 --exclude-folds 1`: fold 0 held out for testing, fold 1 for validation.
   * `--fold -1` (evaluation): predictions for every event, so the report shows the training and
     validation folds next to the test fold.

   | file in `runs/tth_full` | content |
   |---|---|
   | `model_fold0.pkl`, `history_fold0.json`, `predictions_fold0.h5` | as in step 2 |
   | `eval_fold0.h5`, `eval_fold0.h5.json` | predictions and metrics for every event |
   | `report.html` | the results page |

The commands, their options and their output files are the same as in the
[HH → bb̄bb̄ guide](./HH4b.md).
