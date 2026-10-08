# HH → bb̄bb̄ Example Guide

Two Higgs bosons, each decaying to a pair of b jets, against multijet background. This guide goes
through the example end to end: the data, the event description, validation, training, evaluation
and export. Every command is run from the repository root in the pixi environment of the
[README](../README.md).

## Data

The repository ships `examples/data/hh4b_small.h5` (20,000 events, 8.1 MB), a random subsample of the
full file `examples/data/hh4b_all.h5` (454,261 events: 417,978 multijet, 36,283 HH; 184 MB). The full
file is distributed separately ([Zenodo DOI to be added]); download it into `examples/data/`. Both
files have the same layout ([Dataset.md](./Dataset.md)):

| path | shape | dtype | meaning |
|---|---|---|---|
| `INPUTS/Source/pt`, `eta`, `phi`, `mass` | (N, 12) | float32 | the jets of the event, 12 slots |
| `INPUTS/Source/btag` | (N, 12) | float32 | b tag of each jet from the detector simulation, 0 or 1 |
| `INPUTS/Source/MASK` | (N, 12) | bool | True for a real jet, False for an empty slot |
| `CLASSIFICATIONS/EVENT/signal` | (N,) | int64 | label: 0 = multijet, 1 = HH |
| `EVENT/weight` | (N,) | float32 | event weight |
| `EVENT/fold` | (N,) | int64 | cross-validation fold, 0..4 |
| `EVENT/event` | (N,) | int64 | event identifier, copied to the prediction files |

The file stores all jets of the event (at least 4 and at most 12), not only the four Higgs-candidate
jets; those are chosen by the event description when the file is loaded. The five folds hold about 91,000 events each (about 4,000 in the small file).

For signal events, `TARGETS/h1/{b1,b2}` and `TARGETS/h2/{b1,b2}` give the jets of the
generator-matched b quarks of each Higgs boson (−1 if unmatched). No CoReNet command reads them; they are there to
compare with the pairing probabilities in the prediction files (Evaluation below). The file holds a few more datasets that are ignored.

## Event `.yaml` File

The decay-chain part of `examples/hh4b/hh4b.yaml`:

```yaml
objects:
  jets:  {features: [pt, eta, phi, mass, btag], max: 12}   # INPUTS/Source (alias of "jets") in the H5 file
  b:     jets btag 4                                        # the 4 highest-btag jets
  other: jets rest 8                                        # up to 8 remaining jets
event: [nSelJets, HT]                                       # computed from the jet collection

hypotheses:
  HH:
    H1: [b, b]
    H2: [b, b]
    context: other           # the Higgs candidates also see the other jets of the event

outputs:
  signal: HH                 # label 1 = signal, label 0 = rest (multijet background)
combine: [HH]

training:
  features: 8
  attention_heads: 2
  attention_gate: true
  kv_projection: true
  phase_symmetric: true
```

### `objects`

* `jets`: the jet collection, with up to 12 jets per event, read from `INPUTS/Source`. The listed
  features are the columns to read from the input file. `btag` is only used to pick which jets are
  treated as b-jet candidates; it is not an input to the network.
* `b: jets btag 4`: the four jets most likely to be b jets, taken from the jet collection. When fewer
  than four jets are tagged, the remaining places go to untagged jets.
* `other: jets rest 8`: up to eight of the remaining jets. Events with fewer jets are kept.
* `event: [nSelJets, HT]`: the number of jets and their scalar momentum sum, given to the network as
  per-event quantities.

Because the b jets are chosen when the file is loaded, trying `b: jets pt 4` (the four leading
jets), a different number of other jets, or a continuous tagger score only changes this file, not the
data.

### `hypotheses`

`HH → H1 H2`, `H → bb̄`. The two b of a Higgs are interchangeable, and so are the two Higgs bosons, so
the four b jets can be paired into two Higgs bosons in three ways (01|23, 02|13, 03|12). `context: other`
lets each Higgs candidate also see the other jets of the event.

### `outputs` and `combine`

One output, `signal: HH`, hence two classes: 0 = rest (multijet), 1 = signal, read from
`CLASSIFICATIONS/EVENT/signal`. The classifier is built from the hypotheses in `combine`.

### `training`

The network settings, all at their default values here. `examples/hh4b/hh4b_hyperparameters.yaml`
is the same file with every network and training setting written out ([Options.md](./Options.md)).

## Validation

```bash
corenet-validate --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5
```

checks the event description, the input file, and that the two fit together.

## Training

A quick test on the small file (2 epochs):

```bash
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5 \
    --fold 0 --exclude-folds 1 --epochs 2 --device cuda --output-dir runs/hh4b
```

Full training on the full file, fold 0 held out for testing, fold 1 for validation, on a GPU:

```bash
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_all.h5 \
    --fold 0 --exclude-folds 1 --device cuda --output-dir runs/hh4b_full
```

In `--output-dir`:

| file | content |
|---|---|
| `model_fold0.pkl` | checkpoint, self-contained for evaluation and export |
| `history_fold0.json` | the training settings and the per-epoch history |
| `predictions_fold0.h5` | predictions on the held-out fold 0, in the layout of the evaluation below (without the ROC curve) |

**All five folds.** `--fold -1` (the default) trains one model per fold in turn. To validate each
fold on the next one, (k + 1) mod 5, run the folds one by one:

```bash
for k in 0 1 2 3 4; do
  corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_all.h5 \
      --fold $k --exclude-folds $(( (k + 1) % 5 )) --device cuda --output-dir runs/hh4b_full
done
```

## Evaluation

```bash
corenet-evaluate --checkpoint runs/hh4b_full/model_fold0.pkl --input examples/data/hh4b_all.h5 \
    --output runs/hh4b_full/eval_fold0.h5
```

By default the fold held out in training is evaluated (`--fold k` for another fold, `--fold -1` for
all events); pass `--device cuda` to evaluate on the GPU. The metrics are also
written to `eval_fold0.h5.json`. `eval_fold0.h5` contains

| dataset | shape | content |
|---|---|---|
| `class_probabilities` | (90829, 2) | P(rest), P(signal); column 1 is the classifier score |
| `views/HH` | (90829, 3) | probabilities of the three pairings 01\|23, 02\|13, 03\|12 |
| `roc/fpr`, `roc/tpr`, `roc/threshold` | (n,) | ROC curve of the classifier score |
| `EVENT/index`, `event`, `label`, `weight`, `fold` | (90829,) | row in the input file and the event's identifier, label, weight, fold |

The indices 0..3 in `01|23` number the four b jets in the order they were taken by `b: jets btag 4`
(tagged jets first), not the slots of `INPUTS/Source`.

```python
import h5py
with h5py.File("runs/hh4b_full/eval_fold0.h5") as f:
    p_signal = f["class_probabilities"][:, 1]
    pairing = f["views/HH"][...].argmax(1)       # 0: 01|23, 1: 02|13, 2: 03|12
    rows = f["EVENT/index"][...]                 # rows of the events in hh4b_all.h5
```

## Export

```bash
corenet-export --checkpoint runs/hh4b_full/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output-dir runs/hh4b_full/export
```

writes the model as TorchScript (`model_fold0.pt`) and ONNX (`model_fold0.onnx`) with a JSON
description of its inputs and outputs, and checks both against PyTorch on events of `--input`. The
exported model takes the jets already assigned (`b`, `other`) and the event quantities, so the choice
of the b jets has to be reproduced wherever the model is deployed; see Exporting in the
[README](../README.md).

## Changing things

| to change | where | example |
|---|---|---|
| the width of the network | `training: features` | `features: 16` |
| how the 4 b jets are chosen | the `b` line in `objects` | `b: jets pt 4` (the 4 leading jets, b tag ignored) |
| how many other jets the Higgs candidates see | the `other` line in `objects` | `other: jets rest 4` |
| epochs, learning rate, decay | `--epochs`, `--lr`, `--lr-milestones`, `--lr-gamma`, `--freeze-epoch`, or the `trainer:` section | `--epochs 30 --lr-milestones 25 26 27 28 29 --freeze-epoch 25` |
| batch size | `--batch-size-init` or `trainer: batch_size_init` | |
| seed | `--seed` or `trainer: seed` | fold k uses seed + k |
| validation / test folds | `--fold`, `--exclude-folds` | Training above |

A change in `objects` or `training:` gives a different network: validate the new file with
`corenet-validate` and retrain ([Options.md](./Options.md)).
