# Walkthrough: from clone to a trained HH → 4b classifier

This guide goes through the HH → bb̄bb̄ example end to end: installation, the input file, the
configuration line by line, validation, training, evaluation and export. Every command is run from
the repository root. The expected outputs are copied from actual runs of this release.

## a. Install

```bash
git clone https://github.com/[owner]/CoReNet.git     # URL to be confirmed
cd CoReNet
pixi install -e gpu            # CUDA build of PyTorch; `pixi install` alone gives the CPU build
pixi shell -e gpu              # activate the environment; or prefix each command with `pixi run -e gpu`
```

`pixi.lock` records the exact conda-forge build of every package (Python, PyTorch with its own CUDA
and cuDNN, NumPy, h5py, ...), so `pixi install` gives the same environment on any Linux x86-64 machine.
The environment provides `corenet-validate`, `corenet-train`, `corenet-evaluate` and `corenet-export`.

## b. The data

The repository ships `examples/data/hh4b_small.h5` (20,000 events, 7.8 MB), a stratified random
subsample of the full file `examples/data/hh4b_all.h5` (454,261 events: 417,978 multijet, 36,283 HH;
176 MB). The full file is distributed separately: [Zenodo DOI to be added]; download it into
`examples/data/`. Both files have the same layout. The datasets CoReNet reads:

| path | shape | dtype | meaning |
|---|---|---|---|
| `INPUTS/Source/pt`, `eta`, `phi`, `mass` | (N, 12) | float32 | jet four-momenta, 12 slots, pT-ordered |
| `INPUTS/Source/btag` | (N, 12) | float32 | b tag of each jet from the detector simulation, 0 or 1 |
| `INPUTS/Source/MASK` | (N, 12) | bool | True for a real jet, False for a padded slot (features 0) |
| `CLASSIFICATIONS/EVENT/signal` | (N,) | int64 | label: 0 = multijet, 1 = HH |
| `EVENT/weight` | (N,) | float32 | event weight |
| `EVENT/fold` | (N,) | int64 | cross-validation fold, 0..4 |
| `EVENT/event` | (N,) | int64 | event identifier, copied to the prediction files |

The group is called `Source`; the collection `jets` of the configuration finds it through the alias
`jets` → `Jets`, `Source` (docs/data_format.md).

**The file does not pre-select the four Higgs-candidate jets.** It stores all selected jets of the
event (at least 4 and at most 12 per event, pT-ordered, padded with zeros and flagged in `MASK`),
each with its 0/1 b tag. The four candidates are chosen when the file is loaded, by the role
definition in the configuration (section c). The first event of the small file, for example:

```
jet slot      0      1      2      3      (slots 4-11 padded, MASK False)
pt [GeV]  306.2  192.7   88.0   37.6
btag        0.0    0.0    1.0    0.0
```

In the full file 13% of the events have no b-tagged jet, 35% one, 33% two, 15% three and 4% four or
more, so most events have fewer than four tagged jets.

**Weights.** In `hh4b_all.h5` the weights of each class sum to 1 (in the small file to about 0.044
each). The training recipe re-normalises them per class over the training events anyway
(`class_norm`, section e), so only the relative weights within a class matter. The full file has
5 events with negative weights; they count as 0 in the loss and the metrics.

**Folds.** `EVENT/fold` splits the events into 5 folds of about 91,000 events (about 4,000 in the
small file). `--fold k` holds out fold `k` for testing; `--exclude-folds j` takes fold `j` out of
training and uses it for the per-epoch validation.

**Truth pairing (not used by CoReNet).** For signal events `TARGETS/h1/{b1,b2}` and
`TARGETS/h2/{b1,b2}` give the jet slots (0..11) of the generator-matched b quarks of each Higgs boson
(−1 if unmatched; 43% of the signal events are fully matched). No CoReNet command reads them: the
pairing is never supervised. They are there for pairing diagnostics, to compare with the pairing
probabilities of section f. The file holds a few more datasets that the loader ignores
(`EVENT/nSelJets`, `EVENT/jet_flavor`, `EVENT/jet_btag_bits`, `EVENT/is_unique_4b_matched`, ...).

## c. The configuration

`examples/hh4b/hh4b.yaml` in full:

```yaml
objects:
  jets:  {features: [pt, eta, phi, mass, btag], max: 12}   # INPUTS/Source (alias of "jets") in the H5 file
  b:     jets btag 4                                        # the 4 highest-btag jets
  other: jets rest 8                                        # up to 8 remaining jets, pT ordered
event: [nSelJets, HT]                                       # computed from the jet collection

hypotheses:
  HH:
    H1: [b, b]
    H2: [b, b]
    context: other           # the H candidates attend to all jets (the 4 b plus the others)

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

**`jets: {features: [...], max: 12}`** declares a collection. `features` lists every column the
loader **reads** from the file for this collection; it is not the list of network inputs. The
kinematic features `pt`, `eta`, `phi`, `mass` become network inputs through the four-vector
embedding (log(1+pT), η, Δφ, log(1+m) of every jet and every jet combination). `btag` is a
**selection tag**: it is consumed by the `btag` role selector only and never fed to the network
(`TAG_FEATURES` in `src/corenet/models/network.py`). The parameter count is 1,508 with or without
`btag` in the list. This is what the paper means by "b tagging is used only to choose the candidate
jets". `max: 12` is the number of slots used.

**`b: jets btag 4`** is a role: 4 members of `jets`, chosen by the `btag` selector. Among the jets
present in `MASK` the loader sorts by `btag × 10⁶ + pT` and takes the first four: the tagged jets by
decreasing pT, then, if fewer than four jets are tagged, the untagged jets by decreasing pT. So
`btag` in the file is a flag per jet, not a ranking; with a continuous score in that column the
same selector would rank by the score. For the first event above, `b` = (88.0, 306.2, 192.7, 37.6)
GeV: the tagged jet first, then the three untagged ones.

**`other: jets rest 8`** takes up to 8 of the jets that no earlier role used, by decreasing pT
(roles are filled in file order). In the first event all four jets went to `b`, so all eight `other`
slots are padded. Missing members are padded and masked in the network; the event is not dropped.

**Why at load time.** One file serves every role definition: `b: jets pt 4` (the four leading jets),
a different number of context jets, or a continuous tagger only change the configuration; the file
does not have to be regenerated.

**Events with few jets or few tags.** The loader enforces no minimum. Fewer than four tagged jets is
the common case and is handled as above (untagged jets fill `b`); the `b` slots are empty only if the
event has fewer than four jets in total, and then none of the three pairings exists and the event
has no candidate information. The example files were written with at least four jets per event, and
`corenet-validate` reports how often every slot of each role is filled
(`leaf b: ... all 4 slot(s) filled in 100.0% of the events`). It does not count tagged jets. Apply a
four-jet selection when you prepare your own files.

**`event: [nSelJets, HT]`** are event-level scalars. Both are **computed by the loader**, not read
from the file: `nSelJets` is the number of `True` entries of `INPUTS/Source/MASK` and `HT` the scalar
sum of the pT of those jets (any `EVENT/nSelJets` in the file is ignored). They are embedded and
added to the embedding of every jet; `nSelJets` enters as log(nSelJets − 3).

**`hypotheses: HH: {H1: [b, b], H2: [b, b]}`** is the decay tree HH → H1 H2, H → b b. CoReNet
enumerates every assignment of the four `b` jets: the two b of a Higgs are interchangeable, so there
are 6 dijet candidates (01, 23, 02, 13, 03, 12); the two Higgs bosons are interchangeable, so there
are 3 pairings of the four jets into two dijets (01|23, 02|13, 03|12), the **views** of HH. H1 and H2
have the same structure and share one set of weights. A learned score per pairing and a softmax over
the three give the selected HH candidate.

**`context: other`** lets every dijet candidate attend to the jets of the event: the four `b` jets
and the up to eight `other` jets, except its own two jets.

**`outputs: {signal: HH}`, `combine: [HH]`.** One output, hence two classes: 0 = rest (multijet),
1 = signal, read from `CLASSIFICATIONS/EVENT/signal`. The main head classifies from the selected
candidate of the hypotheses listed in `combine`.

**`training:`** sets the network options (all values above are the defaults; `reflect_eta: true` is
the default too). `features` is the embedding dimension d of every pixel, the main size knob. The
parameter counts of this network (all other options at their defaults):

| change | parameters |
|---|---|
| `features: 4` / **`8`** / `16` / `32` | 504 / **1,508** / 5,052 / 18,284 |
| `attention_heads: 1` / `4` | 1,824 / 1,428 |
| `attention_gate: false` | 1,378 |
| `kv_projection: false` | 1,220 |
| `phase_symmetric: false` | 2,260 |
| `reflect_eta: false`, `b: jets pt 4`, `other: jets rest 4` | 1,508 (unchanged) |

`features` must be divisible by `attention_heads` and even when `phase_symmetric` is true.
`examples/hh4b/hh4b_hyperparameters.yaml` is the same network with every network and training-recipe
option written out at its default value (docs/hyperparameters.md).

## d. Validate

```bash
corenet-validate --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5
```

The configuration is parsed by the trainer's own parser and the network is built and summarised;
then every dataset the loader will read is checked (shape, dtype, slots, labels in range, finite
weights, folds), the loader reads the file, and the network runs a forward pass on 512 events.
Expected output:

```
== configuration examples/hh4b/hh4b.yaml
  [  OK] sections, names, types and ranges (parse_config)
  [  OK] collection `jets`: max 12, features ['pt', 'eta', 'phi', 'mass', 'btag']
  [  OK] role `b`: 4 x `jets` selected by `btag`
  [  OK] role `other`: 8 x `jets` selected by `rest`
  [  OK] hypothesis `HH`: 3 view(s); nodes (candidates): H1 (6), HH (3); context `other`
  [  OK] classes ['rest', 'signal']; main output `signal` on `HH`; auxiliary heads []; combine ['HH']
  [  OK] network built: 1,508 parameters

== data examples/data/hh4b_small.h5
  [  OK] 20,000 events (first dimension of INPUTS/Source/pt)
  [  OK] INPUTS/Source/MASK (20000, 12) bool
  [  OK] collection `jets` (used by `b`, `other`): INPUTS/Source with ['pt', 'eta', 'phi', 'mass', 'btag'], 12 slots
  [  OK] event scalar `nSelJets` computed from collection `jets`
  [  OK] event scalar `HT` computed from collection `jets`
  [  OK] labels from CLASSIFICATIONS/EVENT/signal: events per class {'rest': 18403, 'signal': 1597}
  [  OK] EVENT/weight: finite
  [  OK] EVENT/fold: folds {0: 4005, 1: 3965, 2: 4060, 3: 3950, 4: 4020}
  [  OK] leaf `b`: tensor (20000, 6, 4), all 4 slot(s) filled in 100.0% of the events
  [  OK] leaf `other`: tensor (20000, 6, 8), all 8 slot(s) filled in 0.0% of the events
  [  OK] loader + forward pass on 512 events: finite outputs, logits (512, 2)

SUMMARY: 18 checks passed, 0 warnings, 0 errors -> valid
```

`H1 (6), HH (3)` are the 6 dijets and 3 pairings of section c. The leaf tensors are
`(events, 5 features + presence flag, slots)`; `other` is completely filled only in events with 12
jets, hence 0.0%. On `hh4b_all.h5` the same checks pass with one `WARN` for the 5 negative weights
(`SUMMARY: 18 checks passed, 1 warnings, 0 errors -> valid`). A failed check prints `[FAIL]` with the
dataset concerned and the exit status is 1.

## e. Train

CPU smoke test on the small file (2 epochs, a few seconds):

```bash
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5 \
    --fold 0 --exclude-folds 1 --epochs 2 --device cpu --output-dir runs/hh4b
```

Full recipe on the full file, fold 0 held out for testing, fold 1 for validation, on a GPU:

```bash
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_all.h5 \
    --fold 0 --exclude-folds 1 --device cuda --output-dir runs/hh4b_full
```

The recipe (defaults of `src/corenet/train.py`): 20 epochs of Adam at learning rate 0.01; batch size
1024 at epoch 0, doubled at epochs 1, 3, 6 and 10 with 64, 16, 4, 1 and 0 ghost batches; the input
normalisation fixed from the training events before epoch 0; from epoch 15 only the output heads are
trained and the learning rate is multiplied by 0.25 for every further epoch (0.0025 at epoch 15,
6.25e-4 at epoch 16, ...); per-class weight normalisation (`class_norm`); seed 42 + fold. The model
after the last epoch is kept. Output on a V100 (abridged):

```
[INFO] examples/data/hh4b_all.h5: 454,261 events, device=cuda, recipe epochs=20, freeze_epoch=15, lr=0.01, ...
[INFO] fold 0: 272,867 training / 90,565 validation / 90,829 held-out events
[INFO] examples/hh4b/hh4b.yaml: classes ['rest', 'signal'], 1,508 parameters
[INFO] fold 0 | epoch 00 | bs   1024 | gbn 64 | lr 1.00e-02 | loss 0.54948 | val loss 0.48370 | val auc 0.84804
[INFO] fold 0 | epoch 01 | bs   2048 | gbn 16 | lr 1.00e-02 | loss 0.47842 | val loss 0.49644 | val auc 0.85266
...
[INFO] fold 0 | epoch 13 | bs  16384 | gbn  0 | lr 1.00e-02 | loss 0.44804 | val loss 0.45840 | val auc 0.86600
[INFO] fold 0 | epoch 14 | bs  16384 | gbn  0 | lr 2.50e-03 | loss 0.44812 | val loss 0.45932 | val auc 0.86667
[INFO] fold 0: frozen at epoch 15, 18 trainable head parameters
[INFO] fold 0 | epoch 15 | bs  16384 | gbn  0 | lr 6.25e-04 | loss 0.44742 | val loss 0.45559 | val auc 0.86697
...
[INFO] fold 0 | epoch 19 | bs  16384 | gbn  0 | lr 2.44e-06 | loss 0.44574 | val loss 0.45513 | val auc 0.86707
[INFO] fold 0: saved model_fold0.pkl, history_fold0.json and predictions (76s)
```

| field | meaning |
|---|---|
| `bs` | batch size of the epoch |
| `gbn` | number of ghost batches each batch is split into for batch normalisation (0: running statistics) |
| `lr` | learning rate **after** the epoch's scheduler step, i.e. the rate the next epoch runs at (epoch 14 runs at 0.01 and prints 2.50e-03) |
| `loss` | weighted cross entropy on the training events, averaged over the epoch |
| `val loss`, `val auc` | weighted cross entropy and weighted ROC AUC on the `--exclude-folds` events after the epoch |

The whole command takes about 2 minutes on a V100 (128 s including loading the file). In
`--output-dir`:

| file | content |
|---|---|
| `model_fold0.pkl` | checkpoint: weights, configuration, classes, recipe; self-contained for evaluation and export |
| `history_fold0.json` | the recipe and, per epoch, the fields of the log above |
| `predictions_fold0.h5` | predictions on the held-out fold 0, in the layout of section f (without the ROC curve) |

**Expected result.** The fold-0 test AUC (section f) is 0.867 to 0.870; the run above gives 0.8699,
the research-code reference 0.8674. GPU training is seeded but not bit-deterministic, so a spread of
about 0.003 between runs is normal. The 2-epoch CPU smoke test gives a test AUC of about 0.769.

**All five folds.** `--fold -1` (the default) trains one model per fold in turn. The paper's numbers
are the mean ± standard deviation over the five test folds, each trained on three folds with the next
fold, (k + 1) mod 5, for validation. `--exclude-folds` is the same for every fold of a `--fold -1`
run, so for that scheme run the folds one by one:

```bash
for k in 0 1 2 3 4; do
  corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_all.h5 \
      --fold $k --exclude-folds $(( (k + 1) % 5 )) --device cuda --output-dir runs/hh4b_full
done
```

## f. Evaluate

```bash
corenet-evaluate --checkpoint runs/hh4b_full/model_fold0.pkl --input examples/data/hh4b_all.h5 \
    --output runs/hh4b_full/eval_fold0.h5
```

By default the fold held out in training is evaluated (`--fold k` for another fold, `--fold -1` for
all events), on the CPU (`--device cuda` to use a GPU); the configuration is taken from the
checkpoint. Expected output:

```
[INFO] runs/hh4b_full/model_fold0.pkl: 1,508 parameters; evaluating 90,829 events (fold 0) of examples/data/hh4b_all.h5
[INFO] wrote runs/hh4b_full/eval_fold0.h5 and runs/hh4b_full/eval_fold0.h5.json
{
 "main": 0.8698698175283675,
 "one_vs_rest": {
  "rest": 0.8698698191928733,
  "signal": 0.8698698175283675
 },
 "heads": {}
}
```

`main` is the weighted ROC AUC of signal against the rest, scored by P(signal); with two classes the
one-vs-rest AUCs are the same number; `heads` is empty because this network has no auxiliary output.
The metrics are also written to `eval_fold0.h5.json`. `eval_fold0.h5` contains

| dataset | shape | content |
|---|---|---|
| `class_probabilities` | (90829, 2) | P(rest), P(signal); column 1 is the classifier score |
| `views/HH` | (90829, 3) | probabilities of the three pairings 01\|23, 02\|13, 03\|12 (they sum to 1) |
| `roc/fpr`, `roc/tpr`, `roc/threshold` | (n,) | ROC curve of the main output |
| `EVENT/index`, `event`, `label`, `weight`, `fold` | (90829,) | row in the input file and the event's identifier, label, weight, fold |

**Reading the pairing probabilities.** The indices 0..3 in `01|23` are the slots of the `b` role,
that is, the four jets in the order of the `btag` selector (tagged jets by decreasing pT, then
untagged ones), not the slots of `INPUTS/Source`. `views/HH[i].argmax()` is the pairing the network
selects for event `i`; a value near 1 means one pairing dominates, a flat (1/3, 1/3, 1/3) row means
none does. The view scores are learned only through the classification loss, so these
probabilities are a by-product. To compare them with the truth pairing in `TARGETS`, map each `b`
slot back to its jet slot by applying the selector order (stable sort of `-(btag × 10⁶ + pT)` over the
present jets).

```python
import h5py
with h5py.File("runs/hh4b_full/eval_fold0.h5") as f:
    p_signal = f["class_probabilities"][:, 1]
    pairing = f["views/HH"][...].argmax(1)       # 0: 01|23, 1: 02|13, 2: 03|12
    rows = f["EVENT/index"][...]                 # rows of the events in hh4b_all.h5
```

## g. Export

```bash
corenet-export --checkpoint runs/hh4b_full/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output-dir runs/hh4b_full/export
```

writes the model as TorchScript (`model_fold0.pt`) and ONNX (`model_fold0.onnx`) with a JSON
description, and checks both against PyTorch on events of `--input` (TorchScript agrees exactly, ONNX
to float32 rounding, 6.8e-06 on the quick-start model). The exported graph starts after the role
selection: its inputs are the `b` (N, 6, 4) and `other` (N, 6, 8) tensors and the event scalars
(N, 2) as the loader builds them, so the jet selection of section c has to be reproduced wherever the
model is deployed. Details in [export.md](export.md).

## h. Changing things

| to change | where | example |
|---|---|---|
| network width d | `training: features` in the configuration | `features: 16` (5,052 parameters) |
| how the 4 candidate jets are chosen | the `b` role in `objects` | `b: jets pt 4` (the 4 leading jets, b tag ignored) |
| number of context jets | the `other` role in `objects` (the `n` of all roles ≤ `max`) | `other: jets rest 4` |
| order of the `rest` jets | `trainer: rest_order` or `--rest-order` | `selector`: continue the `btag` ordering |
| epochs, learning rate, decay | `--epochs`, `--lr`, `--lr-milestones`, `--lr-gamma`, `--freeze-epoch`, or the `trainer:` section | `--epochs 30 --lr-milestones 25 26 27 28 29 --freeze-epoch 25` |
| batch size | `--batch-size-init` or `trainer: batch_size_init` | the doubling epochs are fixed |
| seed | `--seed` or `trainer: seed` | fold k uses seed + k |
| validation / test folds | `--fold`, `--exclude-folds` | section e |

Changes in `objects` or `training:` give a different network: retrain, and validate the new
configuration first with `corenet-validate`. A command-line flag overrides the `trainer:` section,
which overrides the built-in default (docs/training.md). In YAML, write floating-point numbers with
a decimal point (`5.0e-3`, not `5e-3`).
