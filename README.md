# CoReNet: Combinatorial Residual Network

CoReNet is a small neural network for event classification in collider physics that is built from a
description of the **decay chain** of the final state. You declare the reconstructed objects
(jets, leptons, missing momentum), how they are assigned to roles, and the decay trees of the signal and
background hypotheses in a YAML file; CoReNet enumerates every admissible assignment of objects to the
trees, builds a residual network over the resulting candidates bottom-up, selects the most compatible
candidate of each hypothesis with a learned softmax, and classifies the event. The candidate-selection
probabilities come out as a by-product.

The networks are deliberately compact: the HH → bb̄bb̄ example has **1,508** parameters and the
HH → bb̄WW example (with a tt̄ hypothesis) **3,840**.

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick start](#quick-start)
- [Walkthrough: HH → 4b from clone to a trained classifier](docs/walkthrough_hh4b.md)
- [Defining an event decay chain](#defining-an-event-decay-chain)
- [Validating a configuration and a data file](#validating-a-configuration-and-a-data-file)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model export](#model-export)
- [Hyperparameters](#hyperparameters)
- [Example data](#example-data)
- [Tests](#tests)
- [Citation and licence](#citation-and-licence)

## Installation

The repository ships a [pixi](https://pixi.sh) project (`pixi.toml` and the lock file `pixi.lock`), which installs
the exact same conda-forge builds of Python, PyTorch (with its own CUDA and cuDNN libraries) and every other
dependency on any Linux x86-64 machine, and installs the package itself in editable mode:

```bash
git clone https://github.com/[owner]/CoReNet.git     # URL to be confirmed
cd CoReNet
pixi install -e gpu              # CUDA build of PyTorch (any NVIDIA GPU supported by CUDA 12)
pixi install                     # or: CPU-only build
pixi shell -e gpu                # activate; or prefix every command with `pixi run -e gpu`
```

pixi itself is a single static binary (`curl -fsSL https://pixi.sh/install.sh | sh`) that needs no root access,
so the same two commands work on a laptop and on a cluster login node.

This installs the command-line tools `corenet-validate`, `corenet-train`, `corenet-evaluate` and
`corenet-export` (each has `--help`), and the Python package `corenet`.

## Dependencies

| package | minimum version | used for |
|---|---|---|
| Python | 3.9 | |
| PyTorch | 2.0 | the network |
| NumPy | 1.21 | |
| h5py | 3.0 | input and prediction files |
| PyYAML | 5.4 | configuration files |
| scikit-learn | 1.0 | AUC / ROC in `corenet-evaluate` |
| onnx, onnxruntime | 1.14, 1.15 | `corenet-export` (optional extra `export`) |
| pytest | 7 | tests (optional extra `test`) |

The release was tested with Python 3.13, PyTorch 2.13, NumPy 2.5, h5py 3.16, PyYAML 6.0,
scikit-learn 1.9, onnx 1.22 and onnxruntime 1.28; the minimum versions above are the ones the code
requires by construction but have not all been tested.

## Quick start

The repository ships two small example files (20,000 events each) so that everything below runs on a
laptop CPU in a few minutes. From the repository root:

```bash
# 1. check the configuration and the data file against each other
corenet-validate --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5

# 2. train with fold 0 held out and fold 1 used for validation (2 epochs for the demo; the recipe has 20)
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5 \
    --fold 0 --exclude-folds 1 --epochs 2 --device cpu --output-dir runs/hh4b

# 3. evaluate on the held-out fold: AUCs, ROC curve, class and pairing probabilities
corenet-evaluate --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output runs/hh4b/eval_fold0.h5

# 4. export to TorchScript and ONNX, and check the exported models against PyTorch
corenet-export --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output-dir runs/hh4b/export
```

The same four commands with `bbww` in place of `hh4b` run the HH → bbWW example. The examples are
described in [examples/hh4b/README.md](examples/hh4b/README.md) and
[examples/bbww/README.md](examples/bbww/README.md).
For a step-by-step guide see [docs/walkthrough_hh4b.md](docs/walkthrough_hh4b.md).

From Python:

```python
from corenet import CoReNet, load_config
from corenet.data import H5Dataset

spec = load_config("examples/bbww/bbww.yaml")
model = CoReNet(spec, device="cpu")
ds = H5Dataset("examples/data/bbww_small.h5", spec)
out = model(ds[:512]["inputs"])      # {"logits", "views", "heads", "event"}
```

## Defining an event decay chain

A configuration has up to nine sections; `objects`, `hypotheses`, `outputs` and `combine` are required.
The complete reference is [docs/configuration.md](docs/configuration.md); the forward pass is described
in [docs/architecture.md](docs/architecture.md) and the input file layout in
[docs/data_format.md](docs/data_format.md).

**Objects and roles.** A *collection* is read from the HDF5 file (`features`, `max` slots); a *role*
takes a fixed number of members of a collection with a selector: `btag` (highest b tag, then pT),
`pt`, or `rest` (members not taken by earlier roles). Collections with a single object (a lepton, the
missing momentum) are used directly.

```yaml
objects:
  jets:   {features: [pt, eta, phi, mass, btag], max: 12}
  b:      jets btag 2
  q:      jets rest 2
  lepton: {features: [pt, eta, phi, mass, isE, isM]}
  met:    {features: [pt, phi]}
event: [nSelJets, HT]            # event-level scalars, added to every object embedding
```

`features` lists every column the loader reads from the file for a collection, not the network inputs:
`pt`, `eta`, `phi` and `mass` enter the network through the four-vector embedding, while `btag` is a selection
tag, consumed by the `btag` selector only and never fed to the network. The file does not pre-select the
candidate jets: it stores all jets of the event (12 slots in the example files, pT-ordered, padding flagged in `MASK`), and
the roles choose among them when the file is loaded, so one file serves any role definition.

**Hypotheses.** Each hypothesis is a decay tree over roles, single objects and named intermediate
nodes. Interchangeable children (`[b, b]`, or two structurally identical nodes) and exclusivity (an
object is used once per candidate) are inferred, and all admissible candidates are enumerated: for
`HH: {H1: [b, b], H2: [b, b]}` with four b jets, 6 dijets and 3 pairings. A node may be reused by name
in another hypothesis (one set of weights).

```yaml
hypotheses:
  HH:
    H_bb: [b, b]
    H_WW:
      W_had: [q, q]
      W_lep: lepton + met        # recipe: leptonic W with the W-mass constraint on the neutrino pz
  tt:
    t_had: [b, W_had]            # W_had and W_lep are the HH nodes, reused
    t_lep: [b, W_lep]
```

**Recipes.** `A + B` replaces the plain four-vector sum by a registered reconstruction. The one
registered recipe is `lepton + met`, which also adds the transverse mass and an off-shell score as node
features.

**Attention edges and context.** `attention: [{query: keys}]` lets the candidates of one node attend to
those of another, with the pair (log M, ΔR, Δφ) as bias and a mask for candidates sharing an object;
an edge that leaves one admissible key per query becomes a pairwise merge. `context: <role>` inside a
hypothesis lets its first-level candidates attend to all objects of the hypothesis plus that role.

```yaml
attention:
  - W_lep: W_had
  - t_had: t_lep                 # a top never pairs with the top that uses the same b jet
```

**Relations.** `relations: [[A, B], ...]` adds the (log M, ΔR, Δφ) of two single-candidate nodes or
objects to the input of the main head.

**Outputs, combine and auxiliary heads.** Each output is a class (label 1, 2, ... in file order; 0 is
"rest") and a head. The first is the main head, a softmax over all classes computed from the selected
candidates of the `combine` hypotheses (and the relations); every further output also gets its own
binary auxiliary head on its hypothesis' selected candidate (or, with `~`, on the event pixel).

```yaml
relations:
  - [lepton, met]
  - [H_bb, met]
outputs:
  signal: HH                     # main score: P(signal)
  ttbar:  tt                     # class 2, plus an auxiliary tt head
combine: [HH, tt]
```

The two complete examples, [examples/hh4b/hh4b.yaml](examples/hh4b/hh4b.yaml) (one hypothesis,
three pairings, context attention) and [examples/bbww/bbww.yaml](examples/bbww/bbww.yaml) (two
hypotheses sharing nodes, a recipe, merges and relations), are explained line by line in their READMEs.

## Validating a configuration and a data file

Recommended first step before any training:

```bash
corenet-validate --config my_config.yaml                       # the configuration only
corenet-validate --config my_config.yaml --input my_data.h5    # and the file against it
```

The configuration is checked with the same parser the trainer uses (required sections, unknown keys,
types and ranges, every referenced name declared, the `trainer:` recipe), and the network is built and
summarised (candidates and views per hypothesis, attention vs merge edges, parameter count). With
`--input`, every dataset the loader will read is looked up with the loader's own rules and its dtype
and shape are checked (per-collection features and slots, `MASK`, event scalars, labels in range,
finite weights, folds); then the actual loader reads the file and the network runs a forward pass.
Each finding is reported as `OK`, `WARN` or `FAIL` with the dataset or key concerned, for example

```
  [FAIL] collection `lepton` (used by `lepton`) expects dataset INPUTS/Lepton/isE: not found (INPUTS/Lepton has ['eta', 'mass', 'phi', 'pt'])
```

and the exit status is 1 if anything failed. `--json report.json` writes the report as JSON.

## Training

```bash
corenet-train --config examples/bbww/bbww.yaml --input examples/data/bbww_all.h5 \
    --fold 0 --exclude-folds 1 --output-dir runs/bbww --device cuda
```

The default recipe: Adam with learning rate 0.01 for 20 epochs; batch size 1024 doubled at epochs 1, 3,
6 and 10 (with 64, 16, 4, 1, 0 ghost batches); per-class weight normalisation; the input normalisation
is fixed from the training set before epoch 0; from epoch 15 only the output heads are trained and the
learning rate is multiplied by 0.25 after each of epochs 15 to 24. Every item can be changed with a
flag (`--epochs`, `--lr`, `--lr-milestones`, `--lr-gamma`, `--batch-size-init`, `--freeze-epoch`,
`--no-class-norm`, `--seed`) or in a `trainer:` section of the configuration. `--fold -1` trains one
model per fold. Details, and the output files (checkpoint, history, predictions on the held-out fold),
in [docs/training.md](docs/training.md).

## Evaluation

```bash
corenet-evaluate --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_all.h5 \
    --output runs/bbww/eval_fold0.h5
```

Evaluates the fold held out in training (`--fold k` or `--fold -1` for others), prints the weighted ROC
AUC of the main output (class 1 against the rest), the one-vs-rest AUC of every class and the AUC of
every auxiliary head, and writes

* `eval_fold0.h5`: `class_probabilities` (N, n_classes), `views/<hypothesis>` (the per-hypothesis
  candidate probabilities, e.g. the three HH → 4b pairings), `heads/<output>`, the ROC curve
  `roc/{fpr,tpr,threshold}` and the event labels, weights and folds;
* `eval_fold0.h5.json`: the metrics.

See [docs/evaluation.md](docs/evaluation.md).

## Model export

```bash
corenet-export --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_small.h5 \
    --output-dir runs/bbww/export
```

writes a TorchScript file (`.pt`, `torch.jit.trace`), an ONNX file (`.onnx`, dynamic batch size) and a
JSON description of the inputs and outputs, then reloads both exported models (with `torch.jit.load`
and onnxruntime) and checks that they reproduce the PyTorch outputs on events of `--input` to within
`--atol` (1e-4); TorchScript is bit-identical to PyTorch and ONNX agrees to float32 rounding (≤ 2.3e-5
on the example models). The exported graph takes the role tensors and
event scalars as produced by `corenet.data.H5Dataset` (the role selection itself is done by the
loader, not in the graph) and returns the class probabilities, the per-hypothesis candidate
probabilities and the auxiliary-head probabilities. Inputs, outputs and limitations:
[docs/export.md](docs/export.md).

## Hyperparameters

The network options live in the `training:` section of the configuration (embedding dimension
`features`, attention heads, attention gates, key/value projections, phase symmetry, η reflection,
auxiliary-head loss weights); the training recipe in the `trainer:` section or on the command line.
[docs/hyperparameters.md](docs/hyperparameters.md) lists every option with its default and
constraints, and [examples/hh4b/hh4b_hyperparameters.yaml](examples/hh4b/hh4b_hyperparameters.yaml) /
[examples/bbww/bbww_hyperparameters.yaml](examples/bbww/bbww_hyperparameters.yaml) are commented
configurations with every option written out at its default value. The network has no dropout.

## Example data

| file | events | size | in the repository |
|---|---|---|---|
| `examples/data/hh4b_small.h5` | 20,000 (18,403 multijet, 1,597 HH) | 7.8 MB | yes |
| `examples/data/bbww_small.h5` | 20,000 (1,198 HH, 18,802 tt̄) | 9.8 MB | yes |
| `examples/data/hh4b_all.h5` | 454,261 (417,978 multijet, 36,283 HH) | 176 MB | no, distributed separately: [DOI / URL to be added] |
| `examples/data/bbww_all.h5` | 292,285 (17,511 HH, 274,774 tt̄) | 142 MB | no, distributed separately: [DOI / URL to be added] |

The full files exceed GitHub's 100 MB file limit; download them into `examples/data/` to run the
full-statistics commands. The small files are stratified random subsamples of the full ones with the
same schema, written with `python scripts/make_small_h5.py <full> <small> --events 20000`. All files
carry 5 folds (`EVENT/fold`). Layout: [docs/data_format.md](docs/data_format.md).

## Tests

```bash
pytest tests
```

The unit tests cover the configuration parser, the candidate enumeration, the network (structure,
masking, finiteness, gradients), the loader on the example files, training and evaluation on synthetic
files, export, and the validator. `tests/reference/` holds a parity test against the research code
this package was extracted from; it is skipped unless `CORENET_REFERENCE_REPO` points to that code.

## Citation and licence

Citation information: [CITATION.cff](CITATION.cff) (to be completed). Released under the MIT licence
([LICENSE](LICENSE)).
