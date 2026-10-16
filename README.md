# CoReNet: Combinatorial Residual Network

[![CI](https://github.com/cms-cmu/CoReNet/actions/workflows/ci.yml/badge.svg)](https://github.com/cms-cmu/CoReNet/actions/workflows/ci.yml)

A library for training and evaluating CoReNet classifiers on collider events. Originally developed for
HH → bb̄bb̄, it supports arbitrary final states described by their decay chain.

## Installation

The repository ships a [pixi](https://pixi.sh) project (`pixi.toml` and the lock file `pixi.lock`), which
installs the same conda-forge builds of Python, PyTorch (with its own CUDA and cuDNN libraries) and every
other dependency on any Linux x86-64 machine, and installs the package itself in editable mode:

```bash
git clone https://github.com/cms-cmu/CoReNet.git
cd CoReNet
pixi install                     # CUDA build of PyTorch (any NVIDIA GPU supported by CUDA 12)
pixi shell                       # activate; or prefix every command with `pixi run`
```

pixi itself is a single static binary (`curl -fsSL https://pixi.sh/install.sh | sh`) that needs no root
access, so the same commands work on a laptop and on a cluster front end.

This installs the command-line tools `corenet-validate`, `corenet-train`, `corenet-evaluate` and
`corenet-export` (each has `--help`), and the Python package `corenet`. The unit tests run with
`pytest tests`.

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
scikit-learn 1.9, onnx 1.23 and onnxruntime 1.30; the minimum versions above are the ones the code
requires by construction but have not all been tested.

## Example

The repository ships two small example files of 20,000 events each, so that everything below runs in
under a minute. The HH → bb̄bb̄ example, from the repository root:

```bash
# 1. check the configuration and the data file against each other
corenet-validate --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5

# 2. train with fold 0 held out and fold 1 used for validation (2 epochs for the demo; the default is 20)
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5 \
    --fold 0 --exclude-folds 1 --epochs 2 --device cuda --output-dir runs/hh4b

# 3. evaluate on the held-out fold
corenet-evaluate --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output runs/hh4b/eval_fold0.h5

# 4. export to TorchScript and ONNX, and check the exported models against PyTorch
corenet-export --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output-dir runs/hh4b/export
```

The same four commands with `bbww` in place of `hh4b` run the HH → bb̄WW example.
[docs/HH4b.md](docs/HH4b.md) walks through the HH → bb̄bb̄ example step by step;
[docs/BBWW.md](docs/BBWW.md) does the same for HH → bb̄WW against tt̄.

The decay chain of the HH → bb̄bb̄ example is

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
```

`jets` is the jet collection, with up to 12 jets per event; the listed `features` are the columns to
read from the input file. `btag` is only used to pick which jets are treated as b-jet candidates
(`b: jets btag 4`, the four jets most likely to be b jets); it is not an input to the network. The
file holds all jets of the event, and the b jets are chosen when it is loaded, so one file serves any
way of choosing them. [docs/EventInfo.md](docs/EventInfo.md) explains every
line of such a file and [docs/Dataset.md](docs/Dataset.md) the input file.

The full-statistics files (454,261 HH → bb̄bb̄ events, 292,285 HH → bb̄WW events) are distributed
separately ([DOI / URL to be added]); download them into `examples/data/` as `hh4b_all.h5` and
`bbww_all.h5`.

## Usage

Three steps: describe the final state ([docs/EventInfo.md](docs/EventInfo.md)), prepare the input
file ([docs/Dataset.md](docs/Dataset.md)), and choose the settings ([docs/Options.md](docs/Options.md)).
Then check the two files against each other:

```bash
corenet-validate --config my_config.yaml --input my_data.h5
```

checks the event description against the input file; `--json report.json` also writes the report to a file.

### Training

```bash
corenet-train --config examples/bbww/bbww.yaml --input examples/data/bbww_all.h5 \
    --fold 0 --exclude-folds 1 --output-dir runs/bbww --device cuda
```

`--fold k` holds out the events of fold `k` for testing (`--fold -1`, the default, trains one model per
fold in turn); the folds in `--exclude-folds` are kept out of training and used for validation. For each fold
the output directory receives the checkpoint `model_fold{k}.pkl` (self-contained: the evaluation and
the export need no configuration file), the training history `history_fold{k}.json` and the
predictions on the held-out fold `predictions_fold{k}.h5`. Epochs, learning rate, batch size and the
other settings are changed with flags (`--epochs`, `--lr`, ...) or in the configuration file; see
[docs/Options.md](docs/Options.md) and `corenet-train --help`.

### Evaluation

```bash
corenet-evaluate --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_all.h5 \
    --output runs/bbww/eval_fold0.h5
```

evaluates the fold that was held out in training (`--fold k` for another fold, `--fold -1` for every
event of the file) and writes the metrics to `eval_fold0.h5.json` and the predictions to `eval_fold0.h5`:
`class_probabilities` (one column per class, column 1 is the signal score), `views/<hypothesis>` (the
pairing probabilities, for example the three pairings of the four b jets), `heads/<output>` (the extra
score of every further output), the ROC curve `roc/{fpr,tpr,threshold}`, and the event labels,
weights and folds. `--unweighted` ignores the event weights in the metrics.

### Exporting

```bash
corenet-export --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_small.h5 \
    --output-dir runs/bbww/export
```

writes a TorchScript file (`model_fold0.pt`), an ONNX file (`model_fold0.onnx`, dynamic batch size)
and a JSON description of the inputs and outputs, then reloads both and checks that they reproduce
the PyTorch outputs on events of `--input` to within `--atol` (1e-4). `--formats torchscript` or
`--formats onnx` exports only one of the two; `--opset` sets the ONNX opset (17).

Note that the exported model starts after the jets have been assigned: it takes the jets already
assigned to the b quarks, the light quarks and so on. An application outside this package has to
assign the jets the same way (for `b: jets btag 4`, the four jets most likely to be b jets).

#### Inputs

| input | shape | content |
|---|---|---|
| one per set of assigned jets, e.g. `b`, `other` | (N, features + 1, n) | the file's features of the `n` jets, in configuration order, plus a presence flag |
| one per single object, e.g. `lepton`, `met` | (N, features + 1, 1) | the same for a single object |
| `event` | (N, n_event) | the event quantities (`nSelJets`, `HT`, ...) |

For HH → bb̄bb̄ the inputs are `b` (N, 6, 4), `other` (N, 6, 8) and `event` (N, 2); for HH → bb̄WW
`b` (N, 6, 2), `q` (N, 6, 2), `lepton` (N, 7, 1), `met` (N, 3, 1) and `event` (N, 2). The
exported model takes the same inputs as the training file, without any preprocessing step.

#### Outputs

| output | shape | content |
|---|---|---|
| `class_probabilities` | (N, n_classes) | probability of every class, columns `["rest", <outputs>...]` |
| `views_<hypothesis>` | (N, V) | probability of every pairing of a hypothesis with several pairings |
| `head_<output>` | (N,) | the extra binary score of every output after the first |

```python
import onnxruntime as ort
from corenet import load_config
from corenet.data import H5Dataset

ds = H5Dataset("examples/data/hh4b_small.h5", load_config("examples/hh4b/hh4b.yaml"))
sess = ort.InferenceSession("runs/hh4b/export/model_fold0.onnx")
feeds = {i.name: ds.inputs[i.name][:100].numpy() for i in sess.get_inputs()}
class_probabilities, views_HH = sess.run(None, feeds)
```

## Contributors

[to be added]

## Citation

If you use this software for a publication, please cite it as given in [CITATION.cff](CITATION.cff)
(to be completed) and the paper describing the method: [paper reference / arXiv to be added].
Released under the MIT licence ([LICENSE](LICENSE)).
