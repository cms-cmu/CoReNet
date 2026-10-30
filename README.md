# CoReNet: Combinatorial Residual Network
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23577720.svg)](https://doi.org/10.5281/zenodo.23577720)

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
access. The environment is built for Linux (x86-64) with an NVIDIA GPU (CUDA 12).

This installs the command-line tools `corenet-validate`, `corenet-train`, `corenet-evaluate`,
`corenet-export` and `corenet-report` (each has `--help`), and the Python package `corenet`. The unit tests run with
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
| matplotlib | 3.5 | figures of `corenet-report` |
| onnx, onnxruntime | 1.14, 1.15 | `corenet-export` (optional extra `export`) |
| pytest | 7 | tests (optional extra `test`) |

The release was tested with Python 3.13, PyTorch 2.13, NumPy 2.5, h5py 3.16, PyYAML 6.0,
scikit-learn 1.9, onnx 1.23 and onnxruntime 1.30; the minimum versions above are the ones the code
requires by construction but have not all been tested.

## Example

The repository ships three small example files of 20,000 events each (`hh4b_small.h5`, `bbww_small.h5`,
`tth_small.h5` in `examples/data/`), so that everything below runs in under a minute. The HH → bb̄bb̄
example, from the repository root:

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

# 5. one HTML page with the training curves, ROC curves, pairings and input checks of the run
corenet-report --run-dir runs/hh4b --input examples/data/hh4b_small.h5 --output runs/hh4b/report.html
```

The same five commands with `bbww` or `tth` in place of `hh4b` run the HH → bb̄WW and ttH examples.
[docs/HH4b.md](docs/HH4b.md) walks through the HH → bb̄bb̄ example step by step;
[docs/BBWW.md](docs/BBWW.md) does the same for HH → bb̄WW against tt̄, and
[docs/TTH.md](docs/TTH.md) for ttH (H → bb̄) against semileptonic tt̄.

The decay chain of the HH → bb̄bb̄ example is

```yaml
objects:
  jets:  {features: [pt, eta, phi, mass], max: 12}         # INPUTS/Source (alias of "jets") in the H5 file
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

The optional `labels:` section names the labels to train on, in label order (e.g.
`labels: [multijet, HH, ttbar]`: label 0 = multijet, 1 = HH, 2 = ttbar). Without it the classes are
"rest" (label 0) plus the outputs in order; with it every output name must be one of the labels and the
main score is the class of the first output ([EventInfo.md](docs/EventInfo.md)).

`jets` is the jet collection: `max: 12` is the number of jet slots read per event, and the input file must
have at least that many. The listed `features` are the inputs of the network: these columns of the input
file, and only these, are fed to it, for the `b` jets and for the `other` jets alike. \
`jets btag 4` selects the four jets with the highest `btag` (alternatively `jets pt 4` would select the four
leading jets): any column of the jets in the file can be used this way, and `jets mass 2 lowest` would take the
two jets with the lowest value instead. The selector reads its column from the file for this selection only; it
need not be a feature, and a feature named `btag` would be one more network input. The input
file holds all jets of the event, and the b jets are chosen when it is loaded. For more details on specifying the event decay, check [docs/EventInfo.md](docs/EventInfo.md), and for the input file itself, check [docs/Dataset.md](docs/Dataset.md).

The full-statistics files (454,261 HH → bb̄bb̄ events, 292,285 HH → bb̄WW events, 423,753 ttH and tt̄
events) are distributed separately on Zenodo ([doi:10.5281/zenodo.23567532](https://doi.org/10.5281/zenodo.23567532)); download them into `examples/data/` as `hh4b_all.h5`,
`bbww_all.h5` and `tth_all.h5`.

## Usage

1. Describe the final state in a configuration file ([docs/EventInfo.md](docs/EventInfo.md)).
2. Prepare the input file ([docs/Dataset.md](docs/Dataset.md)).
3. Choose the network and training settings ([docs/Options.md](docs/Options.md)).
4. Before training, check that the configuration and the input file are correct and consistent with each other:

   ```bash
   corenet-validate --config my_config.yaml --input my_data.h5
   ```

   The report ends with `valid` or `NOT VALID`; `--json report.json` also writes it to a file.

The validator only reads the file, so it can run on a machine without a GPU: pixi checks the environment's
CUDA requirement against the host driver, and `CONDA_OVERRIDE_CUDA=12.0 pixi run corenet-validate ...`
lifts that check.

### Training

```bash
corenet-train --config examples/bbww/bbww.yaml --input examples/data/bbww_all.h5 \
    --fold 0 --exclude-folds 1 --output-dir runs/bbww
```

* `--fold 0`: hold out fold 0 for testing. With `--fold -1` (the default), train one model per fold in turn.
* `--exclude-folds 1`: keep fold 1 out of the training and use it for validation. `--exclude-folds next`
  uses the fold `(k + 1) % n_folds` of the held-out fold `k`.
* `--n-folds 5`: take the fold of an event as `squares(event) % 5`, a hash of the event number in
  `EVENT/event`, instead of the `EVENT/fold` column ([docs/Dataset.md](docs/Dataset.md)); `--fold-seed`
  (default 0) selects the hash key. The rule, the number of folds and the seed are stored in the
  checkpoint and reused by `corenet-evaluate`. With 5 folds and seed 0 the rule equals the
  `EVENT/fold` column of the example files.
* `--early-stop-patience 5`: stop when the validation AUC has not improved for 5 epochs and keep the best epoch.
* `--freeze-epoch 15 --finetune-layers head_main. head_reduce. aux_heads. --finetune-lr 1.0e-3 --finetune-batch-size 4096`:
  the fine-tuning stage; from epoch 15 on only the parameters whose names start with the given prefixes
  (default: the output layers; `none` freezes nothing) are trained, at this learning rate and batch size
  (default: the schedule continues).
* `--epochs`, `--lr`, `--seed`, ...: override the training settings of the configuration file.
  See [docs/Options.md](docs/Options.md) or `corenet-train --help`.

The folds come from the input file: the `EVENT/fold` column, or `squares(event) % K` with
`--n-folds K` ([docs/Dataset.md](docs/Dataset.md)). The example files have 5 folds in `EVENT/fold`.

Per fold `k`, the output directory receives:

| file | content |
|---|---|
| `model_fold{k}.pkl` | checkpoint; evaluation and export need no configuration file |
| `history_fold{k}.json` | training settings and per-epoch history |
| `predictions_fold{k}.h5` | predictions on the held-out fold |

### Evaluation

```bash
corenet-evaluate --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_all.h5 \
    --output runs/bbww/eval_fold0.h5
```

* `--fold k`: evaluate fold `k` instead of the fold held out in training. `--fold -1` evaluates every event of the file.
* `--unweighted`: ignore the event weights in the metrics.
* `--device cuda`: evaluate on the GPU (default: CPU).

Output:

| file | content |
|---|---|
| `eval_fold0.h5.json` | the metrics |
| `eval_fold0.h5` | the predictions, as below |
| `report.html` | every result of the run on one page, written by `corenet-report` ([Report](#report) below) |

| dataset in `eval_fold0.h5` | content |
|---|---|
| `class_probabilities` | one column per class; column 1 is the signal score |
| `views/HH`, `views/tt` | probability of every assignment of the jets to the particles of that hypothesis: for HH → bb̄bb̄ the three ways of pairing the four b jets into two Higgs bosons, for HH → bb̄WW the two ways of assigning the b jets to the top quarks of tt̄ |
| `heads/ttbar` | the tt̄ score (one dataset per auxiliary output) |
| `roc/{fpr,tpr,threshold}` | the ROC curve of the signal score |
| `EVENT/{index,event,label,weight,fold}` | row index in the input file, event number, label, weight and fold of every evaluated event |

### Exporting

```bash
corenet-export --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_small.h5 \
    --output-dir runs/bbww/export
```

* `--formats torchscript` or `--formats onnx`: export only one of the two formats.
* `--opset 17`: ONNX opset version.
* `--n-events 1024`, `--atol 1e-4`: number of events of `--input` on which the exported models are checked against PyTorch, and the tolerance.

Output, in `--output-dir`:

| file | content |
|---|---|
| `model_fold0.pt` | TorchScript model |
| `model_fold0.onnx` | ONNX model, dynamic batch size |
| `model_fold0.json` | description of the inputs and outputs |

Note that the exported model starts after the jets have been assigned: it takes the jets already
assigned to the b quarks, the light quarks and so on. An application outside this package has to
assign the jets the same way (for `b: jets btag 4`, the four jets most likely to be b jets).

#### Inputs

| input | shape | content |
|---|---|---|
| one per set of assigned jets, e.g. `b`, `other` | (N, features + 1, n) | the `features` of the `n` jets, in configuration order, plus a presence flag |
| one per single object, e.g. `lepton`, `met` | (N, features + 1, 1) | the same for a single object |
| `event` | (N, n_event) | the event quantities (`nSelJets`, `HT`, ...) |

`features` is the number of features listed for the collection in the configuration; a column read only
by a selector (`btag`) is not an input. For HH → bb̄bb̄ the inputs are `b` (N, 5, 4), `other` (N, 5, 8)
and `event` (N, 2); for HH → bb̄WW `b` (N, 5, 2), `q` (N, 5, 2), `lepton` (N, 7, 1), `met` (N, 3, 1) and
`event` (N, 2). The exported model takes the same inputs as the training file, without any preprocessing step.

#### Outputs

| output | shape | content |
|---|---|---|
| `class_probabilities` | (N, n_classes) | probability of every class, columns `["rest", <outputs>...]` |
| `views_<hypothesis>` | (N, V) | probability of every pairing of a hypothesis with several pairings |
| `head_<output>` | (N,) | the extra binary score of every output after the first |

### Report

```bash
corenet-report --run-dir runs/bbww --input examples/data/bbww_all.h5 --output runs/bbww/report.html
```

* `--folds 0 1`: report these folds only (default: every fold found in the run directory).
* `--max-events 500000`: events of the input file used for the input and weight distributions (random subsample above it; `0` = all).
* `--config my_config.yaml`: event description to use instead of the one stored in the checkpoint.

To see the training and validation folds in the ROC and score figures (overtraining check), first write the
predictions of every event with `corenet-evaluate ... --output runs/bbww/eval_fold0.h5 --fold -1`.

Output:

| file | content |
|---|---|
| `report.html` | self-contained page: run summary, training curves, ROC / AUC / score distributions, view probabilities and pairing accuracy against `TARGETS`, the inputs as the network embeds them, the raw inputs and the event weights ([docs/Report.md](docs/Report.md)) |

Open `report.html` in any browser; from a cluster, copy it to your machine first (`scp <host>:runs/bbww/report.html .`).

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
The software is archived on Zenodo with the concept DOI
[10.5281/zenodo.23577720](https://doi.org/10.5281/zenodo.23577720) (version 1.0.0:
10.5281/zenodo.23577721); [CITATION.cff](CITATION.cff) holds the citation.
Released under the MIT licence ([LICENSE](LICENSE)).
