# CoReNet Options

The settings live in the event description file, in two sections: `training:` for the network and
`trainer:` for the training. Every `trainer:` setting can also be given on the `corenet-train` command
line; a command-line flag wins over the file, which wins over the default. The settings used are
stored in the checkpoint and in the history file of every training.
[`examples/hh4b/hh4b_hyperparameters.yaml`](../examples/hh4b/hh4b_hyperparameters.yaml) and
[`examples/bbww/bbww_hyperparameters.yaml`](../examples/bbww/bbww_hyperparameters.yaml) are the two
examples with every setting written out; copy one and edit it.

## Network settings: `training:`

Changing one of these gives a different network: validate the file with `corenet-validate` and retrain.

| key | default | meaning |
|---|---|---|
| `features` | 8 | the width of the network, its main size knob; must be divisible by `attention_heads`; must be an even number when `phase_symmetric` is on; must be at least twice `attention_heads` when `kv_projection` is on |
| `attention_heads` | 2 | number of heads used when candidates are compared |
| `attention_gate` | true | gated comparison of candidates |
| `kv_projection` | true | reduced-size projections when candidates are compared |
| `phase_symmetric` | true | phase-symmetric layers |
| `reflect_eta` | true | use the forward–backward symmetry of the detector |
| `head_weights` | 1.0 each | weight in the loss of the extra score of every output after the first, e.g. `{ttbar: 1.0}` |

## Training settings: `trainer:` or `corenet-train` flags

| key | flag | default | meaning |
|---|---|---|---|
| `epochs` | `--epochs` | 20 | number of epochs; the model after the last epoch is kept |
| `lr` | `--lr` | 0.01 | learning rate |
| `lr_milestones` | `--lr-milestones` | 15, 16, ..., 24 | epochs after which the learning rate is multiplied by `lr_gamma` |
| `lr_gamma` | `--lr-gamma` | 0.25 | learning-rate decay factor |
| `batch_size_init` | `--batch-size-init` | 1024 | batch size of the first epoch; it is doubled at epochs 1, 3, 6 and 10 |
| `freeze_epoch` | `--freeze-epoch` | 15 | from this epoch on only the output layers are trained (a value ≥ `epochs` disables this) |
| `class_norm` | `--class-norm` / `--no-class-norm` | true | give every class the same total weight in training |
| `seed` | `--seed` | 42 | random seed; fold `k` uses `seed + k` |
| `rest_order` | `--rest-order` | `pt` | order of the jets taken by a `rest` line: `pt`, or `selector` to keep the order of the previous `btag`/`pt` line |

Write floating-point numbers with a decimal point (`0.005` or `5.0e-3`): YAML reads `5e-3` as a string,
which the validation rejects.

Other flags of `corenet-train`: `--config`, `--input`, `--output-dir`, `--fold` (held-out fold, `-1`
for every fold in turn), `--exclude-folds` (validation folds), `--save-every-epoch` (also save
`model_fold{k}_epoch{e}.pkl` after every epoch) and `--device`.

## Fixed in code

These choices are part of the training procedure and the network and have no setting:

* the optimizer is Adam, and the learning rate is multiplied by `lr_gamma` after each epoch listed in `lr_milestones`;
* the batch size doubles at epochs 1, 3, 6 and 10, and the batches are split into 64, 16, 4, 1 and 0 ghost batches over those stages;
* from `freeze_epoch` on, only the output layers keep training;
* the input file needs no standardization of its values;
* the event-level quantities `nSelJets` and `HT` are computed from the jets in the file.

## Choosing the jets

Which jets are the b jets is written in the event description rather than in a setting, but it is
the thing most often changed ([EventInfo.md](./EventInfo.md)):

| to change | example |
|---|---|
| how the b jets are chosen | `b: jets pt 4` (the four leading jets, b tag ignored) instead of `b: jets btag 4` |
| how many other jets the Higgs candidates see | `other: jets rest 4` instead of `other: jets rest 8` |

Any change in `objects` or `training:` is a new network: run `corenet-validate` and retrain.
