# Hyperparameter reference

Complete, commented examples with every option at its default value:
[examples/hh4b/hh4b_hyperparameters.yaml](../examples/hh4b/hh4b_hyperparameters.yaml) and
[examples/bbww/bbww_hyperparameters.yaml](../examples/bbww/bbww_hyperparameters.yaml).
The network has no dropout and no weight decay.

## Network options: the `training:` section of the configuration

These change the architecture; a checkpoint only loads into a network built with the same values.

| key | type | default | meaning |
|---|---|---|---|
| `features` | int ≥ 1 | 8 | embedding dimension `d` of every pixel (object, node, candidate, event). Must be divisible by `attention_heads`, even when `phase_symmetric`, and ≥ 2 × `attention_heads` when `kv_projection` |
| `attention_heads` | int ≥ 1 | 2 | heads of every attention block (context attention and `attention` edges built as attention; merges have no heads) |
| `attention_gate` | bool | true | pair features enter the attention scores and values through small two-layer MLP gates instead of a plain sum |
| `kv_projection` | bool | true | bottleneck query / key projections (d → d/2) and a value projection in every attention block |
| `phase_symmetric` | bool | true | the embedding and reinforce convolutions compute d/2 filters and output (x, −x) plus a bias |
| `reflect_eta` | bool | true | multiply every η by the sign of the η of the first member of the first leaf (a symmetry of the detector) |
| `head_weights` | mapping output → number ≥ 0 | 1.0 each | weight of each auxiliary head in the loss; keys must be outputs after the first |

Parameter counts of the examples: HH → 4b 1,508 and HH → bbWW 3,840 at `features: 8`. In the bbWW
example `attention_heads`, `attention_gate` and `kv_projection` have no effect: it has no `context`, and
both `attention` edges have one admissible key per query and are built as merges.

## Training recipe: the `trainer:` section or `corenet-train` flags

A command-line flag overrides the `trainer:` section, which overrides the default. See
[training.md](training.md) for how they act.

| key | flag | type | default | meaning |
|---|---|---|---|---|
| `epochs` | `--epochs` | int ≥ 1 | 20 | number of epochs |
| `lr` | `--lr` | number > 0 | 0.01 | Adam learning rate |
| `lr_milestones` | `--lr-milestones` | list of int | 15, 16, ..., 24 | epochs after which the learning rate is multiplied by `lr_gamma` |
| `lr_gamma` | `--lr-gamma` | number > 0 | 0.25 | learning-rate decay factor |
| `batch_size_init` | `--batch-size-init` | int ≥ 1 | 1024 | epoch-0 batch size; doubled at epochs 1, 3, 6 and 10 (fixed) |
| `freeze_epoch` | `--freeze-epoch` | int ≥ 0 | 15 | from this epoch on only the output heads are trained (a value ≥ `epochs` disables the freeze) |
| `class_norm` | `--class-norm` / `--no-class-norm` | bool | true | normalise the event weights of every class to 1 over the training set |
| `seed` | `--seed` | int | 42 | random seed; fold k uses `seed + k` |
| `rest_order` | `--rest-order` | `pt` / `selector` | `pt` | order of the members picked by a `rest` role (docs/configuration.md §1.2) |

Flags without a `trainer:` key: `--config`, `--input`, `--output-dir`, `--fold`, `--exclude-folds`,
`--save-every-epoch`, `--device`.

## Fixed in code

| what | value | where |
|---|---|---|
| batch-size doubling epochs, ghost batches per stage | 1, 3, 6, 10; 64, 16, 4, 1, 0 | `corenet.train.batch_schedule` |
| minimum ghost-batch size | 8 events | `corenet.train.MIN_GHOST_BATCH_SIZE` |
| batch-norm running-average momentum | 0.9 | `GhostBatchNorm1d(eta=0.9)` |
| reinforce layers per node | 2 | `ResNetBlock(nLayers=2)` |
| W mass in the `lepton + met` recipe | 80.379 GeV | `corenet.models.fourvectors.W_MASS` |
| Adam β, ε | PyTorch defaults | `corenet.train.train_fold` |
