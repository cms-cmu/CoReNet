# Training

```bash
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_all.h5 \
    --fold 0 --exclude-folds 1 --output-dir runs/hh4b --device cuda
```

`--fold k` trains one model with fold `k` held out (`--fold -1`, the default, trains one model per fold
in turn). The events of the folds given by `--exclude-folds` are left out of training and used for the
per-epoch validation loss and AUC in the history file. The model is trained on everything else.
`--device cuda` falls back to the CPU when no GPU is available; the whole file is kept in (GPU) memory.

## The training recipe

The default recipe, as implemented in `src/corenet/train.py`:

| step | what happens |
|---|---|
| seed | `torch` and `numpy` are seeded with `seed + k` for fold `k` |
| weights | with `class_norm` (default), the event weights of every class are scaled to sum to 1 over the training events; in the loss, weights are clamped at ≥ 0 |
| input normalisation | before epoch 0 the mean and standard deviation of every input embedding (objects, nodes, event scalars, pair features) are accumulated over all training events and frozen |
| optimiser | Adam over all parameters, learning rate `lr` = 0.01 |
| learning rate | MultiStepLR: the learning rate is multiplied by `lr_gamma` = 0.25 after each epoch listed in `lr_milestones` = 15, 16, ..., 24, i.e. epoch `e` runs at `lr × 0.25^(number of milestones ≤ e)`: 0.01 up to epoch 14, 0.0025 at epoch 15, ... |
| batch size | `batch_size_init` = 1024 at epoch 0, doubled at epochs 1, 3, 6 and 10 (1024, 2048, 4096, 8192, 16384); batches are shuffled every epoch and the last incomplete batch is dropped |
| ghost batches | 64, 16, 4, 1, 0 ghost batches in these five stages (never fewer than 8 events per ghost batch); with 0, the batch-norm layers use their running statistics |
| freeze | at epoch `freeze_epoch` = 15 every parameter except the output heads (`head_main`, `head_reduce`, `aux_heads`) is frozen |
| epochs | `epochs` = 20; the model after the last epoch is saved (no early stopping, no best-epoch selection) |
| loss | weighted cross entropy of the main head over the classes, plus `head_weights[o]` (default 1) × weighted binary cross entropy of every auxiliary head `o` |

The batch-size doubling epochs and the ghost-batch counts are fixed in `batch_schedule()`; everything
else in the table can be changed without editing code.

## Changing the recipe

Each recipe option can be set in three places; the first one found wins:

1. a command-line flag: `--epochs`, `--freeze-epoch`, `--lr`, `--lr-milestones`, `--lr-gamma`,
   `--batch-size-init`, `--class-norm / --no-class-norm`, `--seed`, `--rest-order`;
2. the `trainer:` section of the configuration file;
3. the built-in default above.

```yaml
trainer:            # in the configuration file; see examples/*/..._hyperparameters.yaml
  epochs: 30
  lr: 0.005
  lr_milestones: [20, 22, 24, 26, 28]
  freeze_epoch: 25
```

Write floating-point numbers with a decimal point (`0.005` or `5.0e-3`): YAML reads `5e-3` as a
string, which the validation rejects. The recipe used is stored in the checkpoint metadata and in the
history file. The network options (embedding dimension etc.) are in the `training:` section; see
[hyperparameters.md](hyperparameters.md).

## Outputs

For each trained fold `k`, in `--output-dir`:

| file | content |
|---|---|
| `model_fold{k}.pkl` | checkpoint: `{"model": state dict, "config": configuration mapping, "config_path", "classes", "metadata": {fold, epoch, exclude_folds, input, recipe}}`; self-contained, so evaluation and export need no configuration file |
| `history_fold{k}.json` | per epoch: batch size, ghost batches, learning rate, training loss, and (with `--exclude-folds`) validation loss and AUC |
| `predictions_fold{k}.h5` | predictions on the held-out fold, in the layout of [evaluation.md](evaluation.md) |

`--save-every-epoch` also writes `model_fold{k}_epoch{e}.pkl` after every epoch.
