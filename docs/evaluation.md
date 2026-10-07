# Evaluation

```bash
corenet-evaluate --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_all.h5 \
    --output runs/hh4b/eval_fold0.h5
```

By default the events of the fold that was held out when the checkpoint was trained are evaluated
(`--fold k` picks another fold, `--fold -1` every event of the file). The network is rebuilt from the
configuration stored in the checkpoint (`--config` overrides it) and run in eval mode.

## Metrics

Printed and written to `<output>.json` (here `runs/hh4b/eval_fold0.h5.json`):

| key | meaning |
|---|---|
| `auc.main` | ROC AUC of the main output: class 1 against all other classes, scored by P(class 1) |
| `auc.one_vs_rest` | for every class present in the events: class c against the others, scored by P(c) |
| `auc.heads` | for every auxiliary head: its class against the others, scored by the head's probability |
| `n_events`, `fold`, `weighted`, `classes`, `checkpoint`, `input` | bookkeeping |

AUCs use scikit-learn with the file's event weights clipped at ≥ 0 (`--unweighted` ignores the
weights). An AUC is `null` when one side of the comparison has no events.

## Prediction file

The same layout is written by `corenet-evaluate` and, for the held-out fold, by `corenet-train`:

| dataset | shape | content |
|---|---|---|
| `class_probabilities` | `(n, n_classes)` | softmax of the main head; columns follow the `classes` attribute, `["rest", <outputs>...]`, so column 1 is the main score |
| `views/<hypothesis>` | `(n, V)` | per-hypothesis view probabilities (softmax of the selector) for every hypothesis with more than one candidate; for HH → 4b the 3 pairings (01\|23, 02\|13, 03\|12), for HH → bbWW the 2 tt̄ assignments |
| `heads/<output>` | `(n,)` | sigmoid probability of every auxiliary head (e.g. `heads/ttbar`) |
| `roc/fpr`, `roc/tpr`, `roc/threshold` | | ROC curve of the main output (`corenet-evaluate` only) |
| `EVENT/index` | `(n,)` | row of the event in the input file |
| `EVENT/event`, `EVENT/label`, `EVENT/weight`, `EVENT/fold` | `(n,)` | copied from the input file |
| attributes | | `classes` (JSON), `main_hypothesis`, `input`, `checkpoint`, `n_parameters`, `fold`, (`auc_main`) |

Reading it:

```python
import h5py
with h5py.File("runs/hh4b/eval_fold0.h5") as f:
    p_signal = f["class_probabilities"][:, 1]
    pairing = f["views/HH"][...].argmax(1)          # most probable of the 3 pairings
    fpr, tpr = f["roc/fpr"][...], f["roc/tpr"][...]
```

From Python, `corenet.evaluate.predict(model, Batches(ds, device), indices)` returns the same three
objects (class probabilities, view probabilities, head probabilities) as arrays.
