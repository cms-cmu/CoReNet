# Input data format

CoReNet reads one HDF5 file per sample set (all classes and all folds together). Only the datasets
below are read; any other content (for example generator-level `TARGETS/`) is ignored.

| dataset | shape | dtype | used for |
|---|---|---|---|
| `INPUTS/<group>/<feature>` | `(N, slots)` for a collection with `max > 1`, `(N,)` for `max: 1` | numeric | the features listed in the configuration |
| `INPUTS/<group>/MASK` | `(N, slots)` | bool (True = present) | presence of each slot; optional (all present when absent) |
| `EVENT/<name>` | `(N,)` | numeric | event scalars other than the computed `nSelJets` / `HT` |
| `EVENT/label` or, when absent, `CLASSIFICATIONS/EVENT/signal` | `(N,)` | integer | class label: 0 = rest, 1.. = the outputs in configuration order |
| `EVENT/weight` | `(N,)` | float | event weight (1 when absent) |
| `EVENT/fold` | `(N,)` | integer | cross-validation fold of the event (needed for training) |
| `EVENT/event` | `(N,)` | integer | event identifier copied to the prediction files (0..N-1 when absent) |

**Groups.** A collection named `x` is read from `INPUTS/x`, matched case-insensitively, or from one of
the aliases `jets`/`jet` → `Jets`, `Source`; `lepton` → `Lepton`, `Leptons`; `met` → `Met`, `MET`.

**Slots.** A collection with `max: m` needs at least `m` slots in the file. Padded slots must be
marked in `MASK`; their feature values are ignored.

**Weights.** The training recipe normalises the weights of every class to sum to 1 over the training
events (`class_norm`, on by default), so only the relative weights within a class matter for
training. Negative weights count as 0 in the loss and in the metrics.

**Folds.** `corenet-train --fold k` holds out the events with `EVENT/fold == k` for testing,
optionally uses the folds given by `--exclude-folds` for per-epoch validation, and trains on the rest.

Check a file against a configuration with

```bash
corenet-validate --config examples/bbww/bbww.yaml --input examples/data/bbww_small.h5
```

## The example files

`examples/data/hh4b_*.h5` and `examples/data/bbww_*.h5` follow this layout:

* `hh4b`: jets in `INPUTS/Source` (12 slots, `pt eta phi mass btag MASK`, `btag` is 0/1), labels in
  `CLASSIFICATIONS/EVENT/signal` (0 multijet, 1 HH), `EVENT/weight`, `EVENT/fold` (5 folds), `EVENT/event`;
* `bbww`: jets in `INPUTS/Jets` (12 slots), `INPUTS/Lepton` (`pt eta phi mass isE isM`), `INPUTS/Met`
  (`pt phi`), labels in `EVENT/label` (1 HH, 2 tt̄; no class 0), `EVENT/weight`, `EVENT/fold`, `EVENT/event`.

The `*_small.h5` files are stratified random subsamples (20,000 events, class fractions kept) of the
full `*_all.h5` files written by `scripts/make_small_h5.py`; the full files are distributed separately
(see the README).
