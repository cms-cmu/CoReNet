# Dataset Structure

CoReNet reads its training and testing data from HDF5 files. One file holds every class and every
cross-validation fold of a sample. The structure of the file follows your
[event description](./EventInfo.md): every collection named there is a group of the file.

## HDF5 Structure

Keys in `CAPITALCASE` must be exactly as shown; keys in `lower_case` are the names you choose in the
event description. Next to each key is the expected shape and data type.

```
- INPUTS
---- jets:                           (a collection with several jets per event)
------- MASK      : [NUM_EVENTS, MAX_JETS]   : bool
------- pt        : [NUM_EVENTS, MAX_JETS]   : float
------- eta       : [NUM_EVENTS, MAX_JETS]   : float
------- phi       : [NUM_EVENTS, MAX_JETS]   : float
------- mass      : [NUM_EVENTS, MAX_JETS]   : float
------- btag      : [NUM_EVENTS, MAX_JETS]   : float
---- lepton:                         (a single object per event)
------- pt        : [NUM_EVENTS, ]           : float
------- eta       : [NUM_EVENTS, ]           : float
------- ...
---- met:
------- pt        : [NUM_EVENTS, ]           : float
------- phi       : [NUM_EVENTS, ]           : float

- EVENT
---- label        : [NUM_EVENTS, ]           : int      (or CLASSIFICATIONS/EVENT/signal)
---- weight       : [NUM_EVENTS, ]           : float    (optional)
---- fold         : [NUM_EVENTS, ]           : int      (optional: cross-validation fold; else --n-folds)
---- event        : [NUM_EVENTS, ]           : int      (event number; needed with --n-folds)
---- other_scalar : [NUM_EVENTS, ]           : float    (any quantity named under `event:`)
```

Anything else in the file (for example generator-level `TARGETS/`) is ignored.

## Jets

A collection with several members per event (the jets) is a group with one array per feature, each
of shape `[NUM_EVENTS, MAX_JETS]`. Every event is stored with the same number of jet slots,
`MAX_JETS`, so events with fewer jets are padded; `MASK` marks a real jet with `True` and a padding
slot with `False`. Store all jets of the event, not only the b-jet candidates: which jets are the b
jets is decided by the event description when the file is loaded, so one file serves any choice.
The loader turns `MASK` into the last channel of every object tensor (+1 real, -1 padding); for the
`context` objects this channel is fed to the network as given (+1/-1 from the loader).

The column names are the ones used in the event description. The columns listed as `features` there
are the inputs of the network: `pt`, `eta`, `phi` and `mass` are the kinematics, any other feature
(`isE`, `isM`, ...) is an additional input for that object. Any column of a collection can be named in a
candidate line (`btag` for `b: jets btag 4`, [EventInfo.md](./EventInfo.md)); it is read for that choice
only and need not be a feature; listing it under `features` would make it an input as well.

## Single objects

A lepton or the missing transverse momentum is a group with one array per feature of shape
`[NUM_EVENTS, ]` and no `MASK`. The missing momentum needs only `pt` and `phi`.

## Group names

A collection named `x` is read from `INPUTS/x`, matched case-insensitively, or from one of the aliases
`jets`/`jet` → `Jets`, `Source`; `lepton` → `Lepton`, `Leptons`; `met` → `Met`, `MET`.

## Labels, weights and folds

* **Labels.** `EVENT/label` (or, when absent, `CLASSIFICATIONS/EVENT/signal`) holds the class of the
  event: `0` for background not described by a hypothesis ("rest"), and `1, 2, ...` for the outputs
  of the event description in order (`1` = signal in both examples). A `labels:` section in the event
  description replaces this convention by its own list of class names in label order
  ([EventInfo.md](./EventInfo.md)).
* **Weights.** `EVENT/weight` is the event weight (1 when absent). Each class is normalised during
  training, so only the relative weights within a class matter. Negative weights count as 0.
* **Folds.** The cross-validation fold of an event is `EVENT/fold` when the file has that column; the
  ids of `K` folds must be exactly `0 .. K-1`. Without the column, pass `--n-folds K` to
  `corenet-train` (and to `corenet-validate`): the fold is then
  `squares(event) % K`, a hash of the event number in `EVENT/event` (integers ≥ 0) with the Squares
  counter-based generator (`corenet.data.folds.fold_of`; `--fold-seed`, default 0, selects the hash
  key), so any subset of a sample gets the same folds and no random split is ever made. With
  `--n-folds K` an existing `EVENT/fold` column is ignored. The rule used (`EVENT/fold` or
  `squares(event) % n_folds`), the number of folds and the seed are stored in the checkpoint, so
  `corenet-evaluate` applies the same rule to any file. `corenet-train --fold k` holds out fold `k`
  for testing and `--exclude-folds j` uses fold `j` for validation (`--exclude-folds next`: fold
  `(k + 1) % n_folds`). Only selecting a fold needs the folds: `corenet-evaluate --fold -1` and
  `corenet-export` also work on a file with neither `EVENT/fold` nor `--n-folds` (the events then
  carry fold -1 in the prediction file).
* **Flags.** Boolean columns `EVENT/<name>` named by `masked_classes` ([Options.md](./Options.md)) mark
  the events whose listed classes are removed from the softmax in training. They are boolean, or
  integers 0/1. No flagged event may carry one of the classes masked for its flag: `corenet-train`
  and `corenet-validate` reject such a file and give the number of offending events per flag and class.
* **Event identifier.** `EVENT/event` is copied to the prediction files and is the input of the
  `squares(event) % n_folds` rule.

## Example

The example files `examples/data/hh4b_small.h5`, `examples/data/bbww_small.h5` and
`examples/data/tth_small.h5` (20,000 events each) follow this layout:

* `hh4b`: jets in `INPUTS/Source` (12 slots: `pt eta phi mass btag MASK`, `btag` is 0 or 1), labels in
  `CLASSIFICATIONS/EVENT/signal` (0 multijet, 1 HH), `EVENT/weight`, `EVENT/fold` (5 folds),
  `EVENT/event`;
* `bbww`: jets in `INPUTS/Jets` (12 slots), `INPUTS/Lepton` (`pt eta phi mass isE isM`), `INPUTS/Met`
  (`pt phi`), labels in `EVENT/label` (1 HH, 2 tt̄; no class 0), `EVENT/weight`, `EVENT/fold`,
  `EVENT/event`;
* `tth`: the layout of `bbww` without `TARGETS` groups, at least 6 jets per event, labels in
  `EVENT/label` (1 ttH, 0 tt̄).

They are random subsamples of the full files `hh4b_all.h5`, `bbww_all.h5` and `tth_all.h5`, written with
`python scripts/make_small_h5.py <full> <small> --events 20000`. Their `EVENT/fold` column equals the
fold rule with 5 folds and seed 0 (`--n-folds 5 --fold-seed 0`), so the column and `--n-folds 5`
give the same folds on these files; `corenet-validate --n-folds 5` reports that agreement. Check any
file against an event description with

```bash
corenet-validate --config examples/bbww/bbww.yaml --input examples/data/bbww_small.h5
```

On a machine without a GPU (no CUDA driver), prefix the command with `CONDA_OVERRIDE_CUDA=12.0`; the
validator only reads the file.
