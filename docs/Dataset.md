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
---- fold         : [NUM_EVENTS, ]           : int      (needed for training)
---- event        : [NUM_EVENTS, ]           : int      (optional event identifier)
---- other_scalar : [NUM_EVENTS, ]           : float    (any quantity named under `event:`)
```

Anything else in the file (for example generator-level `TARGETS/`) is ignored.

## Jets

A collection with several members per event (the jets) is a group with one array per feature, each
of shape `[NUM_EVENTS, MAX_JETS]`. Every event is stored with the same number of jet slots,
`MAX_JETS`, so events with fewer jets are padded; `MASK` marks a real jet with `True` and a padding
slot with `False`. Store all jets of the event, not only the b-jet candidates: which jets are the b
jets is decided by the event description when the file is loaded, so one file serves any choice.

The feature names are the ones used in the event description. `pt`, `eta`, `phi` and `mass` are the
kinematics; `btag` is only used to pick the b-jet candidates; any other feature (`isE`, `isM`, ...) is
an additional input for that object.

## Single objects

A lepton or the missing transverse momentum is a group with one array per feature of shape
`[NUM_EVENTS, ]` and no `MASK`. The missing momentum needs only `pt` and `phi`.

## Group names

A collection named `x` is read from `INPUTS/x`, matched case-insensitively, or from one of the aliases
`jets`/`jet` → `Jets`, `Source`; `lepton` → `Lepton`, `Leptons`; `met` → `Met`, `MET`.

## Labels, weights and folds

* **Labels.** `EVENT/label` (or, when absent, `CLASSIFICATIONS/EVENT/signal`) holds the class of the
  event: `0` for background not described by a hypothesis ("rest"), and `1, 2, ...` for the outputs
  of the event description in order (`1` = signal in both examples).
* **Weights.** `EVENT/weight` is the event weight (1 when absent). Each class is normalised during
  training, so only the relative weights within a class matter. Negative weights count as 0.
* **Folds.** `EVENT/fold` is the cross-validation fold of the event. `corenet-train --fold k` holds
  out fold `k` for testing and `--exclude-folds j` uses fold `j` for validation.
* **Event identifier.** `EVENT/event` is copied to the prediction files.

## Example

The example files `examples/data/hh4b_small.h5` and `examples/data/bbww_small.h5` (20,000 events each)
follow this layout:

* `hh4b`: jets in `INPUTS/Source` (12 slots: `pt eta phi mass btag MASK`, `btag` is 0 or 1), labels in
  `CLASSIFICATIONS/EVENT/signal` (0 multijet, 1 HH), `EVENT/weight`, `EVENT/fold` (5 folds),
  `EVENT/event`;
* `bbww`: jets in `INPUTS/Jets` (12 slots), `INPUTS/Lepton` (`pt eta phi mass isE isM`), `INPUTS/Met`
  (`pt phi`), labels in `EVENT/label` (1 HH, 2 tt̄; no class 0), `EVENT/weight`, `EVENT/fold`,
  `EVENT/event`.

They are random subsamples of the full files `hh4b_all.h5` and `bbww_all.h5`, written with
`python scripts/make_small_h5.py <full> <small> --events 20000`. Check any file against an event
description with

```bash
corenet-validate --config examples/bbww/bbww.yaml --input examples/data/bbww_small.h5
```
