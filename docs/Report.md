# Results Report

`corenet-report` writes one HTML file with the results of a training and the checks of its inputs. The
page holds every figure and table itself and opens in any browser, offline.

## Running

1. Train (`corenet-train`) and, to also see the training and validation folds in the report, evaluate every
   event of the file into the same directory:

   ```bash
   corenet-evaluate --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_all.h5 \
       --output runs/hh4b/eval_fold0.h5 --fold -1
   ```

2. Write the report:

   ```bash
   corenet-report --run-dir runs/hh4b --input examples/data/hh4b_all.h5 --output runs/hh4b/report.html
   ```

* `--run-dir runs/hh4b`: the directory written by `corenet-train` (and `corenet-evaluate`).
* `--input examples/data/hh4b_all.h5`: the file the run was trained on.
* `--output runs/hh4b/report.html`: the page to write (default: `report.html` in the run directory).
* `--folds 0 1`: report these folds only (default: every fold found in the directory).
* `--config my_config.yaml`: use this event description instead of the one stored in the checkpoint.
* `--max-events 500000`: at most this many events of the input file for the input and weight
  distributions, a random subsample above it; `0` uses every event.
* `--n-folds 5`, `--fold-seed 0`, `--rest-order pt`: the fold rule and the ordering of `rest` jets
  (default: as recorded in the run).

On a machine without a GPU, prefix the command with `CONDA_OVERRIDE_CUDA=12.0`; the report runs on the CPU.

## Input files

Per fold `k`, in `--run-dir`:

| file | content | needed for |
|---|---|---|
| `history_fold{k}.json` | training settings and per-epoch history | run summary, training curves |
| `model_fold{k}.pkl` | checkpoint | event description, parameter count, the inputs as the network sees them |
| `predictions_fold{k}.h5` | predictions on the held-out fold | ROC, AUC, scores, view probabilities, pairing accuracy |
| `eval_fold{k}.h5` | `corenet-evaluate` output; with `--fold -1` every event of the file | the training and validation folds in the ROC, AUC and score figures (overtraining check) |

A fold with some of these files missing is reported with what is there; the last section of the page lists
what could not be produced and why.

## Output

| file | content |
|---|---|
| `report.html` | the page, figures embedded as PNG |

## Sections of the page

1. **Run summary**: the event description (collapsed), the input file, the classes and their event counts
   and weights, the training settings, and per fold the validation folds, the events of the training,
   validation and test sets, the epochs run, the best epoch, the parameter count and which prediction
   files were found.
2. **Training curves**: training loss, validation loss and validation AUC per epoch, one line per fold,
   the best epoch marked; the values of the last epoch in a table.
3. **Classification**: the ROC curve of the signal score (signal against every other class) per fold and
   the mean over folds with its spread, on a linear and a logarithmic background-efficiency axis; the AUC
   per fold and split with mean ± sd; the signal-score distributions of every class on the test folds,
   overlaid with the training folds when `eval_fold{k}.h5` holds them; for more than two classes the
   confusion matrix at the most probable class, the ROC curve of every class against the rest, and the
   ROC curve of every auxiliary score (`heads/`).
4. **Hypotheses and pairing**: for every hypothesis with several candidate assignments (for HH → bb̄bb̄ the
   three pairings of the four b jets into two Higgs bosons), the distribution of the highest view
   probability per class and the share of each assignment; when the input file has `TARGETS` naming the
   generator-matched jets of a particle of the decay chain (`TARGETS/h1/{b1,b2}`, `TARGETS/h2/{b1,b2}`),
   the fraction of events in which the most probable assignment is the generator one, per fold, for the
   events whose matched jets are all among the candidates, and the same fraction as a function of the
   view probability.
5. **Inputs as the network sees them**: the quantities the network embeds, per class: for every decay
   product of every particle its log(1+pT), η, Δφ to the parent candidate and log(1+m) (plus any further
   feature, e.g. `isE`, `isM`), for every particle candidate its own kinematics, the jets of the `context`,
   the `relations` pairs (log(1+M), ΔR, Δφ) and the event scalars (`nSelJets` as log(nSelJets − (min_jets − 1))).
6. **Raw inputs**: the columns of the input file for the jets each role selected, slot by slot (including a
   column read for the selection only, such as `btag`), the single objects and the event scalars, per class.
7. **Weights**: the event weights per class as stored and as the training uses them (clipped at 0, every
   class summing to 1), weight statistics with the effective number of events and the negative weights, the
   events and weights per fold and class, and the training set of every fold.
8. **Other checks**: the mean event weight as a function of the leading and the last candidate jet pT and
   of the jet multiplicity, per class.

A figure whose inputs the run did not provide (no `TARGETS`, no predictions on the training folds, ...) is
left out.

Weighted distributions use the event weights of the file clipped at 0, as the training does; each class is
normalised to unit area. Mean ± sd are over folds.
