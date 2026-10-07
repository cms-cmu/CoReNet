# Example: HH → bb̄ bb̄

Two Higgs bosons, each decaying to a pair of b jets, against multijet background.

## Decay chain

```yaml
objects:
  jets:  {features: [pt, eta, phi, mass, btag], max: 12}
  b:     jets btag 4          # the 4 jets with the highest b tag (0/1 here), then highest pT
  other: jets rest 8          # up to 8 further jets, by pT
event: [nSelJets, HT]

hypotheses:
  HH:
    H1: [b, b]
    H2: [b, b]
    context: other

outputs:
  signal: HH
combine: [HH]
```

* **Objects.** One jet collection with up to 12 jets. The role `b` takes 4 of them; `other` takes the
  remaining ones. `btag` only drives the selection, it is not a network input.
* **Tree.** `HH → H1 H2`, `H → b b`. Because the two b of a Higgs are interchangeable and so are the two
  Higgs bosons, the network enumerates the 6 b-jet pairs as H candidates and the 3 ways to split the four
  b jets into two pairs (01|23, 02|13, 03|12) as the views of HH. H1 and H2 share their weights.
* **Context.** After the dijet level every H candidate attends to all jets of the event (the four b
  plus up to eight `other` jets), except its own two jets.
* **Selection.** A learned score per pairing, softmax over the three pairings: the per-event pairing
  probabilities are written to `views/HH` in the prediction files.
* **Output.** One output, so two classes: 0 = multijet (rest), 1 = signal. The labels of the example
  file are in `CLASSIFICATIONS/EVENT/signal`.

1,508 parameters with `features: 8`. `hh4b_hyperparameters.yaml` is the same network with every
option written out at its default value.

## Train and evaluate (quick start, small file, CPU, 2 epochs)

```bash
corenet-validate --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_small.h5 \
    --fold 0 --exclude-folds 1 --epochs 2 --device cpu --output-dir runs/hh4b
corenet-evaluate --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output runs/hh4b/eval_fold0.h5
corenet-export --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output-dir runs/hh4b/export
```

Full recipe (20 epochs) on the full file, on a GPU:

```bash
corenet-train --config examples/hh4b/hh4b.yaml --input examples/data/hh4b_all.h5 \
    --fold 0 --exclude-folds 1 --device cuda --output-dir runs/hh4b_full
corenet-evaluate --checkpoint runs/hh4b_full/model_fold0.pkl --input examples/data/hh4b_all.h5 \
    --output runs/hh4b_full/eval_fold0.h5
```
