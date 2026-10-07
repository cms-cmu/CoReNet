# Example: HH → bb̄ WW → bb̄ qq̄′ ℓν, against tt̄

One Higgs boson decays to b jets, the other to a hadronic and a leptonic W. The main background,
tt̄ → b qq̄′ b ℓν, has the same objects, so it is described as a second hypothesis on the same inputs.

## Decay chain

```yaml
objects:
  jets:   {features: [pt, eta, phi, mass, btag], max: 12}
  b:      jets btag 2                                  # the 2 highest-b-tag jets
  q:      jets rest 2                                  # the 2 leading remaining jets
  lepton: {features: [pt, eta, phi, mass, isE, isM]}   # one lepton; isE / isM enter the lepton embedding
  met:    {features: [pt, phi]}                        # missing transverse momentum (eta = mass = 0)
event: [nSelJets, HT]

hypotheses:
  HH:
    H_bb: [b, b]
    H_WW:
      W_had: [q, q]
      W_lep: lepton + met      # recipe
  tt:
    t_had: [b, W_had]          # reuses the W_had node of HH
    t_lep: [b, W_lep]          # reuses the W_lep node of HH

attention:
  - W_lep: W_had
  - t_had: t_lep

relations:
  - [lepton, met]
  - [H_bb, met]
  - [lepton, W_had]

outputs:
  signal: HH
  ttbar:  tt
combine: [HH, tt]
training:
  reflect_eta: false          # (plus the network options at their defaults)
```

* **Objects.** Jets (two roles: `b` and `q`), and two single objects, `lepton` and `met`, which are leaves
  by themselves.
* **Signal tree.** `HH → H_bb H_WW`, `H_bb → b b`, `H_WW → W_had W_lep`, `W_had → q q`. Every node has a
  single candidate (the roles fix which jets are b and q).
* **Recipe.** `W_lep: lepton + met` reconstructs the leptonic W with the W-mass constraint on the
  neutrino pz (smaller-|pz| solution) and adds log(1 + m_T) and an off-shell score as node features.
* **Background tree and node reuse.** `tt → t_had t_lep` with `t_had → b W_had`, `t_lep → b W_lep`. `W_had`
  and `W_lep` are the nodes declared under HH (same weights, computed once). Each top takes one of the two
  b jets, and a b jet cannot be used twice, so there are exactly 2 tt assignments (the views of `tt`,
  written to `views/tt`).
* **Attention edges.** `W_lep: W_had` and `t_had: t_lep` add the pair features (log M, ΔR, Δφ) of the two
  nodes. The top pair masks the combination that would use the same b twice; every query is left with a
  single partner, so both edges are built as pairwise merges.
* **Relations.** (log M, ΔR, Δφ) of lepton–MET, H_bb–MET and lepton–W_had feed the main head.
* **Outputs.** Classes 0 = rest, 1 = signal (HH), 2 = tt̄; the example file has only labels 1 and 2. The
  main head sees the selected HH and tt candidates and the three relations; `ttbar` also gets its own
  auxiliary head on the selected tt candidate (`heads/ttbar`).

3,840 parameters with `features: 8`. `bbww_hyperparameters.yaml` is the same network with every option
written out.

## Train and evaluate (quick start, small file, CPU, 2 epochs)

```bash
corenet-validate --config examples/bbww/bbww.yaml --input examples/data/bbww_small.h5
corenet-train --config examples/bbww/bbww.yaml --input examples/data/bbww_small.h5 \
    --fold 0 --exclude-folds 1 --epochs 2 --device cpu --output-dir runs/bbww
corenet-evaluate --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_small.h5 \
    --output runs/bbww/eval_fold0.h5
corenet-export --checkpoint runs/bbww/model_fold0.pkl --input examples/data/bbww_small.h5 \
    --output-dir runs/bbww/export
```

Full recipe (20 epochs) on the full file, on a GPU:

```bash
corenet-train --config examples/bbww/bbww.yaml --input examples/data/bbww_all.h5 \
    --fold 0 --exclude-folds 1 --device cuda --output-dir runs/bbww_full
```
