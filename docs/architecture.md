# Architecture: one forward pass

CoReNet (Combinatorial Residual Network) builds, from the decay trees of the configuration, one
pixel (a `d`-dimensional feature vector) per candidate of every node, bottom-up, and classifies the
event from the selected candidates of the hypotheses. With the examples' `d = 8` the networks have
1,508 (HH → 4b) and 3,840 (HH → bbWW) parameters. The implementation is
`src/corenet/models/network.py` (`CoReNet`).

**Inputs.** One tensor per leaf, `(N, len(features) + 1, n)`: the collection's features in
configuration order and a presence flag (+1 / −1) as the last channel; plus `event`, `(N, n_event)`.
`corenet.data.H5Dataset` builds them from an HDF5 file, including the role selection.

1. **Kinematics.** With `reflect_eta`, every η is multiplied by the sign of the η of the first member
   of the first leaf. Node four-vectors are built bottom-up from the candidate tables (sums, or a
   recipe). A node is present when all its constituents are.
2. **Features.** Per child slot of every composite: log(1+pT), η, Δφ(child, node), log(1+m), plus a
   leaf's extra features; per node: log(1+pT), η, φ or Δφ to its parent, log(1+m) (the top node drops
   φ), plus recipe features; event scalars (`log(nSelJets − 3)` for jet multiplicities). Non-finite
   values (padded objects) are replaced by −1.
3. **Embedding.** Every input goes through a ghost-batch-normalised (GBN) 1×1 convolution to `d`
   features, then a second GBN convolution. The normalisation of these input layers is fixed from the
   training set before training (`updateMeanStd` / `initMeanStd`). The event embedding is added to every
   leaf embedding. Padded leaf slots are zeroed.
4. **Reinforce.** For every composite node in bottom-up order (a node shared by several hypotheses is
   computed once): a residual block of two reinforce layers, each gathering the node's child pixels
   and its own pixel and convolving them (kernel = stride = number of children + 1); for two
   interchangeable composite children their symmetric and antisymmetric combinations are used, so the
   result does not depend on their order. Then, if the hypothesis has a `context`, the first-level
   node attends to the context objects; every `attention` edge is applied as soon as both of its nodes
   exist (attention with the pair bias and exclusivity mask, or a pairwise merge).
5. **Select.** For a hypothesis with several views: a GBN convolution gives one score per view,
   absent views are masked, and the softmax-weighted sum of the view pixels is the selected candidate.
6. **Heads.** The selected candidates of the `combine` hypotheses and the `relations` pixels are
   concatenated and, if there is more than one, reduced by a k·d → d convolution: this is the event
   pixel. The main head maps it to the class logits; auxiliary heads map a hypothesis' selected
   candidate (or the event pixel) to one logit each.

`forward(inputs)` returns `{"logits": (N, n_classes), "views": {hypothesis: (N, V)}, "heads":
{output: (N,)}, "event": (N, d, 1)}` (logits, before softmax / sigmoid).

**Building blocks** (`src/corenet/models/blocks.py`): ghost batch normalisation
(arXiv:1705.08741) in front of every convolution, with masked statistics for padded pixels;
phase-symmetric convolutions (arXiv:1603.05201), which compute half the filters and output
(x, −x); SiLU non-linearities; `MinimalAttention`, multi-head attention with GBN-normalised scores,
a sigmoid-scaled softmax (a query need not pick any key), optional gated pair features
(`attention_gate`) and bottleneck projections (`kv_projection`).
