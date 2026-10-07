# Model export

```bash
corenet-export --checkpoint runs/hh4b/model_fold0.pkl --input examples/data/hh4b_small.h5 \
    --output-dir runs/hh4b/export
```

writes, in `--output-dir` (basename = the checkpoint's, or `--name`):

| file | content |
|---|---|
| `model_fold0.pt` | TorchScript module (`torch.jit.trace`); load with `torch.jit.load`, no `corenet` needed |
| `model_fold0.onnx` | ONNX model (opset 17 by default, `--opset`), dynamic batch dimension |
| `model_fold0.json` | input names and shapes, output names, classes, and the measured agreement with PyTorch |

`--formats torchscript` or `--formats onnx` exports only one of the two.

## Inputs and outputs of the exported model

The exported graph starts **after** the role selection. Its positional inputs are the leaf tensors in
the order of the JSON file's `inputs` (the roles in configuration order, then the single-object
collections, then `event`), each exactly as `corenet.data.H5Dataset` produces it:
`(N, len(features) + 1, n)` float32 with the collection's features in configuration order, the presence
flag (+1 / −1) as the last channel and −1 in every channel of a padded slot, and `event` `(N, n_event)`
with the raw event scalars (`nSelJets`, `HT` as computed by the loader). For the examples:

| example | inputs |
|---|---|
| HH → 4b | `b` (N, 6, 4), `other` (N, 6, 8), `event` (N, 2) |
| HH → bbWW | `b` (N, 6, 2), `q` (N, 6, 2), `lepton` (N, 7, 1), `met` (N, 3, 1), `event` (N, 2) |

All normalisations and derived quantities (four-vector sums, the neutrino solution, log transforms,
batch-norm statistics) are inside the graph. The outputs are probabilities:

| output | shape | content |
|---|---|---|
| `class_probabilities` | (N, n_classes) | softmax of the main head, columns `["rest", <outputs>...]` |
| `views_<hypothesis>` | (N, V) | candidate probabilities of every hypothesis with more than one candidate |
| `head_<output>` | (N,) | sigmoid of every auxiliary head |

To use an exported model outside this package, reproduce the role selection of the loader
(`src/corenet/data/h5.py`: `btag` = largest `btag × 10⁶ + pT`, `pt`, `rest`, filled in configuration
order) and the input layout above.

```python
import onnxruntime as ort
from corenet import load_config
from corenet.data import H5Dataset

ds = H5Dataset("examples/data/hh4b_small.h5", load_config("examples/hh4b/hh4b.yaml"))
sess = ort.InferenceSession("runs/hh4b/export/model_fold0.onnx")
feeds = {i.name: ds.inputs[i.name][:100].numpy() for i in sess.get_inputs()}
class_probabilities, views_HH = sess.run(None, feeds)
```

## Verification

After writing, both files are reloaded (TorchScript with `torch.jit.load`, ONNX with onnxruntime on the
CPU) and run on the first `--n-events` (1024) events of `--input` and on a second batch of a different
size (to check the dynamic batch dimension). The maximum absolute difference of any output to the
PyTorch model is written to the JSON file and must be ≤ `--atol` (default 1e-4), otherwise the command
fails.

Measured on the quick-start models (2 training epochs, events of the small files):

| example | TorchScript | ONNX Runtime (max) | ONNX Runtime (median) |
|---|---|---|---|
| HH → 4b | 0 (bit-identical) | 6.8e-6 | |
| HH → bbWW | 0 (bit-identical) | 2.3e-5 (5,000 events) | 6e-8 |

The ONNX differences are float32 rounding: on the same bbWW events, PyTorch in float32 itself differs from
a float64 evaluation of the same network by up to 4.3e-5.

## Notes and limitations

* **TorchScript** is produced by tracing. The network's control flow depends only on the configuration
  (the candidate tables are fixed when the network is built), not on the data, so the trace is exact
  for any batch; the tracer warning about `allidx.numel()` refers to such a structural constant.
  `torch.jit` is deprecated in recent PyTorch releases (2.13 prints a DeprecationWarning) but works.
* **ONNX** uses PyTorch's TorchScript-based exporter (`dynamo=False`). Two adjustments were needed:
  the context mask builds its identity matrix as float and casts it to bool (ONNX Runtime has no
  boolean `EyeLike` kernel; the mask is unchanged), and `aten::sinh` / `aten::asinh`, used by the
  `lepton + met` recipe, are mapped to the native ONNX `Sinh` / `Asinh` operators at export time.
* The exported models are inference-only (eval mode, frozen batch-norm statistics). Ghost-batch
  normalisation in training mode is not exported.
