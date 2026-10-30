# Contributing to CoReNet

Thank you for contributing to CoReNet! We welcome contributions, bug fixes, feature requests, and documentation improvements.

---

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/cms-cmu/CoReNet.git
cd CoReNet
```

### 2. Environment Setup

```bash
pixi install
pixi shell
```

---

## Code Quality & Standards

We use [Ruff](https://astral.sh/ruff) for fast, consistent code formatting and linting.

### Formatting & Linting
Before submitting a pull request, ensure your code passes formatting and linting:

```bash
# Format code
ruff format .

# Check formatting
ruff format --check .

# Run static analysis / linter
ruff check .

# Automatically fix simple lint issues
ruff check --fix .
```

---

## Running Tests

Run the test suite with `pytest`:

```bash
pytest tests/ -v
```

To run with test coverage:

```bash
pytest --cov=corenet tests/
```

### Parity with the research code

```bash
CORENET_REFERENCE_REPO=/path/to/research-checkout pytest tests/reference -s
```

- `CORENET_REFERENCE_REPO`: checkout of the research repository; its `src/hcr` is imported, `data/h5/{hh4b,bbww}_all.h5` and the checkpoints under `models/` are read. Without it these tests are skipped.

The tests require identical configurations, parameter counts, initial weights, loader tensors, eval-mode outputs and checkpoint evaluation. Training-mode outputs are not compared with the research code, because CoReNet embeds the slots of one leaf role in one node together (they share the ghost-batch statistics); the golden-file test below covers them.

### Golden files of the training-mode numerics

`pytest tests` runs `tests/test_training_regression.py`: training-mode outputs, first-step gradients, five SGD steps and the eval-mode outputs after them, on 1,024 events of `examples/data/{hh4b,bbww}_small.h5`, on the CPU with one thread, compared with `tests/golden/{hh4b,bbww}.npz`.

After an intentional change of the training-mode numerics, or of `examples/data/{hh4b,bbww}_small.h5`:

1. Regenerate the golden files:
   ```bash
   python scripts/make_golden.py
   ```
   - `hh4b` or `bbww`: regenerate this case only (default: both)
2. Read the printed change per stage against the old files and the two noise probes. The script writes nothing if a probe exceeds a tenth of the test tolerance or the files exceed 2 MB.
3. Run `pytest tests/test_training_regression.py`.
4. Commit `tests/golden/*.npz` in the same commit as the change and give the reason in the commit message.

| file | content |
|---|---|
| `tests/golden/hh4b.npz` | HH → 4b: initial weights, frozen input statistics, training-mode outputs, first-step gradients, losses, weights after five SGD steps, eval-mode outputs |
| `tests/golden/bbww.npz` | the same for HH → bbWW |

---

## Pull Request Guidelines

1. **Branch from `main`**: Create a descriptive feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Ensure All Checks Pass**: All CI checks (formatting, linting, package validation, and test suite across supported Python versions) must pass.
3. **Commit Discipline**: Keep commits scoped and clean. Do not stage unrelated scratch or temporary files.
4. **Open a PR**: Submit a pull request against `main` describing the rationale and changes made.
