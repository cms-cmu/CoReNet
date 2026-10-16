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

#### Option A: Using Pixi (Recommended on Linux with CUDA GPU)
```bash
pixi install
pixi shell
```

#### Option B: Using Pip / Virtual Environment (CPU / Local Dev)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test,export,lint]"
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

---

## Pull Request Guidelines

1. **Branch from `main`**: Create a descriptive feature branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Ensure All Checks Pass**: All CI checks (formatting, linting, package validation, and test suite across supported Python versions) must pass.
3. **Commit Discipline**: Keep commits scoped and clean. Do not stage unrelated scratch or temporary files.
4. **Open a PR**: Submit a pull request against `main` describing the rationale and changes made.
