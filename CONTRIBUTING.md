# Contributing to ToposAI

Thank you for your interest in contributing! ToposAI sits at the intersection of category theory and deep learning — we welcome both mathematical and engineering contributions.

## Getting Started

```bash
git clone https://github.com/Tehlikeli107/ToposAI.git
cd ToposAI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,full]"
```

## Running Tests

```bash
# Fast CPU suite used by CI
pytest \
  tests/test_core.py \
  tests/test_models.py \
  tests/test_generation.py \
  tests/test_category_core.py \
  tests/test_claim_hygiene.py \
  tests/test_runtime.py \
  -m "not cuda and not triton and not slow" \
  -v --tb=short

# Full test suite (skips CUDA/Triton tests if no GPU)
pytest -m "not cuda and not triton" -v

# Syntax/import safety check
python -m compileall -q topos_ai tests experiments benchmarks

# With coverage
pytest --cov=topos_ai --cov-report=html
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check topos_ai/          # check
ruff check topos_ai/ --fix    # auto-fix
```

All pull requests must pass `ruff check topos_ai/` before merging.

## Claim Hygiene

ToposAI contains many research demos, so wording matters. Avoid presenting
toy experiments as formal proofs, product guarantees, or production-grade
benchmarks unless the repository contains the corresponding evidence.

Before opening a PR, run:

```bash
pytest tests/test_claim_hygiene.py -q
```

Prefer language like "research demo", "proxy", "measured on this benchmark",
or "inspired by" when a module is not a formal mathematical construction.

## Project Structure

```
topos_ai/           # Core library (installable package)
  logic.py          #   Heyting algebra & Gödel T-norms
  topology.py       #   Persistent homology (Betti numbers)
  nn.py             #   ToposTransformer building blocks
  models.py         #   ToposTransformer end-to-end model
  math.py           #   Categorical composition operators
  cohomology.py     #   Cech cohomology engine
  kernels.py        #   Triton CUDA kernels (FlashTopos)
  optim.py          #   ToposAdam optimizer
  generation.py     #   Constrained decoding
  verification.py   #   Lean 4 proof bridge
  yoneda.py         #   Yoneda embedding
  hott.py           #   Homotopy type theory utilities
  reasoning.py      #   Defeasible reasoning engine
  ...
experiments/        # Theoretical simulations (proof-of-concept)
applications/       # Domain-specific demos (finance, NLP, bio...)
benchmarks/         # Performance comparisons
tests/              # pytest test suite
docs/               # MkDocs documentation
```

## Contribution Types

### Mathematical Contributions
- Corrections to category-theoretic implementations (e.g., sheaf conditions, topos axioms)
- New fuzzy logic operators with proper differentiability proofs
- Improved Betti number / homology algorithms

### Engineering Contributions
- Triton kernel optimizations for FlashTopos
- KV-cache and inference improvements
- New application scripts in `applications/`

### Documentation
- Improving docstrings (English or Turkish)
- Adding worked examples to `docs/`

## Pull Request Guidelines

1. **Branch naming:** `feature/your-feature`, `fix/bug-description`, `docs/what-you-added`
2. **Tests:** Add or update tests in `tests/` for any change to `topos_ai/`
3. **Honest positioning:** If a script is a simulation or proof-of-concept, say so in the docstring. Avoid overclaiming.
4. **One concern per PR:** Keep PRs focused — mathematical fixes separate from engineering changes.

## Reporting Issues

Please use [GitHub Issues](https://github.com/Tehlikeli107/ToposAI/issues) with:
- A minimal reproducible example
- Expected vs. actual behavior
- Python and PyTorch version

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
