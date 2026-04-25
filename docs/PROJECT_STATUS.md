# Project Status

ToposAI is an experimental neuro-symbolic research codebase. The core package
contains tested PyTorch components inspired by category theory, fuzzy logic,
sheaf-style consistency, probe-based reconstruction, and related structures.

## What Is Solid Today

- Core package imports and CPU tests run through pytest.
- Runtime smoke checks cover the main lightweight demos and benchmarks.
- Claim-hygiene tests guard against unqualified grand claims in docs and demos.
- Packaging metadata is present in `pyproject.toml` and `setup.py`.
- CI runs lint, compile checks, CPU tests, claim hygiene, and package build.
- CI enforces a core coverage floor and verifies wheel installation in a
  fresh virtual environment.

## What Is Still Experimental

- Many `experiments/` scripts are narrative research demos, not formal proofs.
- Benchmark numbers are environment-dependent and should be re-run on clean
  hardware before publication.
- CUDA/Triton paths need dedicated GPU CI or a reproducible benchmark machine.
- Language-model quality still needs standard perplexity/evaluation benchmarks.
- Several category-theory modules are differentiable proxies rather than full
  categorical constructions.

## Release Checklist

Before publishing a release:

```bash
ruff check topos_ai/
python -m compileall -q topos_ai tests experiments benchmarks
pytest -q
pytest --cov=topos_ai --cov-report=term-missing --cov-fail-under=85
python -m build
```

Recommended manual checks:

- Confirm `git status --short` contains no generated cache files.
- Re-run key benchmarks on a clean machine and record hardware details.
- Review README benchmark wording against the latest measured results.
- Build and install the wheel in a fresh virtual environment.

## Maturity Matrix

| Area | Status | Evidence |
|------|--------|----------|
| Formal finite category/topos core | Research-library prototype | Law tests, Yoneda density tests, sheafification tests, quantifier tests |
| Finite quasi-category skeletons | Research-library prototype | Inner horn filler tests, degeneracy identity tests, 3-simplex associativity coherence tests |
| HoTT finite groupoid semantics | Research-library prototype | Identity type, inverse path, composition, and transport functoriality tests |
| Neural/proxy modules | Experimental | Shape, boundedness, runtime, and smoke tests |
| Benchmarks | Experimental | Reproducibility docs required before publication |
| GPU/Triton kernels | Experimental | CPU fallback tested; dedicated GPU CI still needed |

## Research-library readiness

The repository is ready to be presented as a research-library prototype when these checks pass together:

- mathematical contracts exist and are linked from API docs
- public API contract exists and names stable formal symbols
- examples run without network access
- formal property tests cover Yoneda density, Heyting adjunction, inverse image laws, and finite inner-Kan checks
- documentation builds in CI
- wheel smoke imports formal category, quasi-category, and HoTT APIs
