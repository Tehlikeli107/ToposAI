# ToposAI: Neuro-Symbolic AI via Category-Theory-Inspired Operators

[![ToposAI CI](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**ToposAI** is an experimental research library with a formal finite core for category/topos computations and neural/proxy components inspired by categorical structures. The formal core covers finite categories, presheaf topoi, Yoneda reconstruction, sheafification, Kripke-Joyal style internal logic, finite quasi-category horn checks, and 1-truncated HoTT path groupoid semantics.

The neural/proxy components explore how Goedel-Heyting logic, sheaf-style consistency, topological features, and categorical constraints can be embedded into PyTorch models. Proxy modules are documented as research scaffolds rather than complete implementations of the full mathematical theories that inspire them.

For the current maturity level, known limitations, and release checklist, see
[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

---

## Architecture at a Glance

| Component | Classical DL | ToposAI |
|-----------|-------------|---------|
| Attention scoring | Softmax(QK^T / sqrt(d)) | Goedel-Heyting implication, dot-product free scoring |
| Weight space | R, unconstrained | [0, 1] via sigmoid (morphism strength) |
| Residual connection | x + f(x) | x + f(x) - x * f(x) (probabilistic T-conorm) |
| Normalization | LayerNorm | Max-norm to [0, 1] |
| Output projection | Linear classifier | Cosine similarity (topological reachability) |
| Reasoning | Statistical next-token prediction | Relation composition and transitive closure demos |
| Optimizer | AdamW (Euclidean) | ToposAdam (Fisher-style natural-gradient scaling) |

---

## Installation

```bash
# Core library only
pip install -e .

# Full dependencies (Gradio, HuggingFace, yfinance, Triton, ...)
pip install -e ".[full]"

# Development (adds ruff, pytest-cov)
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
import topos_ai

model = topos_ai.models.ToposTransformer(vocab_size=50000, d_model=256, num_universes=8)
idx = torch.randint(0, 50000, (1, 64))
logits, kv_cache = model(idx)

R = torch.rand(10, 10)
R_inf = topos_ai.math.transitive_closure(R, max_steps=5)
```

---

## Repository Structure

```text
topos_ai/           # Core installable library
  logic.py          #   Goedel-Heyting algebra and subobject-classifier proxy
  formal_category.py # Finite categories/functors, presheaf topoi, Yoneda, Omega/J, CCC, sites/sheafification
  topology.py       #   Persistent homology - Betti numbers via boundary rank
  nn.py             #   TopologicalLinear, TopologicalNorm, MoE attention
  models.py         #   ToposTransformer end-to-end model
  math.py           #   Goedel relation composition, optional Lukasiewicz comparison
  cohomology.py     #   Cech cohomology - H0 consensus, H1 obstruction
  kernels.py        #   Triton CUDA kernel with PyTorch fallback
  optim.py          #   ToposAdam (Fisher-style scaling)
  generation.py     #   Reachability-constrained decoding mask
  reasoning.py      #   Defeasible reasoning and theorem-discovery demos
  verification.py   #   Lean 4 theorem transpiler bridge
  yoneda.py         #   Yoneda-inspired probe-distance reconstruction
  hott.py           #   Homotopy type theory inspired path finding

experiments/        # Theoretical simulations and proof-of-concept scripts
applications/       # Domain demos (finance, NLP, bioinformatics, seismic...)
benchmarks/         # Performance comparison scripts
tests/              # Pytest test suite
docs/               # MkDocs documentation and release notes
```

---

## Benchmarks

| Script | What it measures |
|--------|-----------------|
| `benchmark_sota.py` | Ultrametric tree search vs. dense softmax attention |
| `scaling_laws_benchmark.py` | VRAM vs. sequence length for SRAM gradient accumulation |
| `babi_logic_benchmark.py` | bAbI Task 15 style logical reasoning |
| `real_world_ontology_benchmark.py` | WordNet asymmetry vs. dot-product symmetry |

Benchmark claims should be treated as environment-specific until reproduced.
Use [docs/BENCHMARKS.md](docs/BENCHMARKS.md) to record hardware, commands, and
raw measurements before citing results.

### Running Applications and Benchmarks (Package-Friendly)

Scripts under `applications/` and `benchmarks/` can now be run without
manually editing `sys.path`:

```bash
# CLI entrypoints (installed via pyproject scripts)
topos-application real_world_solidity_auditor
topos-benchmark benchmark_sota

# Equivalent module execution
python -m applications.real_world_solidity_auditor
python -m benchmarks.benchmark_sota
```

---

## Limitations

This is an early-stage research framework. The following limitations apply:

- No standard perplexity benchmarks have been run against conventional language models yet.
- MoE speedups are not guaranteed; the current PyTorch implementation needs dedicated profiling.
- Scripts in `experiments/` are theoretical simulations, not empirical results.
- `TopologicalTokenizer` and distributed training are scaffolding-stage.
- Attention scoring is dot-product free; the output head uses cosine similarity.

---

## Running Tests

```bash
# Fast CPU tests
pytest tests/test_core.py tests/test_models.py -v -m "not cuda and not triton"

# Full suite with coverage
pytest --cov=topos_ai --cov-report=html
```

---

## Release And Project Health

- [Project status](docs/PROJECT_STATUS.md)
- [Benchmark reporting](docs/BENCHMARKS.md)
- [Release guide](docs/RELEASE_GUIDE.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

---

## Future Work

- Lean 4 integration against a larger formal theorem-proving corpus.
- Standard perplexity and downstream evaluations.
- Fused sparse MoE kernels for measured top-k expert compute savings.
- Better documentation for GPU/Triton benchmark reproducibility.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR
guidelines.

## License

MIT - see [LICENSE](LICENSE).

## Citation

If you use ToposAI in your research, please cite:

```bibtex
@software{toposai2026,
  title   = {{ToposAI}: Neuro-Symbolic AI via Category-Theory-Inspired Operators},
  year    = {2026},
  url     = {https://github.com/Tehlikeli107/ToposAI},
  license = {MIT}
}
```
k

- Lean 4 integration against a larger formal theorem-proving corpus.
- Standard perplexity and downstream evaluations.
- Fused sparse MoE kernels for measured top-k expert compute savings.
- Better documentation for GPU/Triton benchmark reproducibility.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR
guidelines.

## License

MIT - see [LICENSE](LICENSE).

## Citation

If you use ToposAI in your research, please cite:

```bibtex
@software{toposai2026,
  title   = {{ToposAI}: Neuro-Symbolic AI via Category-Theory-Inspired Operators},
  year    = {2026},
  url     = {https://github.com/Tehlikeli107/ToposAI},
  license = {MIT}
}
```
