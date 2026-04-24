# ToposAI: Neuro-Symbolic AI via Category Theory

[![ToposAI CI](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**ToposAI** is an experimental open-source research framework bridging Continuous Deep Learning (PyTorch) with Formal Category Theory — Toposes, Lukasiewicz Logic, Heyting Algebra, and the Yoneda Lemma.

Modern LLMs operate on statistical dot-products. This project investigates whether **logical truth values** and **morphism computations** can be embedded into differentiable neural architectures to produce more structured, verifiable reasoning.

---

## Architecture at a Glance

| Component | Classical DL | ToposAI |
|-----------|-------------|---------|
| Attention scoring | Softmax(QKᵀ / √d) | Lukasiewicz T-norm — dot-product free |
| Weight space | ℝ, unconstrained | [0, 1] via sigmoid (morphism strength) |
| Residual connection | x + f(x) | x + f(x) − x·f(x) (probabilistic T-conorm) |
| Normalization | LayerNorm | Max-norm to [0, 1] |
| Output projection | Linear classifier | Cosine similarity (topological reachability) |
| Reasoning | Statistical next-token | Transitive closure via Gödel/Lukasiewicz |
| Optimizer | AdamW (Euclidean) | ToposAdam (Fisher natural gradient) |

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

# Topos Language Model (attention scoring is dot-product free)
model = topos_ai.models.ToposTransformer(vocab_size=50000, d_model=256, num_universes=8)
idx   = torch.randint(0, 50000, (1, 64))
logits, kv_cache = model(idx)

# FlashTopos kernel — Lukasiewicz implication, O(1) SRAM memory
Q = torch.rand(1, 4096, 64).cuda()
K = torch.rand(1, 4096, 64).cuda()
attn = topos_ai.kernels.flash_topos_attention(Q, K)

# Sheaf consensus (two agents must agree within threshold)
ok, global_truth = topos_ai.math.sheaf_gluing(truth_A, truth_B, threshold=0.05)

# Transitive closure via categorical composition
R     = torch.rand(10, 10)
R_inf = topos_ai.math.transitive_closure(R, max_steps=5)
```

---

## Repository Structure

```
topos_ai/           # Core installable library
  logic.py          #   Heyting algebra (smooth Gödel implication)
  topology.py       #   Persistent homology — correct Betti numbers via boundary rank
  nn.py             #   TopologicalLinear, TopologicalNorm, MoE attention
  models.py         #   ToposTransformer end-to-end model
  math.py           #   Lukasiewicz / Gödel composition, sheaf gluing
  cohomology.py     #   Cech cohomology — H0 consensus, H1 obstruction
  kernels.py        #   Triton CUDA kernel (FlashTopos fwd + bwd)
  optim.py          #   ToposAdam (Fisher metric natural gradient)
  generation.py     #   Topologically constrained decoding (no hallucination mask)
  reasoning.py      #   Defeasible reasoning + autonomous theorem discovery
  verification.py   #   Lean 4 proof transpiler (Curry-Howard-Lambek)
  yoneda.py         #   Yoneda embedding & universe
  hott.py           #   Homotopy type theory (Procrustes path finding)

experiments/        # ~40 theoretical simulations (proof-of-concept)
applications/       # ~39 domain demos (finance, NLP, bioinformatics, seismic...)
benchmarks/         # Performance comparisons (tree search vs. attention, scaling laws)
tests/              # pytest test suite
docs/               # MkDocs documentation
```

---

## Benchmarks

| Script | What it measures |
|--------|-----------------|
| `benchmark_sota.py` | Ultrametric tree search vs. dense softmax attention |
| `scaling_laws_benchmark.py` | VRAM vs. sequence length for SRAM gradient accumulation |
| `babi_logic_benchmark.py` | bAbI Task 15 (logical reasoning) |
| `real_world_ontology_benchmark.py` | WordNet asymmetry vs. dot-product symmetry |

---

## Limitations

> This is an early-stage research framework. The following limitations apply:

- **No perplexity benchmarks yet.** ToposTransformer has not been compared to standard Transformers on real corpora.
- **MoE is simulated.** True sparse computation (only computing selected experts) requires a custom CUDA kernel. Current PyTorch implementation computes all experts, then masks — no actual FLOPs saving.
- **`experiments/` are simulations.** Scripts like `holographic_ads_cft_universe.py`, `recursive_self_improvement.py`, or `artificial_consciousness_phi.py` are narrative-driven theoretical explorations, not empirical results.
- **`TopologicalTokenizer` and `Distributed FSDP`** are scaffolding-stage. Multi-node training is not validated.
- **Attention is dot-product free; the output head is not.** The vocabulary projection uses cosine similarity (which is mathematically a normalized dot-product).

---

## Running Tests

```bash
# Fast CPU tests (no GPU required)
pytest tests/test_core.py tests/test_models.py -v -m "not cuda and not triton"

# Full suite with coverage
pytest --cov=topos_ai --cov-report=html
```

---

## Future Work

- **Lean 4 integration:** Training ToposAI as a formal theorem prover on Mathlib4 via the Curry-Howard-Lambek correspondence.
- **Perplexity benchmarks:** Rigorous comparison against standard Transformers on standard corpora.
- **True sparse MoE:** Custom CUDA kernel for genuine O(top-k / num_experts) compute.
- **∞-Topoi / HoTT:** Extending from discrete logits to continuous geometric manifolds.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use ToposAI in your research, please cite:

```bibtex
@software{toposai2024,
  author  = {Kurnaz, Salih Can},
  title   = {{ToposAI}: Neuro-Symbolic AI via Category Theory},
  year    = {2024},
  url     = {https://github.com/Tehlikeli107/ToposAI},
  license = {MIT}
}
```