# ToposAI

**ToposAI** is an experimental open-source research framework that bridges
continuous deep learning (PyTorch) with category-theory-inspired operators:
toposes, sheaf-style consistency, Yoneda-style reconstruction, and Heyting
algebra.

The core library now separates two layers: formal finite-category utilities
for categories, presheaves, Yoneda, sieves, and subobject classifiers; and
neural proxy modules that use those ideas inside PyTorch experiments.

## Research Library Guides

- [Mathematical Contracts](MATH_CONTRACTS.md)
- [Public API Contract](PUBLIC_API.md)
- [API Reference](api.md)

## Motivation

Modern LLMs operate mostly on statistical similarity and can struggle with
long-horizon consistency. ToposAI investigates whether **logical truth values**
and **morphism-like computations** can be embedded into neural architectures to
produce more structured, auditable reasoning experiments.

## Key Ideas

| Concept | Classical DL | ToposAI |
|---------|-------------|---------|
| Attention scoring | Softmax(QK^T / sqrt(d)) | Goedel-Heyting implication (dot-product free scoring) |
| Weights | Real, unconstrained | [0, 1] via sigmoid (morphism strength) |
| Residual | x + f(x) | x + f(x) - x * f(x) (T-conorm) |
| Output head | Linear classifier | Cosine similarity (topological reachability) |

## Quick Start

```bash
pip install -e .
```

```python
import torch
import topos_ai

model = topos_ai.models.ToposTransformer(vocab_size=50000, d_model=256, num_universes=8)
idx = torch.randint(0, 50000, (1, 64))
logits, kv_cache = model(idx)

R = torch.rand(10, 10)
R_inf = topos_ai.math.transitive_closure(R, max_steps=5)
```

## Honest Limitations

- Attention scoring is dot-product free; the output vocabulary projection uses cosine similarity.
- Scripts in `experiments/` are theoretical simulations, not empirical benchmarks.
- MoE sparsity is simulated; true sparse computation requires a custom CUDA kernel.
- No perplexity benchmarks against standard language models have been run yet.

## Navigation

- [API Reference](api.md)
- [Project Status](PROJECT_STATUS.md)
- [Benchmark Reporting](BENCHMARKS.md)
- [Release Guide](RELEASE_GUIDE.md)
- [GitHub Repository](https://github.com/Tehlikeli107/ToposAI)
- [Contributing](../CONTRIBUTING.md)
