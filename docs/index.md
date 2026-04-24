# ToposAI

**ToposAI** is an experimental open-source research framework that bridges continuous deep learning (PyTorch) with formal category theory — Toposes, Sheaves, the Yoneda Lemma, and Heyting algebra.

## Motivation

Modern LLMs operate on statistical dot-products and suffer from hallucination in long-horizon reasoning. ToposAI investigates whether **logical truth values** and **morphism computations** can be embedded into neural architectures to produce more structured, verifiable reasoning.

## Key Ideas

| Concept | Classical DL | ToposAI |
|---------|-------------|---------|
| Attention scoring | Softmax(QKᵀ / √d) | Lukasiewicz T-norm (dot-product free) |
| Weights | Real, unconstrained | [0, 1] via sigmoid (morphism strength) |
| Residual | x + f(x) | x + f(x) - x·f(x) (T-conorm) |
| Output head | Linear classifier | Cosine similarity (topological reachability) |

## Quick Start

```bash
pip install -e .
```

```python
import torch
import topos_ai

model = topos_ai.models.ToposTransformer(vocab_size=50000, d_model=256, num_universes=8)
idx   = torch.randint(0, 50000, (1, 64))
logits, kv_cache = model(idx)

R     = torch.rand(10, 10)
R_inf = topos_ai.math.transitive_closure(R, max_steps=5)
```

## Honest Limitations

- Attention scoring is dot-product free; the output vocabulary projection uses cosine similarity.
- Scripts in `experiments/` are theoretical simulations, not empirical benchmarks.
- MoE sparsity is simulated — true sparse computation requires a custom CUDA kernel.
- No perplexity benchmarks against standard LMs have been run yet.

## Navigation

- [API Reference](api.md)
- [GitHub Repository](https://github.com/Tehlikeli107/ToposAI)
- [Contributing](../CONTRIBUTING.md)
