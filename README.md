# ToposAI: Experimental Explorations in Neuro-Symbolic AI & Category Theory

[![ToposAI CI](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ToposAI** is an experimental open-source research framework aiming to bridge the gap between Continuous Deep Learning (PyTorch) and Formal Category Theory (Topos, Lukasiewicz Logic, Yoneda Lemma).

While modern LLMs operate purely on statistical dot-products (frequently suffering from hallucination in long-horizon reasoning), this project investigates whether **Logical Truth Values** and **Morphism Computations** can be successfully embedded into neural architectures.

---

## 🎯 Research Scope and Key Implementations

This repository serves as both a pip-installable framework (`topos_ai`) and a collection of 15+ proof-of-concept scripts validating category theory theorems on tensor operations:

1. **Logical Attention via Lukasiewicz MV-Algebra (`topos_ai.math`)**
   Replaces $Q \cdot K^T$ with differentiable implication: $\min(1, 1 - Q + K)$.
2. **O(1) VRAM Logical Scaling (`topos_ai.kernels.flash_topos`)**
   A custom C++/Triton kernel (`flash_topos_fixed.py`) performing $O(N^2)$ Lukasiewicz composition directly in SRAM, bypassing PyTorch's $O(N^2 \cdot D)$ memory explosion for long-context reasoning.
3. **Zero-Embedding Language Modeling (`topos_ai.nn.YonedaEmbedding`)**
   Eliminating traditional `nn.Embedding(vocab_size, dim)`. The meaning of a token is dynamically computed strictly via its morphisms to the rest of the vocabulary ($X \cong \text{Hom}(-, X)$).
4. **Sheaf Gluing Consensus (`topos_multi_agent_swarm.py`)**
   Mathematical resolution of conflicting local truths (Multi-Agent Swarm) without statistical averaging.
5. **Topological Constrained Decoding (`topos_ai.generation`)**
   Eliminating autoregressive hallucination by filtering next-token probabilities through the Topological Reachability Matrix. If an LLM statistically favors a token (e.g. memorization) but it lacks a formal morphism path, its logit is masked to $-\infty$.

## ⚠️ Limitations & Honest Positioning

As an early-stage research repository, ToposAI has notable limitations that must be acknowledged:

*   **Accuracy Drops in Dense Retrieval (`benchmark_sota.py`):** In large-scale vector similarity tests ($N=1M$), ToposAI’s "Hard Routing" tree search achieves significantly lower recall (~10-25%) compared to SOTA FAISS/HNSW models. Future implementation of Soft/Beam-Search tree routing is required to make this competitive.
*   **Thought Experiments vs. Production Code:** Scripts such as `godel_incompleteness_engine.py` and `hofstadter_topoi_sentience.py` are heavily narrative-driven *simulations* rather than production-ready AI layers. They serve to mathematically demonstrate concepts like Limit Cycles and Self-Reference but are not integrated into the main `ToposTransformer` yet.

## ⚙️ Installation

ToposAI is fully compatible with standard PyTorch workflows.

```bash
pip install pytest networkx matplotlib triton
pip install -e .
```

## 🔬 Quick Start

```python
import torch
import topos_ai

# Use the SRAM-optimized FlashTopos Kernel:
Q = torch.rand(1, 4096, 64).cuda()
K = torch.rand(1, 4096, 64).cuda()

# Exact logical composition in milliseconds
attention_matrix = topos_ai.kernels.flash_topos_attention(Q, K)

# Consensus engine using Sheaf Condition
is_compatible, global_truth = topos_ai.math.sheaf_gluing(Universe_A, Universe_B)

# Using Zero-Embedding ToposTransformer
model = topos_ai.models.ToposTransformer(vocab_size=50000)
```

## 🌌 Future Work (The Ultimate Horizon)
*   **Integration with Lean 4:** Leveraging the *Curry-Howard-Lambek correspondence* to train ToposAI as a formal Automated Theorem Prover on `Mathlib4`.
*   **$\infty$-Topoi and Homotopy Type Theory (HoTT):** Extending the concept bender (`hott_concept_bender.py`) from discrete logits to continuous geometric manifolds for scientific discovery (e.g., Quantum Chemistry).

*Contributions, mathematical critiques, and pull requests are highly welcomed. The path to AGI requires rigorous, formal logic.*