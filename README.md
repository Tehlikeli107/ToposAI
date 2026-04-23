# ToposAI: Experimental Explorations in Neuro-Symbolic AI & Category Theory

[![ToposAI CI](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Tehlikeli107/ToposAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ToposAI** is an experimental open-source research framework aiming to bridge the gap between Continuous Deep Learning (PyTorch) and Formal Category Theory (Topos, Lukasiewicz Logic, Yoneda Lemma).

While modern LLMs operate purely on statistical dot-products (frequently suffering from hallucination in long-horizon reasoning), this project investigates whether **Logical Truth Values** and **Morphism Computations** can be successfully embedded into neural architectures.

---

## 🎯 Research Scope and Key Implementations

This repository serves as both a pip-installable framework (`topos_ai`) and a collection of 50+ proof-of-concept scripts (~39 applications, 10 benchmarks, ~40 experiments) validating category theory theorems on tensor operations, organized cleanly into modules:

### 🔬 Benchmarks (`benchmarks/`)
*   **`benchmark_sota.py`**: Hardware scale tree search vs Dense Attention.
*   **`scaling_laws_benchmark.py`**: VRAM measurement proving O(1) SRAM gradient accumulation within VRAM limits.
*   **`babi_logic_benchmark.py`**: Meta bAbI Task 15 evaluation.
*   **`real_world_ontology_benchmark.py`**: Asymmetry tests on NLTK WordNet vs Dot-Product.

### 🚀 Applications (`applications/`)
*   **`app.py`**: Gradio Web Dashboard for interactive Neuro-Symbolic reasoning.
*   **`real_world_medical_fact_checker.py`**: RAG-killer resolving Adverse Drug Reactions.
*   **`real_world_solidity_auditor.py`**: Topological formal verification for Smart Contracts (Reentrancy).
*   **`real_world_finance_topos.py`**: Systemic risk contagion using live S&P500 `yfinance` data.
*   **`real_world_seismic_topos.py`**: Spatiotemporal causal discovery using USGS earthquake data.
*   **`dynamic_ontology_nlp.py`**: Text-to-Topos builder and reasoner.

### 🧪 Experiments (`experiments/`)
*   Contains deeply theoretical simulations like **Homotopy Type Theory (`hott_concept_bender.py`)**, **Temporal Topoi (`temporal_topos_retrocausality.py`)**, **Gödel Incompleteness Engine**, and **Recursive Self-Improvement (Singularity)**.

## ⚠️ Limitations & Honest Positioning

As an early-stage research repository, ToposAI has notable limitations that must be acknowledged:

*   **Trade-offs in Dense Retrieval Routing:** In large-scale vector similarity tests, ToposAI abandons computationally expensive dot-products for hierarchical tree search.
*   **Heuristic Demos & Theoretical Claims:** Scripts within `experiments/` (e.g., Reversible Computing, Holographic Universe, Consciousness scores) and `applications/` (e.g., Cosmology, Epidemiology Minimum Cut, LLM Chat interfaces) are heavily narrative-driven *simulations* (Proof-of-Concepts) rather than production-ready empirical laws. They demonstrate *Categorical Tendencies* and *Heuristic Topologies* rather than deterministic zero-shot solutions. (e.g., "Zero-Joule AI" claims are strictly theoretical frameworks).
*   **API Scaffolding:** Modules like `TopologicalTokenizer` and `Distributed FSDP` are currently at the scaffolding stage. While they compile and demonstrate Top-K MoE logic and causal emergence, they require substantial real-world data and multi-node hardware clusters to validate their "trillion-parameter" claims.

## ⚙️ Installation

ToposAI is fully compatible with standard PyTorch workflows. The `setup.py` automatically resolves heavy dependencies (transformers, datasets, yfinance, networkx).

```bash
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