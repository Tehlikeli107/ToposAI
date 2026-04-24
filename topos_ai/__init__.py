"""
ToposAI: A neuro-symbolic AI framework bridging Category Theory with PyTorch.

Core modules:
    models      - ToposTransformer end-to-end language model
    nn          - Building blocks (TopologicalLinear, ToposAttention, etc.)
    logic       - Heyting algebra & Gödel T-norms (SubobjectClassifier)
    math        - Categorical composition & sheaf operators
    topology    - Persistent homology (Betti numbers)
    cohomology  - Cech cohomology engine
    kernels     - Triton CUDA kernels (FlashTopos attention)
    optim       - ToposAdam optimizer (Fisher natural gradient)
    generation  - Topologically constrained decoding
    reasoning   - Defeasible reasoning & theorem discovery
    verification- Lean 4 proof transpiler
    yoneda      - Yoneda embedding & universe
    hott        - Homotopy type theory utilities
"""

import logging

from . import (
    cohomology,
    distributed,
    generation,
    hott,
    kernels,
    logic,
    math,
    models,
    nn,
    optim,
    reasoning,
    tokenization,
    topology,
    verification,
    yoneda,
)

__version__ = "1.0.0"
__license__ = "MIT"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "math",
    "nn",
    "models",
    "kernels",
    "logic",
    "generation",
    "topology",
    "cohomology",
    "reasoning",
    "verification",
    "tokenization",
    "distributed",
    "optim",
    "yoneda",
    "hott",
]
