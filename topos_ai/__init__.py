"""
ToposAI: A neuro-symbolic AI framework bridging Category Theory with PyTorch.

Core modules:
    models               - ToposTransformer end-to-end language model
    nn                   - Building blocks (TopologicalLinear, ToposAttention, etc.)
    logic                - Heyting algebra & Gödel T-norms (SubobjectClassifier)
    math                 - Categorical composition & sheaf operators
    topology             - Persistent homology (Betti numbers)
    cohomology           - Cech cohomology engine
    kernels              - Triton CUDA kernels (FlashTopos attention)
    optim                - ToposAdam optimizer (Fisher natural gradient)
    generation           - Topologically constrained decoding
    reasoning            - Defeasible reasoning & theorem discovery
    verification         - Lean 4 proof transpiler
    yoneda               - Yoneda embedding & universe
    hott                 - Homotopy type theory utilities
    adjoint              - Adjoint functor pairs (F ⊣ G), unit/counit, triangle identities
    monad                - Monads (Giry, Continuation, Writer) & Kleisli category
    kan                  - Left/Right Kan extensions (categorical attention)
    optics               - Lenses, Prisms, Traversals, Van Laarhoven representation
    polynomial_functors  - Poly category: dynamical systems & wiring diagrams
"""

import logging

from . import (
    adjoint,
    cohomology,
    distributed,
    generation,
    hott,
    kan,
    kernels,
    logic,
    math,
    models,
    monad,
    nn,
    optics,
    optim,
    polynomial_functors,
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
    "adjoint",
    "cohomology",
    "distributed",
    "generation",
    "hott",
    "kan",
    "kernels",
    "logic",
    "math",
    "monad",
    "models",
    "nn",
    "optics",
    "optim",
    "polynomial_functors",
    "reasoning",
    "tokenization",
    "topology",
    "verification",
    "yoneda",
]
