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

__version__ = "1.0.0"
__author__ = "Salih Can Kurnaz"
__email__ = "salihcankurnaz@gmail.com"
__license__ = "MIT"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import math
from . import nn
from . import models
from . import kernels
from . import logic
from . import generation
from . import topology
from . import cohomology
from . import reasoning
from . import verification
from . import tokenization
from . import distributed
from . import optim
from . import yoneda
from . import hott

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
