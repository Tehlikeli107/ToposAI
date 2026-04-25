"""
ToposAI: neuro-symbolic research components bridging PyTorch with
category-theory-inspired operators.

Several modules are intentionally experimental toy/proxy implementations.
They should be read as differentiable research scaffolds unless their
docstrings state a stricter mathematical contract.
"""

import logging

try:
    from . import (
        adjoint,
        cohomology,
        distributed,
        elementary_topos,
        formal_category,
        generation,
        hott,
        infinity_categories,
        kan,
        kernels,
        lawvere_tierney,
        logic,
        mamba,
        math,
        models,
        monad,
        motives,
        nn,
        optics,
        optim,
        polynomial_functors,
        quantum_logic,
        reasoning,
        rl_killer,
        sheaf_dataloader,
        tame_geometry,
        tokenization,
        topology,
        verification,
        yoneda,
    )
except ImportError:
    # Allow pure Python formal mathematics imports when neural dependencies (e.g. PyTorch) are not installed.
    pass

__version__ = "1.0.0"
__license__ = "MIT"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "adjoint",
    "cohomology",
    "distributed",
    "elementary_topos",
    "formal_category",
    "generation",
    "hott",
    "infinity_categories",
    "kan",
    "kernels",
    "lawvere_tierney",
    "logic",
    "mamba",
    "math",
    "models",
    "monad",
    "motives",
    "nn",
    "optics",
    "optim",
    "polynomial_functors",
    "quantum_logic",
    "reasoning",
    "rl_killer",
    "sheaf_dataloader",
    "tame_geometry",
    "tokenization",
    "topology",
    "verification",
    "yoneda",
]
