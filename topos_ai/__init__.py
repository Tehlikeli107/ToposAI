"""
ToposAI: neuro-symbolic research components bridging PyTorch with
category-theory-inspired operators.

Several modules are intentionally experimental toy/proxy implementations.
They should be read as differentiable research scaffolds unless their
docstrings state a stricter mathematical contract.
"""

import logging
from importlib import import_module

from .formal_category import (
    FiniteCategory,
    FiniteFunctor,
    Presheaf,
    PresheafTopos,
    pullback_presheaf,
    whisker_transformation,
)
from .lazy.free_category import FreeCategoryGenerator
from .storage.cql_database import CategoricalDatabase
from .topology.sheaf_computer import ToposSheafComputer

_FORMAL_MODULES = (
    "formal_category",
    "hott",
    "infinity_categories",
)

_TORCH_BACKED_MODULES = (
    "adjoint",
    "cohomology",
    "distributed",
    "elementary_topos",
    "generation",
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
)


def _has_dependency(name):
    try:
        import_module(name)
        return True
    except (ImportError, OSError):
        return False


for _module_name in _FORMAL_MODULES:
    globals()[_module_name] = import_module(f"{__name__}.{_module_name}")

if _has_dependency("torch"):
    for _module_name in _TORCH_BACKED_MODULES:
        globals()[_module_name] = import_module(f"{__name__}.{_module_name}")

__version__ = "1.0.0"
__license__ = "MIT"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    *_FORMAL_MODULES,
    *(name for name in _TORCH_BACKED_MODULES if name in globals()),
    "CategoricalDatabase",
    "FiniteCategory",
    "FiniteFunctor",
    "FreeCategoryGenerator",
    "Presheaf",
    "PresheafTopos",
    "ToposSheafComputer",
    "pullback_presheaf",
    "whisker_transformation",
]
