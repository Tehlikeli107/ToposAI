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
from .monoidal import (
    FiniteMonoidalCategory,
    FiniteSymmetricMonoidalCategory,
    strict_monoidal_from_monoid,
)
from .enriched import FiniteEnrichedCategory, discrete_enriched_category
from .formal_kan import (
    FiniteSetFunctor,
    left_kan_extension,
    right_kan_extension,
    all_natural_transformations,
    verify_left_kan_universal_property,
    verify_right_kan_universal_property,
    left_kan_unit,
)
from .sites import (
    Sieve,
    GrothendieckTopology,
    GrothendieckSite,
    FinitePresheaf,
    is_sheaf,
    sheaf_condition_failure,
    matching_families,
    amalgamations,
    maximal_sieve,
    trivial_topology,
    discrete_topology,
    omega_presheaf,
)
from .adjunction import FiniteAdjunction
from .formal_yoneda import (
    representable_functor,
    yoneda_evaluate,
    yoneda_inverse,
    verify_yoneda,
    verify_yoneda_naturality_in_A,
)
from .topos import (
    finset_product,
    finset_exponential,
    curry,
    uncurry,
    verify_ccc,
    SubobjectClassifier,
    verify_subobject_classifier,
)
from .lean4_export import (
    category_to_lean4,
    functor_to_lean4,
    monoidal_to_lean4,
    nat_trans_to_lean4,
    export_to_file,
)
from .lazy.free_category import FreeCategoryGenerator
from .storage.cql_database import CategoricalDatabase
from .topology.sheaf_computer import ToposSheafComputer

_FORMAL_MODULES = (
    "adjunction",
    "enriched",
    "formal_category",
    "formal_kan",
    "formal_yoneda",
    "hott",
    "infinity_categories",
    "lean4_export",
    "monoidal",
    "sites",
    "topos",
)

_TORCH_BACKED_MODULES = (
    "adjoint",
    "cohomology",
    "distributed",
    "sheaf_nn",
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
    "FiniteAdjunction",
    "FiniteCategory",
    "FiniteFunctor",
    "FiniteMonoidalCategory",
    "FiniteSymmetricMonoidalCategory",
    "FreeCategoryGenerator",
    "Presheaf",
    "PresheafTopos",
    "SubobjectClassifier",
    "ToposSheafComputer",
    "discrete_enriched_category",
    "FiniteEnrichedCategory",
    "category_to_lean4",
    "curry",
    "export_to_file",
    "finset_exponential",
    "finset_product",
    "functor_to_lean4",
    "monoidal_to_lean4",
    "nat_trans_to_lean4",
    "pullback_presheaf",
    "representable_functor",
    "strict_monoidal_from_monoid",
    "uncurry",
    "verify_ccc",
    "verify_subobject_classifier",
    "verify_yoneda",
    "verify_yoneda_naturality_in_A",
    "whisker_transformation",
    "yoneda_evaluate",
    "yoneda_inverse",
    # formal_kan
    "FiniteSetFunctor",
    "left_kan_extension",
    "right_kan_extension",
    "all_natural_transformations",
    "verify_left_kan_universal_property",
    "verify_right_kan_universal_property",
    "left_kan_unit",
    # sites
    "Sieve",
    "GrothendieckTopology",
    "GrothendieckSite",
    "FinitePresheaf",
    "is_sheaf",
    "sheaf_condition_failure",
    "matching_families",
    "amalgamations",
    "maximal_sieve",
    "trivial_topology",
    "discrete_topology",
    "omega_presheaf",
]
