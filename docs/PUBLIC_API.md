# Public API Contract

This document defines the stable research-facing API surface for ToposAI. These names should remain importable across patch releases unless a release note announces a breaking change.

## Formal category and topos core

Import from `topos_ai.formal_category`:

- `FiniteCategory`
- `FiniteFunctor`
- `Presheaf`
- `NaturalTransformation`
- `FrozenNaturalTransformation`
- `Subpresheaf`
- `GrothendieckTopology`
- `PresheafTopos`
- `natural_transformations`
- `representable_presheaf`
- `yoneda_element_to_transformation`
- `yoneda_transformation_to_element`
- `yoneda_lemma_bijection`
- `category_of_elements`
- `yoneda_density_colimit`

Stable `PresheafTopos` method families include finite limits and colimits,
subobject classifiers, exponentials, sheafification, Lawvere-Tierney
operators, internal quantifiers, finite Kan extensions, and the explicit
Kan-adjunction witnesses `left_kan_transpose`, `left_kan_untranspose`,
`right_kan_transpose`, `right_kan_untranspose`, `left_kan_unit`,
`left_kan_counit`, `right_kan_unit`, `right_kan_counit`,
`validate_left_kan_adjunction`, and `validate_right_kan_adjunction`.
The finite universal-property validators for products, pullbacks,
equalizers, coproducts, coequalizers, exponentials, and subobject classifiers
are also part of the research-facing formal API.
Exactness validators such as `validate_regular_image_factorization` and
`validate_effective_epimorphism` are stable formal-core methods.

## Finite simplicial and quasi-category core

Import from `topos_ai.infinity_categories`:

- `FiniteHorn`
- `FiniteSimplicialSet`
- `nerve_2_skeleton`
- `nerve_3_skeleton`
Stable formal methods include `missing_inner_horns`, `is_inner_kan`, and
`has_unique_inner_horn_fillers`.
- `SimplicialComplexBuilder`
- `HodgeLaplacianEngine`
- `InfinityCategoryLayer`

## HoTT finite groupoid core

Import from `topos_ai.hott`:

- `FinitePathGroupoid`
- `PathFamily`
- `HomotopyEquivalence`
Stable formal methods include `transport_equivalence` and
`validate_transport_equivalences` for finite dependent families.

## Stability expectations

- Constructors should continue validating mathematical laws at construction time.
- Functions that return natural transformations should return validated transformations.
- Finite examples in `examples/` should run without network access.
- Experimental modules may evolve faster, but their non-claims must stay documented in `docs/MATH_CONTRACTS.md`.
