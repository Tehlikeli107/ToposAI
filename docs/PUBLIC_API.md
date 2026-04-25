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

## Finite simplicial and quasi-category core

Import from `topos_ai.infinity_categories`:

- `FiniteHorn`
- `FiniteSimplicialSet`
- `nerve_2_skeleton`
- `nerve_3_skeleton`
- `SimplicialComplexBuilder`
- `HodgeLaplacianEngine`
- `InfinityCategoryLayer`

## HoTT finite groupoid core

Import from `topos_ai.hott`:

- `FinitePathGroupoid`
- `PathFamily`
- `HomotopyEquivalence`

## Stability expectations

- Constructors should continue validating mathematical laws at construction time.
- Functions that return natural transformations should return validated transformations.
- Finite examples in `examples/` should run without network access.
- Experimental modules may evolve faster, but their non-claims must stay documented in `docs/MATH_CONTRACTS.md`.