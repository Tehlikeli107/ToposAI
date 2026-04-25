# Mathematical Contracts

ToposAI has two kinds of modules: formal finite mathematics and neural or proxy research components. Formal modules expose explicit finite objects and validate algebraic laws. Proxy modules are useful experimental scaffolds but do not claim to implement the full mathematical theory named in their inspiration.

## Formal finite mathematics

| Module | Contract | Main limitations |
|--------|----------|------------------|
| `topos_ai.formal_category.FiniteCategory` | Explicit finite category with typed morphisms, identities, composition, identity laws, and associativity validation. | Finite categories only. No enriched, large, or higher categories. |
| `topos_ai.formal_category.PresheafTopos` | Finite fragment of `Set^(C^op)` with finite limits, colimits, exponentials, subobject classifier, Heyting operations, Kripke-Joyal forcing, quantifiers, sheafification, Lawvere-Tierney operators, and Yoneda density reconstruction. | Computes finite presheaf topoi. It is not a general theorem prover for arbitrary Grothendieck topoi. |
| `topos_ai.formal_category.GrothendieckTopology` | Finite Grothendieck topology with maximal-sieve, pullback-stability, and transitivity checks. | Covering data must be explicitly finite. |
| `topos_ai.infinity_categories.FiniteSimplicialSet` | Finite simplicial-set skeleton with face identities, optional degeneracy identities, horn enumeration, and finite inner-Kan checks. | Skeleton-limited. It checks enumerated finite horns up to supplied dimension. |
| `topos_ai.infinity_categories.nerve_3_skeleton` | Builds the 3-skeleton of the nerve of a finite category, including 2-horn composition and 3-simplex associativity coherence. | Not a complete infinity-category implementation beyond the represented skeleton. |
| `topos_ai.hott.FinitePathGroupoid` | 1-truncated HoTT identity-type semantics as a finite groupoid with reflexivity, inverse paths, composition, and associativity. | Models groupoid semantics. It is not a dependent type checker. |
| `topos_ai.hott.PathFamily` | Dependent family over a finite path groupoid with functorial transport validation. | Transport maps are finite and explicit. |

## Neural and proxy research components

| Module | Intended use | Non-claim |
|--------|--------------|-----------|
| `topos_ai.hott.HomotopyEquivalence` | Orthogonal Procrustes alignment for point clouds. | HomotopyEquivalence is not a HoTT proof kernel. |
| `topos_ai.infinity_categories.InfinityCategoryLayer` | Hodge message passing over finite simplicial complexes. | InfinityCategoryLayer is not a full infinity-category engine. |
| `topos_ai.yoneda.YonedaUniverse` | Probe-distance reconstruction experiment inspired by Yoneda-style observation. | It is not the categorical Yoneda lemma. Use `topos_ai.formal_category.yoneda_lemma_bijection` and `yoneda_density_colimit` for finite categorical Yoneda computations. |
| `topos_ai.logic.SubobjectClassifier` | Goedel-Heyting fuzzy logic layer for neural experiments. | It is a differentiable finite-valued algebra, not the subobject classifier of an arbitrary topos. |

## Publication language

Use precise claims:

- "finite presheaf topos computation" instead of "general topos engine"
- "finite quasi-category horn checks" instead of "full infinity-category implementation"
- "finite groupoid semantics for identity types" instead of "complete HoTT kernel"
- "neural proxy inspired by category theory" for differentiable modules that do not validate formal laws

## Verification policy

Formal modules must include tests for their defining laws. Proxy modules must include boundedness, shape, stability, or smoke tests and must document the mathematical non-claims above.