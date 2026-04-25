# API Reference

## `topos_ai.models`

### `ToposTransformer(vocab_size, d_model=64, num_universes=4, num_layers=2, max_seq_len=2048)`

End-to-end trainable categorical language model. Attention scoring uses
Goedel-Heyting implication rather than a raw dot product. The output projection
uses cosine similarity.

```python
model = ToposTransformer(vocab_size=50000, d_model=256, num_universes=8, num_layers=6)
logits, kv_cache = model(idx)
logits, kv_cache = model(idx, kv_caches)
```

---

## `topos_ai.kernels`

### `flash_topos_attention(q, k)`

Computes Goedel-Heyting implication over the feature dimension:
`1 if Q <= K else K`. Uses Triton CUDA when available and falls back to pure
PyTorch otherwise, including CPU tensors when Triton is installed.

---

## `topos_ai.formal_category`

For the boundary between formal finite mathematics and neural/proxy components, see [Mathematical Contracts](MATH_CONTRACTS.md).

### `FiniteCategory(objects, morphisms, identities, composition)`

Explicit small finite category. It validates identity laws, composition typing,
and associativity. Composition uses `compose(g, f) = g o f`.

### `Presheaf(category, sets, restrictions)`

Finite presheaf `C^op -> FinSet`. For a morphism `f: A -> B`, the restriction
map is `F(f): F(B) -> F(A)`. Construction checks identity and contravariant
functor laws.

### `FiniteFunctor(source, target, object_map, morphism_map)`

Explicit functor between finite categories. Construction validates source/target
typing, identity preservation, and composition preservation.

### `representable_presheaf(category, obj)`

Builds the Yoneda presheaf `y(obj) = Hom(-, obj)`.

### `yoneda_element_to_transformation(category, obj, presheaf, element)`

Maps `x in F(obj)` to the natural transformation `y(obj) -> F` whose component
at `A` sends `h: A -> obj` to `F(h)(x)`.

### `natural_transformations(source, target)`

Enumerates all natural transformations between two finite presheaves by
checking each component family against the naturality squares.

### `yoneda_lemma_bijection(category, obj, presheaf)`

Enumerates the finite Yoneda bijection `Nat(y(obj), F) ~= F(obj)` and returns
a dictionary from elements of `F(obj)` to their corresponding natural
transformations.

### `category_of_elements(presheaf)`

Builds the finite category of elements `int F` for a presheaf `F`, together
with the canonical projection functor `int F -> C`.

### `yoneda_density_colimit(presheaf)`

Reconstructs `F` as the finite colimit of representables over `int F`, returning
the density presheaf and mutually inverse natural transformations between that
colimit and `F`.

### `GrothendieckTopology(category, covering_sieves)`

Finite Grothendieck topology on a small category, represented by covering
sieves. Construction validates the maximal-sieve, pullback-stability, and
transitivity axioms.

### `PresheafTopos(category)`

Finite fragment of the presheaf topos `Set^(C^op)`. Provides sieves, pullback
of sieves, the subobject classifier `Omega`, characteristic maps for
subpresheaves, and pullback along truth.

It also implements finite elementary-topos structure:

- `initial_presheaf()` - initial object, pointwise empty set.
- `terminal_presheaf()` - terminal object, pointwise singleton.
- `product_presheaf(F, G)` - pointwise product with both projections.
- `coproduct_presheaf(F, G)` - pointwise tagged coproduct with injections.
- `reindex_presheaf(u, F)` / `reindex_transformation(u, alpha)` - inverse
  image of presheaves and natural transformations along a finite functor
  `u: C -> D`.
- `left_kan_extension_presheaf(u, F)` / `right_kan_extension_presheaf(u, F)` -
  finite Kan extensions giving the adjoint triple `Sigma_u -| u* -| Pi_u`.
- `compose_transformations(beta, alpha)` - natural transformation composition
  `beta o alpha`.
- `is_monomorphism(alpha)` / `is_epimorphism(alpha)` - objectwise mono/epi
  predicates for finite presheaves.
- `matching_families(F, c, S)` / `amalgamations(F, c, family)` - finite
  covering-sieve gluing data.
- `is_separated(F, J)` / `is_sheaf(F, J)` - separated and sheaf conditions for
  a finite Grothendieck topology.
- `plus_construction(F, J)` - finite plus construction: quotient covering-sieve
  matching families by local equality.
- `sheafification(F, J)` - associated sheaf via the finite plus-plus
  construction.
- `extend_to_plus(alpha, J)` / `sheafification_factorization(alpha, J)` -
  reflector factorization of maps from presheaves into sheaves.
- `lawvere_tierney_operator(J)` - local operator `j: Omega -> Omega` induced
  by a Grothendieck topology.
- `validate_lawvere_tierney_axioms(J)` - finite check of `j(true)=true`,
  idempotence, and finite-meet preservation.
- `topology_from_lawvere_tierney_operator(j)` - recover covering sieves as
  those with `j(S) = true`.
- `omega_j(J)` - J-closed-sieve classifier for the sheaf topos.
- `truth_map_j(J)` / `characteristic_map_j(S, J)` / `pullback_truth_j(chi, J)`
  - sheaf-topos truth and J-modal characteristic maps.
- `sieve_j_meet(J, c, S, T)` / `sieve_j_join(J, c, S, T)` /
  `sieve_j_implication(J, c, S, T)` - Heyting operations in `Omega_J(c)`.
- `validate_omega_j_heyting_laws(J)` - finite Heyting adjunction checks for
  the J-closed-sieve classifier.
- `subobject_closure(S, J)` - J-closure of a subobject via the local operator.
- `is_j_closed_subobject(S, J)` / `is_dense_subobject(S, J)` - closure-fixed
  and dense subobject predicates.
- `subobjects(F)` / `j_closed_subobjects(F, J)` - finite subobject enumeration.
- `subobject_j_meet(A, B, J)` / `subobject_j_join(A, B, J)` /
  `subobject_j_implication(A, B, J)` / `subobject_j_negation(A, J)` - internal
  Heyting algebra operations for J-closed subobjects.
- `validate_j_subobject_heyting_laws(F, J)` - finite adjunction-law check for
  the J-closed subobject lattice.
- `pullback(alpha, beta)` - fiber product of maps with common codomain.
- `equalizer(alpha, beta)` - subpresheaf equalizer of parallel transformations.
- `subpresheaf_object(S)` / `subpresheaf_inclusion(S)` - represent a subobject
  as a presheaf and its canonical monomorphism.
- `inverse_image(alpha, S)` - pull back a subobject along a transformation.
- `forcing_sieve(S, c, x)` / `forces(S, c, x)` - Kripke-Joyal forcing
  semantics for membership in a subobject.
- `truth_value(S)` - global truth value of a closed proposition `S <= 1`.
- `diagonal_transformation(F)` / `equality_subobject(F)` - internal equality
  predicate as the diagonal `F -> F x F`.
- `equality_truth(F, c, x, y)` - Kripke-Joyal truth value of `x = y`.
- `exists_along(alpha, S)` / `forall_along(alpha, S)` - internal existential
  and universal quantification along a map.
- `validate_quantifier_adjunctions(alpha)` - finite check of
  `exists_alpha -| alpha* -| forall_alpha`.
- `validate_frobenius_reciprocity(alpha)` - finite check of
  `exists_alpha(S meet alpha*T) = exists_alpha(S) meet T`.
- `validate_beck_chevalley(alpha, beta)` - finite Beck-Chevalley checks for
  existential and universal quantifiers over the pullback of `alpha` and `beta`.
- `exponential_presheaf(F, G)` - exponential `G^F` using
  `(G^F)(c) = Nat(y(c) x F, G)`.
- `evaluation_map(F, G)` - evaluation morphism `G^F x F -> G`.
- `transpose(alpha, H, F, G)` - curries `alpha: H x F -> G` into
  `H -> G^F`.
- `power_object(F)` - power object `P(F) = Omega^F`.
- `membership_relation(F)` - internal membership subobject `in <= F x P(F)`.
- `truth_map()` - truth morphism `1 -> Omega`, selecting maximal sieves.
- `name_subobject(S)` / `extension_of_name(F, name)` - subobject classifier
  naming and extension round-trip.
- `image(alpha)` - image subpresheaf of a natural transformation.
- `image_factorization(alpha)` - factor `alpha` as epi followed by mono.
- `kernel_pair(alpha)` - pullback of a map along itself.
- `coequalizer(alpha, beta)` - pointwise quotient of parallel transformations.
- `subobject_meet(A, B)` / `subobject_join(A, B)` - subobject lattice operations.
- `subobject_implication(A, B)` / `subobject_negation(A)` - Heyting implication
  and pseudocomplement for subobjects.

---

## `topos_ai.math`

### `godel_composition(R1, R2)`

Matrix composition using the Goedel/min t-norm:
`max_k min(R1[i,k], R2[k,j])`.

### `soft_godel_composition(R1, R2, tau=10.0)`

Exact Goedel composition in the forward pass with a smooth log-sum-exp backward
path for neural experiments.

### `lukasiewicz_composition(R1, R2)`

Optional comparison operator using Lukasiewicz T-norm:
`max_k clamp(R1[i,k] + R2[k,j] - 1, 0)`.

### `transitive_closure(R, max_steps=5, composition="godel")`

Computes the reachability closure of a relation matrix. The default composition
is Goedel/max-min; pass `composition="lukasiewicz"` for the comparison variant.

### `sheaf_gluing(truth_A, truth_B, threshold=0.05)`

Returns `(True, global_section)` if `max|A - B| <= threshold`, else `(False, None)`.

---

## `topos_ai.logic`

### `SubobjectClassifier`

Implements a Goedel-Heyting algebra with exact forward values:

- `implies(A, B)` - `1 if A <= B else B`, with a smooth custom backward.
- `logical_not(A)` - Heyting pseudocomplement, `A => false`.
- `logical_and(A, B)` - `min(A, B)`.
- `logical_or(A, B)` - `max(A, B)`.

### `HeytingNeuralLayer(in_features, out_features)`

A neural layer where each output neuron computes `AND_i (x_i => w_ji)` via
vectorized broadcasting.

---

## `topos_ai.infinity_categories`

### `FiniteSimplicialSet(simplices, faces, degeneracies=None)`

Finite simplicial-set skeleton with explicit face maps. It validates the
simplicial face identities, validates degeneracy identities when degeneracies
are supplied, and can enumerate compatible horns.

### `FiniteHorn(dimension, missing_face, faces)`

Finite horn `Lambda^n_k -> X`, represented by all boundary faces except the
missing face `k`.

### `nerve_2_skeleton(category)`

Builds the 2-skeleton of the nerve of a finite category. Inner 2-horn fillers
encode ordinary categorical composition.

### `nerve_3_skeleton(category)`

Builds the 3-skeleton of the nerve of a finite category. Its 3-simplices encode
associativity coherence for triples of composable arrows, including the two
middle faces that compare the two parenthesizations.

### `FiniteSimplicialSet.is_inner_kan(max_dimension=None)`

Finite quasi-category check: enumerates inner horns up to the requested
dimension and verifies that each has at least one filler.

### `SimplicialComplexBuilder`, `HodgeLaplacianEngine`, `InfinityCategoryLayer`

Point-cloud simplicial-complex and Hodge message-passing utilities. These are
neural/simplicial operators; the explicit `FiniteSimplicialSet` API is the
formal horn-filler layer.

---

## `topos_ai.hott`

### `FinitePathGroupoid(objects, paths, identities, inverses, composition)`

Finite groupoid semantics for 1-truncated HoTT identity types. Objects are
terms/types-as-points, paths are identity proofs, and composition is validated
against identity, inverse, and associativity laws.

### `PathFamily(base, fibers, transports)`

Dependent family over a finite path groupoid. Transport maps are checked as a
functor from the path groupoid into finite sets, including identity transport
and composition preservation.

### `HomotopyEquivalence`

Orthogonal Procrustes alignment for point clouds. This remains a numerical
homotopy-equivalence-inspired tool, separate from the formal finite groupoid
identity-type layer.

---

## `topos_ai.topology`

### `PersistentHomology(num_nodes)`

#### `calculate_betti(distance_matrix, threshold) -> (beta0, beta1)`

Builds a Vietoris-Rips complex at `threshold` and computes Betti numbers using
the boundary matrix rank for `beta1`.

---

## `topos_ai.optim`

### `ToposAdam(params, lr=1e-3, betas=(0.9, 0.999), topological_weight_decay=0.01)`

Adam variant with Fisher information scaling for parameters constrained to the
sigmoid manifold. Gradient amplification is clamped to prevent instability.

---

## `topos_ai.generation`

### `ToposConstrainedDecoder(reachability_matrix, threshold=0.1)`

Masks next-token logits using a categorical reachability matrix, then samples
via `torch.multinomial`.

---

## `topos_ai.cohomology`

### `CechCohomology(num_nodes, edges)`

Computes H0 consensus disagreement and H1 obstruction magnitude for distributed
systems using boundary operators.

---

## `topos_ai.verification`

### `Lean4VerificationBridge(entities)`

Transpiles categorical reasoning chains into Lean 4 theorem syntax and
optionally runs the Lean compiler for formal verification.
