# Mathematical Contracts

ToposAI acts as a bridge between pure Category Theory and Deep Learning heurism. The laboratory has two strict boundaries: 
1. **Formal Modules (The Math Core):** These modules strictly enforce Category Laws (Associativity, Composition, Identity).
2. **Neural / Proxy Modules (The Heuristics):** These modules use continuous variables (Floats, Neural Networks) and compute optimizations or alignments. They are inspired by Category Theory but do NOT represent formal mathematical proofs.

## Formal Mathematics & Scalable Topos Engines

| Module | Contract | Scientific Boundary (Honesty) |
|--------|----------|-------------------------------|
| `topos_ai.formal_category.FiniteCategory` | Strict finite category with $O(N^3)$ transitivity closure. | Exact and rigid. Crashes out of memory (OOM) if nodes exceed manageable limits due to global verification. |
| `topos_ai.formal_category.PresheafTopos` | Formal Kripke-Joyal Truth logic ($\Omega$). | Operates flawlessly on discrete mappings, turning exception handling into mathematical theorem generation. |
| `topos_ai.storage.CategoricalDatabase` | Disk-based (CQL) SQLite B-Tree engine. | Eliminates RAM bottlenecks. However, massive writes run into Disk I/O bottlenecks unless OS syncs (PRAGMA) are dangerously disabled. |
| `topos_ai.lazy.FreeCategoryGenerator` | Trivial pathfinder (Lazy evaluation) simulating categorical composability. | Achieves zero-RAM closure by bypassing full universe computation, resolving queries dynamically (BFS style). |
| `topos_ai.topology.ToposSheafComputer` | Sharded topological sheaves computing via Restriction Maps. | Avoids $O(N^3)$ locks by cutting matrices into chunks (overlapping patches). A true mathematical bypass. |

## The Combinatorial Purification (100% Formal combinatorial)

We previously used Neural Networks (Floats) to simulate Category Theory. At the user's explicit command ("make them purely mathematical"), all probabilistic heuristics were stripped. The engine now operates solely on exact Combinatorial Mathematics.

| Module | Intended Use | Formal Guarantee |
|--------|--------------|-----------|
| `topos_ai.hott.FormalHomotopyEquivalence` | Strict Categorical Isomorphism (Univalence) Search. | Computes actual functorial bijections ($O(N!)$ combinatorial search) to prove Category Equivalence. |
| `topos_ai.infinity_categories.FormalInfinityCategoryValidator` | Inner Kan Horn Filler for Quasi-Categories. | Evaluates 0, 1, and 2-simplices mathematically to enforce $\Lambda^2_1 \to \Delta^2$ transitive closure compositions. |
| `Autonomous Theorem Prover (Lean 4)` | Python logic translating graphical paths into text templates (`.lean` syntax). | It creates the Blueprint (Path) but writes `sorry` for the actual micro-proof steps. It is a **Pathfinder**, not a formal compiler itself. |
| `topos_ai.monoidal.FiniteMonoidalCategory` | Finite monoidal category with bifunctor ⊗, associator α, and unitors λ/ρ. | Validates Pentagon and Triangle coherence laws exactly. Raises `ValueError` on any violation. |
| `topos_ai.monoidal.FiniteSymmetricMonoidalCategory` | Symmetric monoidal category with braiding γ. | Checks Hexagon coherence, involutivity (γ²=id), and naturality. |
| `topos_ai.enriched.FiniteEnrichedCategory` | Category enriched over a FiniteSymmetricMonoidalCategory V: hom-sets are V-objects, composition is a V-morphism. | Validates composition types (∘: hom(B,C)⊗hom(A,B)→hom(A,C)), identity types (j: I→hom(A,A)), associativity pentagon, and left/right unitality triangles. All checks are exact and combinatorial over discrete V. |
| `topos_ai.formal_kan.left_kan_extension` | Left Kan extension Lan_K X: D → FinSet via the colimit formula over comma categories (K↓d). | Exact quotient of ∐ X(c) by the comma-category equivalence relation. Verifiable adjunction: `verify_left_kan_universal_property` checks \|Nat(Lan_K X, Y)\| == \|Nat(X, Y∘K)\| by explicit enumeration. |
| `topos_ai.formal_kan.right_kan_extension` | Right Kan extension Ran_K X: D → FinSet via matching families (limit over (d↓K)). | Exact enumeration of all matching families. Verifiable co-adjunction: `verify_right_kan_universal_property` checks \|Nat(Y∘K, X)\| == \|Nat(Y, Ran_K X)\|. |
| `topos_ai.sites.GrothendieckSite` | A finite category equipped with a Grothendieck topology (covering sieves). | Validates maximality (maximal sieve covers), stability (covering sieves stable under pullback), and transitivity. `is_sheaf` checks the unique-gluing equalizer condition for every covering sieve. |
| `topos_ai.adjunction.FiniteAdjunction` | Formal adjunction F ⊣ G with unit η: id_C → G∘F and counit ε: F∘G → id_D. | Checks functor laws for F and G, naturality of η and ε, both triangle identities (ε_{F(c)}∘F(η_c)=id, G(ε_d)∘η_{G(d)}=id), and hom-bijection Φ: C(c,G(d)) ≅ D(F(c),d). All verifications are exact and combinatorial. |
| `topos_ai.formal_yoneda` | Yoneda Lemma: Nat(C(A,-), F) ≅ F(A) for any object A and set-valued functor F. | `representable_functor` builds C(A,-) exactly. `verify_yoneda` confirms \|Nat\|=\|F(A)\|, Φ∘Φ⁻¹=id, and Φ⁻¹∘Φ=id. `verify_yoneda_naturality_in_A` checks the naturality square for any f: A→B. |
| `topos_ai.topos` | Elementary topos structure on FinSet: Cartesian products, exponential objects Z^Y, curry/uncurry bijection, and subobject classifier Ω={T,F}. | `verify_ccc` checks \|Hom(X×Y,Z)\|=\|Hom(X,Z^Y)\| and roundtrip. `SubobjectClassifier.characteristic_morphism` produces the unique χ: X→Ω with χ⁻¹(T)=A. `verify_subobject_classifier` confirms the universal property of the pullback square. |
| `topos_ai.formal_lawvere_tierney` | Lawvere-Tierney topologies j: Ω→Ω on the elementary topos FinSet with Ω={T,F}. Verifies all three LT axioms: LT1 (j(⊤)=⊤), LT2 (j∘j=j, idempotent), LT3 (j(p∧q)=j(p)∧j(q), preserves meets). | `all_lt_topologies()` returns exactly 2 valid topologies (identity and dense). `j_closure` computes j̄(A)={x∈X \| j(χ_A(x))=T}. `j_closed_subobjects` enumerates fixed points of the closure operator. `j_dense_monomorphism` tests j̄(A)=X. `verify_closure_operator` checks C1 (extensive), C2 (idempotent), C3 (monotone) on Sub(X). All verifications are exact and combinatorial. |
