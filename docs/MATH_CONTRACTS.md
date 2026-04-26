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
