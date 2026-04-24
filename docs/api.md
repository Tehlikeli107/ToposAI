# API Reference

## `topos_ai.models`

### `ToposTransformer(vocab_size, d_model=64, num_universes=4, num_layers=2, max_seq_len=2048)`

End-to-end trainable categorical language model. Attention scoring uses Lukasiewicz T-norm (dot-product free). Output projection uses cosine similarity.

```python
model  = ToposTransformer(vocab_size=50000, d_model=256, num_universes=8, num_layers=6)
logits, kv_cache = model(idx)              # training
logits, kv_cache = model(idx, kv_caches)   # inference with cache
```

---

## `topos_ai.kernels`

### `flash_topos_attention(q, k)`

Triton CUDA kernel computing Lukasiewicz implication `clamp(1 - Q + K, 0, 1)` over the feature dimension. Falls back to pure PyTorch when Triton is unavailable.

---

## `topos_ai.math`

### `lukasiewicz_composition(R1, R2)`
Matrix composition using Lukasiewicz T-norm: `max_k clamp(R1[i,k] + R2[k,j] - 1, 0)`.

### `transitive_closure(R, max_steps=5)`
Computes the reachability closure of a relation matrix.

### `sheaf_gluing(truth_A, truth_B, threshold=0.05)`
Returns `(True, global_section)` if `max|A - B| <= threshold`, else `(False, None)`.

---

## `topos_ai.logic`

### `SubobjectClassifier`
Implements Gödel/Heyting algebra with smooth (differentiable) approximations:
- `implies(A, B)` — smooth Gödel implication via `sigmoid((B - A) * hardness)`
- `logical_not(A)` — smooth intuitionistic negation via `sigmoid(-A * hardness)`
- `logical_and(A, B)` — `min(A, B)`
- `logical_or(A, B)` — `max(A, B)`

### `HeytingNeuralLayer(in_features, out_features)`
A neural layer where each output neuron computes `AND_i (x_i => w_ji)` via vectorized broadcasting.

---

## `topos_ai.topology`

### `PersistentHomology(num_nodes)`

#### `calculate_betti(distance_matrix, threshold) -> (β0, β1)`
Builds a Vietoris-Rips complex at `threshold` and computes Betti numbers using the correct boundary matrix rank for `β1`.

---

## `topos_ai.optim`

### `ToposAdam(params, lr=1e-3, betas=(0.9, 0.999), topological_weight_decay=0.01)`
Adam variant with Fisher information scaling for parameters constrained to the sigmoid manifold. Gradient amplification is clamped to prevent instability.

---

## `topos_ai.generation`

### `ToposConstrainedDecoder(reachability_matrix, threshold=0.1)`
Masks next-token logits using a categorical reachability matrix, then samples via `torch.multinomial`.

---

## `topos_ai.cohomology`

### `CechCohomology(num_nodes, edges)`
Computes H⁰ consensus disagreement and H¹ obstruction magnitude for distributed systems using proper boundary operators.

---

## `topos_ai.verification`

### `Lean4VerificationBridge(entities)`
Transpiles categorical reasoning chains into Lean 4 theorem syntax and optionally runs the Lean compiler for formal verification.
