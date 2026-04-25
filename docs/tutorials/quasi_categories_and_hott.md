# Quasi-Categories and HoTT Tutorial

This tutorial connects two finite formal layers:

- `FiniteSimplicialSet` checks face identities, degeneracy identities, horn fillers, and finite inner-Kan conditions.
- `FinitePathGroupoid` models 1-truncated identity types with functorial transport.

Run:

```bash
python examples/quasi_category_horns.py
python examples/hott_transport.py
```

Mathematical statements:

- Inner 2-horn fillers in the nerve of a category encode composition.
- Inner 3-horn coherence in the 3-skeleton encodes associativity.
- Finite path groupoids model identity proofs in a 1-truncated HoTT semantics.
- Path families validate transport as a functor out of the path groupoid.