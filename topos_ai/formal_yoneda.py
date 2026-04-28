"""
topos_ai.formal_yoneda — Formal Yoneda Lemma.

Proves and verifies the Yoneda Lemma for finite categories:

    Nat(C(A, -), F) ≅ F(A)

for any object A ∈ C and any set-valued functor F: C → FinSet.

The bijection is given by:
  Φ(α) = α_A(id_A)          (evaluation at the identity)
  Φ⁻¹(x)_X(m) = F(m)(x)    (transport along morphisms)

Mathematical contracts
----------------------
• ``representable_functor(cat, A)`` returns the functor C(A,-): C → FinSet
  exactly as a FiniteSetFunctor.
• ``all_natural_transformations`` (from formal_kan) enumerates Nat(C(A,-), F).
• ``yoneda_map`` and ``yoneda_inverse`` form the explicit bijection.
• ``verify_yoneda`` confirms |Nat| = |F(A)| and that Φ∘Φ⁻¹ = id and Φ⁻¹∘Φ = id.
• Naturality in A: for f: A' → A in C, the bijection commutes with pre-composition.

References
----------
S. Mac Lane, "Categories for the Working Mathematician" (2nd ed.),
Chapter III (Representability and the Yoneda Lemma).
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional

from .formal_kan import FiniteSetFunctor, all_natural_transformations


# ---------------------------------------------------------------------------
# Representable functor  C(A, -): C → FinSet
# ---------------------------------------------------------------------------

def representable_functor(cat, A: str) -> FiniteSetFunctor:
    """
    Build the representable functor  h_A = C(A, -): C → FinSet.

    • h_A(X)  = { all morphisms  m: A → X  in C }   (as a frozenset)
    • h_A(f: X → Y)(m) = f ∘ m   (post-composition)

    Parameters
    ----------
    cat : FiniteCategory
    A   : str — the representing object.

    Returns
    -------
    FiniteSetFunctor on cat.
    """
    if A not in cat.object_set:
        raise ValueError(f"representable_functor: {A!r} not an object of the category")

    obj_map: Dict[str, FrozenSet[str]] = {}
    for X in cat.objects:
        obj_map[X] = frozenset(cat.hom(A, X))

    mor_map: Dict[str, Dict[str, str]] = {}
    for f in cat.morphisms:
        X = cat.source(f)
        Y = cat.target(f)
        # h_A(f): h_A(X) → h_A(Y)   sends m ↦ f ∘ m
        f_action: Dict[str, str] = {}
        for m in obj_map[X]:
            f_action[m] = cat.compose(f, m)
        mor_map[f] = f_action

    return FiniteSetFunctor(
        category=cat,
        objects_map=obj_map,
        morphism_map=mor_map,
        validate=True,
    )


# ---------------------------------------------------------------------------
# Yoneda bijection
# ---------------------------------------------------------------------------

def yoneda_map(nat: Dict[str, Dict], A: str) -> Any:
    """
    Apply the Yoneda map  Φ(α) = α_A(id_A).

    Parameters
    ----------
    nat : dict — a natural transformation α: C(A,-) → F,
          represented as  { object_name: { morphism: element } }.
    A   : str — the representing object.

    Returns
    -------
    The element  α_A(id_A)  in F(A).
    """
    # id_A's name is found by looking for the morphism that maps to itself
    # under C(A,-)(A) = Hom(A, A); but we need the specific identity name.
    # The identity is the element that maps to itself under every F(f).
    # Actually nat[A] is α_A: C(A,A) → F(A), so we just look up id_A in α_A.
    # But we don't store the category here — so we use the hom-set element
    # that is the identity: the one element whose image under all morphisms
    # is preserved.  The caller must pass the identity name explicitly or
    # nat must include it directly.
    # CONVENTION: nat[A] is a dict {morphism_name: F(A)_element}, and
    # id_A is the identity morphism name stored in cat. We look for the
    # key that is the identity by requiring its value to be the "identity
    # element" — but here we don't have the category. So we return the
    # mapping of the first key that has source=A target=A.
    # Simplest approach: caller provides the id_name.
    raise NotImplementedError(
        "Use yoneda_evaluate(nat, cat, A) instead — it resolves id_A from the category."
    )


def yoneda_evaluate(nat: Dict[str, Dict], cat, A: str) -> Any:
    """
    Apply the Yoneda map  Φ(α) = α_A(id_A).

    Parameters
    ----------
    nat : dict — a natural transformation α: C(A,-) → F.
    cat : FiniteCategory — used to look up the identity morphism name.
    A   : str — the representing object.

    Returns
    -------
    The element  α_A(id_A)  in F(A).
    """
    id_A = cat.identities[A]
    return nat[A][id_A]


def yoneda_inverse(element: Any, cat, F: FiniteSetFunctor, A: str) -> Dict[str, Dict]:
    """
    Build the natural transformation  α: C(A,-) → F  corresponding to
    an element  x ∈ F(A).

    The inverse Yoneda map:
      α_X(m: A→X) = F(m)(x)

    Parameters
    ----------
    element : Any — an element x ∈ F(A).
    cat     : FiniteCategory
    F       : FiniteSetFunctor on cat.
    A       : str — the representing object.

    Returns
    -------
    A dict  { X: { m: F(m)(element) } }  for each object X and morphism m: A→X.
    """
    if element not in F.apply_obj(A):
        raise ValueError(
            f"yoneda_inverse: element {element!r} ∉ F({A!r}) = {F.apply_obj(A)}"
        )
    nat: Dict[str, Dict] = {}
    for X in cat.objects:
        component: Dict[str, Any] = {}
        for m in cat.hom(A, X):
            component[m] = F.apply_mor(m)[element]
        nat[X] = component
    return nat


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------

def verify_yoneda(cat, F: FiniteSetFunctor, A: str) -> bool:
    """
    Verify the Yoneda lemma for (cat, F, A):

    1. |Nat(C(A,-), F)| == |F(A)|
    2. Φ∘Φ⁻¹ = id:  for each x ∈ F(A), Φ(Φ⁻¹(x)) = x
    3. Φ⁻¹∘Φ = id:  for each α ∈ Nat, Φ⁻¹(Φ(α)) = α

    Returns True iff all three hold.
    Raises ValueError with a description if any fails.
    """
    h_A = representable_functor(cat, A)
    nats = all_natural_transformations(h_A, F)
    FA = F.apply_obj(A)

    # 1. Cardinality
    if len(nats) != len(FA):
        raise ValueError(
            f"verify_yoneda: cardinality mismatch — "
            f"|Nat(C({A!r},-), F)| = {len(nats)} ≠ {len(FA)} = |F({A!r})|"
        )

    # 2. Φ∘Φ⁻¹ = id
    for x in FA:
        alpha = yoneda_inverse(x, cat, F, A)
        recovered = yoneda_evaluate(alpha, cat, A)
        if recovered != x:
            raise ValueError(
                f"verify_yoneda: Φ(Φ⁻¹({x!r})) = {recovered!r} ≠ {x!r}"
            )

    # 3. Φ⁻¹∘Φ = id
    for alpha in nats:
        x = yoneda_evaluate(alpha, cat, A)
        recovered_alpha = yoneda_inverse(x, cat, F, A)
        if recovered_alpha != alpha:
            raise ValueError(
                f"verify_yoneda: Φ⁻¹(Φ(α)) ≠ α  (got different natural transformation)"
            )

    return True


def verify_yoneda_naturality_in_A(cat, F: FiniteSetFunctor, A: str, B: str, f: str) -> bool:
    """
    Check naturality of the Yoneda bijection in the first argument.

    Given f: A -> B in C, the diagram commutes:

      Nat(C(A,-), F) --Phi_A--> F(A)
            |                     |
         f^*|                  F(f)|
            v                     v
      Nat(C(B,-), F) --Phi_B--> F(B)

    where f^*(alpha)_X(m: B->X) = alpha_X(m o f)  (pre-composition with f: A->B).

    Naturality condition:
      F(f)(Phi_A(alpha)) = Phi_B(f^*(alpha))

    Concretely:
      LHS = F(f)(alpha_A(id_A))
      RHS = f^*(alpha)_B(id_B) = alpha_B(id_B o f) = alpha_B(f)

    Returns True iff the diagram commutes for all alpha in Nat(C(A,-), F).
    """
    if cat.source(f) != A or cat.target(f) != B:
        raise ValueError(
            f"verify_yoneda_naturality_in_A: {f!r} has type "
            f"{cat.source(f)!r}->{cat.target(f)!r}, expected {A!r}->{B!r}"
        )
    h_A = representable_functor(cat, A)
    nats_A = all_natural_transformations(h_A, F)

    for alpha in nats_A:
        # LHS: F(f)(Phi_A(alpha)) = F(f)(alpha_A(id_A))
        x_A = yoneda_evaluate(alpha, cat, A)
        lhs = F.apply_mor(f).get(x_A)

        # RHS: Phi_B(f^*(alpha)) = alpha_B(id_B o f) = alpha_B(f)
        # f: A->B is in C(A, B), so alpha_B maps C(A,B) -> F(B)
        rhs = alpha[B].get(f)   # alpha_B(f: A->B)

        if lhs != rhs:
            return False
    return True
