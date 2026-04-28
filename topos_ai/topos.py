"""
topos_ai.topos — Elementary Topos Structures on FinSet.

Implements the key structures of an elementary topos concretely in the
category FinSet of finite sets:

  • **Finite products** A × B and their projection morphisms.
  • **Exponential objects** Z^Y (function sets) and the currying bijection
    Hom(X×Y, Z) ≅ Hom(X, Z^Y).
  • **Subobject classifier** Ω = {T, F}, truth morphism true: 1 → Ω, and
    the unique characteristic morphism χ_A: X → Ω for each subobject A ↪ X.

All objects are plain Python frozensets; morphisms are frozensets of
(input, output) pairs for hashability.  No PyTorch dependency.

Mathematical contracts
----------------------
Cartesian closedness:
  For any finite sets X, Y, Z:
  curry(f: X×Y→Z)   = g: X→Z^Y  with  ev(g(x), y) = f(x,y)
  uncurry(g: X→Z^Y) = f: X×Y→Z  with  f(x,y)      = ev(g(x), y)
  |Hom(X×Y, Z)| = |Hom(X, Z^Y)| = |Z|^(|X|·|Y|)

Subobject classifier:
  For A ⊆ X, the unique χ: X → Ω with χ(x) = T ⟺ x ∈ A is verified to
  be the ONLY morphism making the square a pullback.

References
----------
P. Johnstone, "Sketches of an Elephant" (2002), Part A (Toposes).
S. Mac Lane & I. Moerdijk, "Sheaves in Geometry and Logic" (1992), Ch. IV.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, FrozenSet, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A morphism in FinSet is a frozenset of (input, output) pairs.
FinSetMorphism = FrozenSet[Tuple[Any, Any]]

# Subobject classifier truth values
TRUE = "T"
FALSE = "F"
OMEGA: FrozenSet[str] = frozenset({TRUE, FALSE})


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def finset_product(A: FrozenSet, B: FrozenSet) -> FrozenSet:
    """
    Return the Cartesian product  A × B  as a frozenset of pairs.

    >>> finset_product(frozenset({1,2}), frozenset({'a','b'}))
    frozenset({(1,'a'), (1,'b'), (2,'a'), (2,'b')})
    """
    return frozenset((a, b) for a in A for b in B)


def product_projection_1(A: FrozenSet, B: FrozenSet) -> FinSetMorphism:
    """Return π₁: A × B → A."""
    return frozenset(((a, b), a) for a in A for b in B)


def product_projection_2(A: FrozenSet, B: FrozenSet) -> FinSetMorphism:
    """Return π₂: A × B → B."""
    return frozenset(((a, b), b) for a in A for b in B)


def product_morphism(f: FinSetMorphism, g: FinSetMorphism, A: FrozenSet, B: FrozenSet) -> FinSetMorphism:
    """
    Build the pairing morphism  ⟨f, g⟩: C → A × B from
    f: C → A and g: C → B.
    """
    f_dict = dict(f)
    g_dict = dict(g)
    domain = frozenset(f_dict.keys()) & frozenset(g_dict.keys())
    return frozenset((c, (f_dict[c], g_dict[c])) for c in domain)


# ---------------------------------------------------------------------------
# Exponential objects
# ---------------------------------------------------------------------------

def finset_exponential(Y: FrozenSet, Z: FrozenSet) -> FrozenSet[FrozenSet]:
    """
    Return the exponential  Z^Y  as a frozenset of functions Y → Z.

    Each function is represented as a frozenset of (y, z) pairs.
    """
    Y_sorted = sorted(Y, key=str)
    exps = []
    for outputs in product(sorted(Z, key=str), repeat=len(Y_sorted)):
        func = frozenset(zip(Y_sorted, outputs))
        exps.append(func)
    return frozenset(exps)


def evaluation_morphism(Y: FrozenSet, Z: FrozenSet) -> FinSetMorphism:
    """
    Build the evaluation morphism  ev: Z^Y × Y → Z.

    ev(h, y) = h(y)  where h ∈ Z^Y and y ∈ Y.
    """
    ZY = finset_exponential(Y, Z)
    ev_pairs = []
    for h in ZY:
        h_dict = dict(h)
        for y in Y:
            ev_pairs.append(((h, y), h_dict[y]))
    return frozenset(ev_pairs)


def curry(f: FinSetMorphism, X: FrozenSet, Y: FrozenSet, Z: FrozenSet) -> FinSetMorphism:
    """
    Curry  f: X × Y → Z  to  curry(f): X → Z^Y.

    curry(f)(x)(y) = f(x, y).

    Parameters
    ----------
    f : FinSetMorphism — the morphism X×Y → Z.
    X, Y, Z : FrozenSet — the object sets.

    Returns
    -------
    A FinSetMorphism from X to Z^Y.
    """
    f_dict = dict(f)
    ZY = finset_exponential(Y, Z)
    zY_by_dict: Dict[FrozenSet, FrozenSet] = {h: h for h in ZY}

    result: list = []
    for x in X:
        # Build the function  y ↦ f(x, y)
        func = frozenset((y, f_dict[(x, y)]) for y in Y)
        if func not in zY_by_dict:
            raise ValueError(
                f"curry: constructed function {dict(func)} not in Z^Y — "
                f"f may not have X×Y as domain"
            )
        result.append((x, func))
    return frozenset(result)


def uncurry(g: FinSetMorphism, X: FrozenSet, Y: FrozenSet, Z: FrozenSet) -> FinSetMorphism:
    """
    Uncurry  g: X → Z^Y  to  uncurry(g): X × Y → Z.

    uncurry(g)(x, y) = g(x)(y).

    Parameters
    ----------
    g : FinSetMorphism — the morphism X → Z^Y.
    X, Y, Z : FrozenSet — the object sets.

    Returns
    -------
    A FinSetMorphism from X×Y to Z.
    """
    g_dict = dict(g)
    result: list = []
    for x in X:
        h = g_dict[x]
        h_dict = dict(h)
        for y in Y:
            result.append(((x, y), h_dict[y]))
    return frozenset(result)


def all_finset_morphisms(A: FrozenSet, B: FrozenSet) -> FrozenSet[FinSetMorphism]:
    """Enumerate all functions A → B as FinSetMorphisms."""
    A_sorted = sorted(A, key=str)
    B_sorted = sorted(B, key=str)
    if not A_sorted:
        return frozenset({frozenset()})
    if not B_sorted:
        return frozenset()
    morphisms = []
    for outputs in product(B_sorted, repeat=len(A_sorted)):
        morphisms.append(frozenset(zip(A_sorted, outputs)))
    return frozenset(morphisms)


def verify_ccc(X: FrozenSet, Y: FrozenSet, Z: FrozenSet) -> bool:
    """
    Verify the Cartesian-closed adjunction for FinSet:

      |Hom(X×Y, Z)|  ==  |Hom(X, Z^Y)|

    and that curry/uncurry form explicit mutual inverses.

    Returns True iff all checks pass; raises ValueError otherwise.
    """
    XY = finset_product(X, Y)
    ZY = finset_exponential(Y, Z)

    hom_XY_Z = all_finset_morphisms(XY, Z)
    hom_X_ZY = all_finset_morphisms(X, ZY)

    if len(hom_XY_Z) != len(hom_X_ZY):
        raise ValueError(
            f"verify_ccc: cardinality mismatch — "
            f"|Hom(X×Y,Z)| = {len(hom_XY_Z)} ≠ {len(hom_X_ZY)} = |Hom(X,Z^Y)|"
        )

    # Check curry/uncurry are inverse
    for f in hom_XY_Z:
        g = curry(f, X, Y, Z)
        f_back = uncurry(g, X, Y, Z)
        if f_back != f:
            raise ValueError(
                "verify_ccc: uncurry(curry(f)) ≠ f"
            )

    for g in hom_X_ZY:
        f = uncurry(g, X, Y, Z)
        g_back = curry(f, X, Y, Z)
        if g_back != g:
            raise ValueError(
                "verify_ccc: curry(uncurry(g)) ≠ g"
            )

    return True


# ---------------------------------------------------------------------------
# Subobject classifier
# ---------------------------------------------------------------------------

class SubobjectClassifier:
    """
    The subobject classifier  (Ω, true)  for FinSet.

    Ω = {T, F}
    true: {*} → Ω  maps  * ↦ T

    For any finite set X and subset A ⊆ X (given as an inclusion
    mono m: A → X), the unique characteristic morphism
    χ_A: X → Ω  satisfies  χ_A(x) = T  ⟺  x ∈ image(m).
    """

    def __init__(self):
        self.Omega: FrozenSet[str] = OMEGA
        self.terminal: FrozenSet = frozenset({"*"})
        self.true_morphism: FinSetMorphism = frozenset({("*", TRUE)})

    def characteristic_morphism(
        self, X: FrozenSet, mono: FinSetMorphism
    ) -> FinSetMorphism:
        """
        Compute the unique characteristic morphism χ: X → Ω.

        Parameters
        ----------
        X    : FrozenSet — the ambient set.
        mono : FinSetMorphism — the inclusion map A → X (a monomorphism).

        Returns
        -------
        The FinSetMorphism χ: X → Ω  with χ(x) = T iff x ∈ image(mono).
        """
        image = frozenset(y for _, y in mono)
        return frozenset((x, TRUE if x in image else FALSE) for x in X)

    def verify_pullback(
        self, X: FrozenSet, A: FrozenSet, mono: FinSetMorphism
    ) -> bool:
        """
        Verify that χ_A is the UNIQUE morphism X → Ω making A the pullback.

        Pullback condition: preimage of T under χ equals image(mono).

        Returns True iff exactly one χ satisfies the pullback condition,
        and it equals the characteristic morphism.
        """
        image = frozenset(y for _, y in mono)

        # Verify that χ is injective on A: check mono is monic
        # (All distinct inputs in A must map to distinct outputs in X.)
        mono_dict = dict(mono)
        if len(frozenset(mono_dict.values())) != len(mono_dict):
            raise ValueError("verify_pullback: mono is not a monomorphism (not injective)")

        # image must be a subset of X
        if not image <= X:
            raise ValueError(f"verify_pullback: image of mono {image} ⊄ X = {X}")

        chi = self.characteristic_morphism(X, mono)
        chi_dict = dict(chi)

        # 1. Commutativity: chi(m(a)) = T for all a in A
        for a, ma in mono_dict.items():
            if chi_dict[ma] != TRUE:
                raise ValueError(
                    f"verify_pullback: commutativity fails — χ({ma!r}) = {chi_dict[ma]!r}, expected T"
                )

        # 2. Universality: image of mono = preimage of T under chi
        preimage_T = frozenset(x for x in X if chi_dict[x] == TRUE)
        if preimage_T != image:
            raise ValueError(
                f"verify_pullback: universality fails — preimage(T) = {preimage_T} ≠ {image} = image(mono)"
            )

        # 3. Uniqueness: enumerate all X→Ω and count those satisfying the pullback
        valid = []
        for candidate in all_finset_morphisms(X, self.Omega):
            cand_dict = dict(candidate)
            # Check commutativity
            ok = all(cand_dict.get(ma) == TRUE for ma in image)
            # Check universality
            if ok:
                ok = frozenset(x for x in X if cand_dict[x] == TRUE) == image
            if ok:
                valid.append(candidate)

        if len(valid) != 1:
            raise ValueError(
                f"verify_pullback: found {len(valid)} valid characteristic morphisms, expected exactly 1"
            )

        return True

    def __repr__(self):
        return f"SubobjectClassifier(Ω={self.Omega})"


# ---------------------------------------------------------------------------
# Convenience: verify the subobject classifier for a subset
# ---------------------------------------------------------------------------

def verify_subobject_classifier(X: FrozenSet, A: FrozenSet) -> bool:
    """
    Verify the subobject classifier axiom for the subset  A ⊆ X  in FinSet.

    Constructs the inclusion mono A → X, then checks uniqueness via
    SubobjectClassifier.verify_pullback.

    Returns True; raises ValueError if the axiom fails.
    """
    if not A <= X:
        raise ValueError(f"verify_subobject_classifier: A = {A} ⊄ X = {X}")
    mono = frozenset((a, a) for a in A)   # inclusion / identity on A
    sc = SubobjectClassifier()
    return sc.verify_pullback(X, A, mono)
