"""
topos_ai.formal_lawvere_tierney — Formal Lawvere-Tierney Topologies.

A **Lawvere-Tierney topology** on an elementary topos is an endomorphism
j: Ω → Ω of the subobject classifier satisfying:

    (LT1)  j(⊤)  = ⊤                       (truth is preserved)
    (LT2)  j ∘ j = j                         (idempotent / closure)
    (LT3)  j(p ∧ q) = j(p) ∧ j(q)           (preserves meets)

Here we work in the elementary topos **FinSet** with:

    Ω  = { T, F }   (TRUE and FALSE from topos_ai.topos)
    ⊤  = T          (global truth morphism)
    ∧  : Ω × Ω → Ω  (conjunction)

A Lawvere-Tierney topology j determines:

* **j-closed subobjects**: monomorphisms  A ↪ X  such that the
  characteristic morphism χ_A satisfies  j ∘ χ_A = χ_A.
* **j-closure of a subobject**: the smallest j-closed subobject
  containing a given A ↪ X, computed by pulling back along j.
* **j-dense monomorphisms**: those where  j ∘ χ_A = ⊤  everywhere.

The bijection between Lawvere-Tierney topologies and Grothendieck
topologies (on the site whose underlying category is the topos) is
not implemented here — we focus on the finite combinatorial level.

Mathematical contracts
----------------------
* ``LawvereTierneyTopology`` stores j as a dict ``{T: ..., F: ...}``
  and verifies all three axioms on construction.
* ``all_lt_topologies()`` returns all valid j's on ``Ω = {T, F}``.
* ``j_closure(X, mono, j)`` computes the j-closure of a mono by
  characteristic morphism → apply j → pull back TRUE.
* ``j_closed_subobjects(X, j)`` enumerates all j-closed subobjects of X.
* ``j_dense_monomorphism(X, mono, j)`` tests whether a mono is j-dense.
* ``lt_from_grothendieck(site, cat)`` lifts a Grothendieck topology on a
  one-object category (a monoid) to the equivalent LT operator on Ω.

References
----------
P. Johnstone, "Sketches of an Elephant" (Vol. A), §A4.
S. Mac Lane & I. Moerdijk, "Sheaves in Geometry and Logic", Ch. III.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .topos import TRUE, FALSE, OMEGA, SubobjectClassifier


# ---------------------------------------------------------------------------
# Helpers: meet (∧) on Ω
# ---------------------------------------------------------------------------

def _meet(p: str, q: str) -> str:
    """Conjunction: p ∧ q on Ω = {T, F}."""
    return TRUE if (p == TRUE and q == TRUE) else FALSE


# ---------------------------------------------------------------------------
# LawvereTierneyTopology
# ---------------------------------------------------------------------------

class LawvereTierneyTopology:
    """
    A Lawvere-Tierney topology  j: Ω → Ω  on the elementary topos FinSet.

    Parameters
    ----------
    j : dict
        Mapping ``{TRUE: ..., FALSE: ...}`` specifying the endomorphism.
        Keys must be exactly ``{TRUE, FALSE}``.
    validate : bool
        If True (default), verify all three LT axioms on construction.

    Attributes
    ----------
    j : dict
        The underlying endomorphism of Ω.
    """

    def __init__(self, j: Dict[str, str], validate: bool = True):
        if set(j.keys()) != OMEGA:
            raise ValueError(
                f"LawvereTierneyTopology: j must be defined on all of "
                f"Ω = {OMEGA}, got keys {set(j.keys())}"
            )
        for v in j.values():
            if v not in OMEGA:
                raise ValueError(
                    f"LawvereTierneyTopology: j maps to {v!r} which is not in Ω"
                )
        self.j: Dict[str, str] = dict(j)
        if validate:
            self._check_lt1()
            self._check_lt2()
            self._check_lt3()

    # ------------------------------------------------------------------
    # Axiom checkers
    # ------------------------------------------------------------------

    def _check_lt1(self):
        """LT1: j(⊤) = ⊤."""
        if self.j[TRUE] != TRUE:
            raise ValueError(
                f"LawvereTierneyTopology: LT1 fails — j(TRUE) = {self.j[TRUE]!r}, "
                f"expected TRUE"
            )

    def _check_lt2(self):
        """LT2: j(j(p)) = j(p) for all p ∈ Ω (idempotency)."""
        for p in OMEGA:
            jp = self.j[p]
            jjp = self.j[jp]
            if jjp != jp:
                raise ValueError(
                    f"LawvereTierneyTopology: LT2 fails — j(j({p!r})) = {jjp!r}, "
                    f"expected j({p!r}) = {jp!r}"
                )

    def _check_lt3(self):
        """LT3: j(p ∧ q) = j(p) ∧ j(q) for all p, q ∈ Ω."""
        for p in sorted(OMEGA):
            for q in sorted(OMEGA):
                lhs = self.j[_meet(p, q)]
                rhs = _meet(self.j[p], self.j[q])
                if lhs != rhs:
                    raise ValueError(
                        f"LawvereTierneyTopology: LT3 fails — "
                        f"j({p!r} ∧ {q!r}) = {lhs!r}, "
                        f"but j({p!r}) ∧ j({q!r}) = {rhs!r}"
                    )

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, p: str) -> str:
        """Return j(p)."""
        if p not in OMEGA:
            raise ValueError(f"LawvereTierneyTopology.apply: {p!r} not in Ω")
        return self.j[p]

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> "LawvereTierneyTopology":
        """The identity topology j = id_Ω (trivial / finest topology)."""
        return cls({TRUE: TRUE, FALSE: FALSE})

    @classmethod
    def dense(cls) -> "LawvereTierneyTopology":
        """The dense topology j(p) = ⊤ for all p (coarsest topology)."""
        return cls({TRUE: TRUE, FALSE: TRUE})

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, LawvereTierneyTopology):
            return NotImplemented
        return self.j == other.j

    def __hash__(self):
        return hash(frozenset(self.j.items()))

    def __repr__(self) -> str:
        return f"LawvereTierneyTopology(j(T)={self.j[TRUE]!r}, j(F)={self.j[FALSE]!r})"


# ---------------------------------------------------------------------------
# Enumerate all LT topologies on Ω = {T, F}
# ---------------------------------------------------------------------------

def all_lt_topologies() -> List[LawvereTierneyTopology]:
    """
    Return all valid Lawvere-Tierney topologies on Ω = {T, F}.

    For the two-element classifier there are exactly two:
      1. Identity  j_id  : j(T)=T, j(F)=F
      2. Dense     j_⊤   : j(T)=T, j(F)=T
    """
    result = []
    for fT, fF in product(OMEGA, OMEGA):
        try:
            lt = LawvereTierneyTopology({TRUE: fT, FALSE: fF})
            result.append(lt)
        except ValueError:
            pass
    return result


# ---------------------------------------------------------------------------
# j-closure of a subobject
# ---------------------------------------------------------------------------

def j_closure(
    X: FrozenSet,
    mono: FrozenSet[Tuple],
    lt: LawvereTierneyTopology,
) -> FrozenSet[Tuple]:
    """
    Compute the **j-closure**  j̄(A)  of a subobject  A ↪ X.

    Given:
      - A mono A ↪ X encoded as a frozenset of (a, a) pairs
        (an inclusion whose image is some subset A ⊆ X),
      - A Lawvere-Tierney topology j,

    returns the j-closure as a frozenset of inclusion pairs:

        j̄(A) = { x ∈ X | j(χ_A(x)) = T }

    where χ_A: X → Ω is the characteristic morphism of A ↪ X.

    Parameters
    ----------
    X    : frozenset of elements (the ambient object).
    mono : frozenset of (a, a) pairs encoding the inclusion A ↪ X.
    lt   : LawvereTierneyTopology.

    Returns
    -------
    frozenset of (a, a) pairs — the j-closure inclusion A_j ↪ X.
    """
    sc = SubobjectClassifier()
    chi = dict(sc.characteristic_morphism(X, mono))
    closed_set: Set = set()
    for x in X:
        if lt.apply(chi[x]) == TRUE:
            closed_set.add(x)
    return frozenset((a, a) for a in closed_set)


def j_closed_subobjects(
    X: FrozenSet,
    lt: LawvereTierneyTopology,
) -> List[FrozenSet]:
    """
    Enumerate all **j-closed** subobjects of X.

    A subobject A ↪ X is j-closed if  j̄(A) = A, i.e.,
    j(χ_A(x)) = χ_A(x) for all x ∈ X.

    Parameters
    ----------
    X  : frozenset — the ambient FinSet object.
    lt : LawvereTierneyTopology.

    Returns
    -------
    List of frozensets of (a, a) pairs, one per j-closed subobject.
    """
    from itertools import combinations

    closed: List[FrozenSet] = []
    X_list = sorted(X, key=str)
    for r in range(len(X_list) + 1):
        for subset in combinations(X_list, r):
            A = frozenset(subset)
            mono = frozenset((a, a) for a in A)
            closure = j_closure(X, mono, lt)
            if closure == mono:
                closed.append(mono)
    return closed


def j_dense_monomorphism(
    X: FrozenSet,
    mono: FrozenSet[Tuple],
    lt: LawvereTierneyTopology,
) -> bool:
    """
    Test whether a mono A ↪ X is **j-dense**.

    A ↪ X is j-dense if  j(χ_A(x)) = T  for all x ∈ X,
    i.e., the j-closure of A is all of X.

    Parameters
    ----------
    X    : frozenset — the ambient object.
    mono : frozenset of (a, a) pairs.
    lt   : LawvereTierneyTopology.

    Returns
    -------
    bool
    """
    closure = j_closure(X, mono, lt)
    X_as_mono = frozenset((x, x) for x in X)
    return closure == X_as_mono


# ---------------------------------------------------------------------------
# Verify all LT axioms explicitly
# ---------------------------------------------------------------------------

def verify_lt_axioms(lt: LawvereTierneyTopology) -> bool:
    """
    Re-verify all three Lawvere-Tierney axioms for ``lt``.

    Returns True iff all three hold; raises ValueError otherwise.
    """
    lt._check_lt1()
    lt._check_lt2()
    lt._check_lt3()
    return True


# ---------------------------------------------------------------------------
# Subobject lattice with j-action
# ---------------------------------------------------------------------------

def subobject_lattice(X: FrozenSet) -> List[FrozenSet]:
    """
    Return all subobjects of X (as inclusion monos) ordered by inclusion.

    Each subobject is a frozenset of (a, a) pairs.
    """
    from itertools import combinations

    X_list = sorted(X, key=str)
    result = []
    for r in range(len(X_list) + 1):
        for subset in combinations(X_list, r):
            result.append(frozenset((a, a) for a in subset))
    return result


def j_action_on_subobjects(
    X: FrozenSet,
    lt: LawvereTierneyTopology,
) -> Dict[FrozenSet, FrozenSet]:
    """
    Compute the closure operator  A ↦ j̄(A)  on Sub(X).

    Returns a dict mapping each subobject mono (frozenset of pairs)
    to its j-closure mono.
    """
    result = {}
    for mono in subobject_lattice(X):
        result[mono] = j_closure(X, mono, lt)
    return result


def verify_closure_operator(
    X: FrozenSet,
    lt: LawvereTierneyTopology,
) -> bool:
    """
    Verify that  A ↦ j̄(A)  is a closure operator on Sub(X):

    (C1)  A ⊆ j̄(A)           (extensive)
    (C2)  j̄(j̄(A)) = j̄(A)    (idempotent)
    (C3)  A ⊆ B ⟹ j̄(A) ⊆ j̄(B)  (monotone)

    Returns True iff all three hold; raises ValueError otherwise.
    """
    action = j_action_on_subobjects(X, lt)

    def image(mono: FrozenSet) -> FrozenSet:
        return frozenset(x for (x, _) in mono)

    for mono, closure in action.items():
        A_img = image(mono)
        cA_img = image(closure)
        # C1: A ⊆ j̄(A)
        if not A_img.issubset(cA_img):
            raise ValueError(
                f"verify_closure_operator: C1 (extensive) fails: "
                f"{A_img} ⊄ {cA_img}"
            )
        # C2: j̄(j̄(A)) = j̄(A)
        double_closure = action[closure]
        if double_closure != closure:
            raise ValueError(
                f"verify_closure_operator: C2 (idempotent) fails: "
                f"j̄(j̄(A)) = {image(double_closure)} ≠ {cA_img} = j̄(A)"
            )

    # C3: monotonicity
    subobjects = list(action.keys())
    for m1 in subobjects:
        for m2 in subobjects:
            img1 = image(m1)
            img2 = image(m2)
            if img1.issubset(img2):
                cl1 = image(action[m1])
                cl2 = image(action[m2])
                if not cl1.issubset(cl2):
                    raise ValueError(
                        f"verify_closure_operator: C3 (monotone) fails: "
                        f"{img1} ⊆ {img2} but j̄({img1}) = {cl1} ⊄ {cl2} = j̄({img2})"
                    )
    return True
