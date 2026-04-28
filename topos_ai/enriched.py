"""
V-Enriched Categories.

A *V-enriched category* C, where V = (V, ⊗, I) is a monoidal category, assigns:
  - A set of objects  Ob(C)
  - For each pair of objects A, B: a *hom-object*  C(A, B) ∈ Ob(V)
  - For each triple A, B, C: a *composition morphism* in V:
        ∘_{A,B,C} : C(B, C) ⊗ C(A, B) → C(A, C)
  - For each object A: an *identity morphism* in V:
        j_A : I → C(A, A)

subject to associativity and unitality coherence axioms (stated as equalities
of V-morphisms).

This module provides ``FiniteEnrichedCategory``, which works over any
``FiniteMonoidalCategory`` as the enriching base and validates all axioms.

Examples
--------
1. **Ord-enriched (2-categories)**: V = finite preorders (viewed as SMC under ×).
   Hom-posets with monotonic composition.

2. **Ab-enriched (preadditive)**: V = discrete abelian monoids as strict SMC.
   Hom-abelian-groups with bilinear composition.

3. **Self-enriched**: any closed monoidal category is enriched over itself.
   The formal combinatorial version is a ``FiniteMonoidalCategory`` enriched
   over itself via the internal-hom object (not implemented here — requires
   a closed monoidal structure).

References
----------
- Kelly, G.M. (1982). *Basic Concepts of Enriched Category Theory*.
- Lawvere, F.W. (1973). *Metric Spaces, Generalized Logic, and Closed Categories*.
"""

from __future__ import annotations

from .formal_category import FiniteCategory
from .monoidal import FiniteMonoidalCategory


class FiniteEnrichedCategory:
    """
    A finite V-enriched category.

    Parameters
    ----------
    objects : iterable of object labels
    enriching : FiniteMonoidalCategory
        The monoidal category V = (V, ⊗, I) over which C is enriched.
    hom_objects : dict[(A, B), V-object]
        Maps each pair of C-objects to a V-object  C(A, B) ∈ Ob(V).
    compositions : dict[(A, B, C_obj), V-morphism]
        Maps each composable triple to the V-morphism
            ∘_{A,B,C} : C(B,C) ⊗ C(A,B) → C(A,C).
        In V-notation: a morphism from
            ``enriching.tensor_obj(hom(B,C), hom(A,B))``
        to  ``hom(A,C)``.
    identity_elements : dict[A, V-morphism]
        Maps each C-object to  j_A : I → C(A, A),
        a V-morphism from the monoidal unit to the identity hom-object.

    Validated axioms
    ----------------
    - **Hom-object totality**: every hom-object is a V-object.
    - **Composition type**: ∘_{A,B,C} is a V-morphism with the right type.
    - **Identity type**: j_A is a V-morphism  I → C(A,A).
    - **Associativity**: for all A,B,C,D the pentagon
          ∘_{A,C,D} ∘ (id ⊗ ∘_{A,B,C}) = ∘_{A,B,D} ∘ (∘_{B,C,D} ⊗ id) ∘ α
      holds as an equality of V-morphisms.
    - **Left unitality**: ∘_{A,A,B} ∘ (j_A ⊗ id) ∘ λ_inv = id
      (stated as ∘ ∘ (j ⊗ id) = λ_hom).
    - **Right unitality**: ∘_{A,B,B} ∘ (id ⊗ j_B) ∘ ρ_inv = id
      (stated as ∘ ∘ (id ⊗ j) = ρ_hom).
    """

    def __init__(
        self,
        objects,
        enriching: FiniteMonoidalCategory,
        hom_objects,
        compositions,
        identity_elements,
    ):
        self.objects = tuple(objects)
        self.object_set = frozenset(self.objects)
        self.enriching = enriching
        self.hom_objects = dict(hom_objects)       # (A, B) → V-obj
        self.compositions = dict(compositions)      # (A, B, C) → V-mor
        self.identity_elements = dict(identity_elements)  # A → V-mor

        self._validate()

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def hom(self, A, B):
        """Return the hom-object  C(A, B) ∈ Ob(V)."""
        try:
            return self.hom_objects[(A, B)]
        except KeyError as exc:
            raise ValueError(f"Hom-object C({A!r}, {B!r}) is not defined.") from exc

    def compose(self, A, B, C):
        """
        Return the composition V-morphism
            ∘_{A,B,C} : C(B,C) ⊗ C(A,B) → C(A,C).
        """
        try:
            return self.compositions[(A, B, C)]
        except KeyError as exc:
            raise ValueError(f"Composition ∘_{{{A},{B},{C}}} is not defined.") from exc

    def identity_element(self, A):
        """Return  j_A : I → C(A, A)."""
        try:
            return self.identity_elements[A]
        except KeyError as exc:
            raise ValueError(f"Identity element j_{A!r} is not defined.") from exc

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def _validate(self):
        self._check_hom_totality()
        self._check_composition_types()
        self._check_identity_types()
        self._check_associativity()
        self._check_left_unitality()
        self._check_right_unitality()

    def _check_hom_totality(self):
        """Every hom-object must be a V-object."""
        V = self.enriching.category
        for A in self.objects:
            for B in self.objects:
                if (A, B) not in self.hom_objects:
                    raise ValueError(f"Missing hom-object C({A!r}, {B!r}).")
                h = self.hom_objects[(A, B)]
                if h not in V.object_set:
                    raise ValueError(
                        f"Hom-object C({A!r}, {B!r}) = {h!r} is not a V-object."
                    )

    def _check_composition_types(self):
        """
        ∘_{A,B,C} must be a V-morphism  C(B,C) ⊗ C(A,B) → C(A,C).
        """
        V = self.enriching
        C_cat = V.category
        for A in self.objects:
            for B in self.objects:
                for C in self.objects:
                    if (A, B, C) not in self.compositions:
                        raise ValueError(
                            f"Missing composition morphism ∘_{{{A},{B},{C}}}."
                        )
                    mor = self.compositions[(A, B, C)]
                    if mor not in C_cat.morphisms:
                        raise ValueError(
                            f"Composition ∘_{{{A},{B},{C}}} = {mor!r} is not a V-morphism."
                        )
                    expected_src = V.tensor_obj(self.hom(B, C), self.hom(A, B))
                    expected_dst = self.hom(A, C)
                    actual_src, actual_dst = C_cat.morphisms[mor]
                    if actual_src != expected_src or actual_dst != expected_dst:
                        raise ValueError(
                            f"Composition ∘_{{{A},{B},{C}}} has type "
                            f"{actual_src!r} → {actual_dst!r}, "
                            f"expected {expected_src!r} → {expected_dst!r}."
                        )

    def _check_identity_types(self):
        """j_A must be a V-morphism  I → C(A, A)."""
        V = self.enriching
        C_cat = V.category
        for A in self.objects:
            if A not in self.identity_elements:
                raise ValueError(f"Missing identity element j_{A!r}.")
            mor = self.identity_elements[A]
            if mor not in C_cat.morphisms:
                raise ValueError(
                    f"Identity element j_{A!r} = {mor!r} is not a V-morphism."
                )
            expected_src = V.unit
            expected_dst = self.hom(A, A)
            actual_src, actual_dst = C_cat.morphisms[mor]
            if actual_src != expected_src or actual_dst != expected_dst:
                raise ValueError(
                    f"Identity element j_{A!r} has type "
                    f"{actual_src!r} → {actual_dst!r}, "
                    f"expected {expected_src!r} → {expected_dst!r}."
                )

    def _check_associativity(self):
        """
        Associativity coherence for all quadruples (A, B, C, D):

            ∘_{A,C,D} ∘ (id_{C(C,D)} ⊗ ∘_{A,B,C})
          = ∘_{A,B,D} ∘ (∘_{B,C,D} ⊗ id_{C(A,B)}) ∘ α_{C(C,D), C(B,C), C(A,B)}

        Both sides are V-morphisms
            C(C,D) ⊗ (C(B,C) ⊗ C(A,B)) → C(A,D).
        """
        V = self.enriching
        C_cat = V.category

        for A in self.objects:
            for B in self.objects:
                for C in self.objects:
                    for D in self.objects:
                        hAB = self.hom(A, B)
                        hBC = self.hom(B, C)
                        hCD = self.hom(C, D)
                        hAC = self.hom(A, C)
                        hAD = self.hom(A, D)
                        hBD = self.hom(B, D)

                        cABC = self.compose(A, B, C)   # C(B,C)⊗C(A,B) → C(A,C)
                        cACD = self.compose(A, C, D)   # C(C,D)⊗C(A,C) → C(A,D)
                        cABD = self.compose(A, B, D)   # C(B,D)⊗C(A,B) → C(A,D)
                        cBCD = self.compose(B, C, D)   # C(C,D)⊗C(B,C) → C(B,D)

                        id_hCD = C_cat.identities[hCD]
                        id_hAB = C_cat.identities[hAB]
                        alpha = V.alpha(hCD, hBC, hAB)

                        # LHS: ∘_{A,C,D} ∘ (id_{C,D} ⊗ ∘_{A,B,C})
                        # id_{C,D} ⊗ cABC : C(C,D)⊗(C(B,C)⊗C(A,B)) → C(C,D)⊗C(A,C)
                        step1_lhs = V.tensor_mor(id_hCD, cABC)
                        lhs = C_cat.compose(cACD, step1_lhs)

                        # RHS: ∘_{A,B,D} ∘ (∘_{B,C,D} ⊗ id_{A,B}) ∘ α
                        # cBCD ⊗ id_{A,B} : (C(C,D)⊗C(B,C))⊗C(A,B) → C(B,D)⊗C(A,B)
                        step1_rhs = V.tensor_mor(cBCD, id_hAB)
                        step2_rhs = C_cat.compose(cABD, step1_rhs)
                        rhs = C_cat.compose(step2_rhs, alpha)

                        if lhs != rhs:
                            raise ValueError(
                                f"Enriched associativity fails for "
                                f"(A={A!r}, B={B!r}, C={C!r}, D={D!r}): "
                                f"lhs={lhs!r}, rhs={rhs!r}."
                            )

    def _check_left_unitality(self):
        """
        Left unitality for all pairs (A, B):

            ∘_{A,A,B} ∘ (id_{C(A,B)} ⊗ j_A) = λ_{C(A,B)}

        Both sides: I ⊗ C(A,B) → C(A,B).
        """
        V = self.enriching
        C_cat = V.category

        for A in self.objects:
            for B in self.objects:
                hAB = self.hom(A, B)
                hAA = self.hom(A, A)
                cAAB = self.compose(A, A, B)
                j_A = self.identity_elements[A]
                id_hAB = C_cat.identities[hAB]
                lambda_hAB = V.lambda_(hAB)

                # id_{C(A,B)} ⊗ j_A : I ⊗ C(A,B) → C(A,A) ⊗ C(A,B)
                step = V.tensor_mor(id_hAB, j_A)
                lhs = C_cat.compose(cAAB, step)
                rhs = lambda_hAB

                if lhs != rhs:
                    raise ValueError(
                        f"Left unitality fails for (A={A!r}, B={B!r}): "
                        f"lhs={lhs!r}, rhs={rhs!r}."
                    )

    def _check_right_unitality(self):
        """
        Right unitality for all pairs (A, B):

            ∘_{A,B,B} ∘ (j_B ⊗ id_{C(A,B)}) = ρ_{C(A,B)}

        Both sides: C(A,B) ⊗ I → C(A,B).
        """
        V = self.enriching
        C_cat = V.category

        for A in self.objects:
            for B in self.objects:
                hAB = self.hom(A, B)
                cABB = self.compose(A, B, B)
                j_B = self.identity_elements[B]
                id_hAB = C_cat.identities[hAB]
                rho_hAB = V.rho(hAB)

                # j_B ⊗ id_{C(A,B)} : C(A,B) ⊗ I → C(B,B) ⊗ C(A,B)
                step = V.tensor_mor(j_B, id_hAB)
                lhs = C_cat.compose(cABB, step)
                rhs = rho_hAB

                if lhs != rhs:
                    raise ValueError(
                        f"Right unitality fails for (A={A!r}, B={B!r}): "
                        f"lhs={rhs!r}, rhs={rhs!r}."
                    )

    # ------------------------------------------------------------------ #
    # Derived operations                                                   #
    # ------------------------------------------------------------------ #

    def underlying_category(self) -> FiniteCategory:
        """
        Extract the **underlying ordinary category** of the enriched category.

        Objects are the same as C.  A morphism  f : A → B  in the underlying
        category corresponds to an element of  C(A, B)  (i.e. a morphism
        in V from the unit I to C(A,B)).  The identity at A is j_A, and
        composition is induced by ∘_{A,B,C}.

        In the finite setting we represent each such element as a V-morphism
        I → C(A,B) and label morphisms accordingly.
        """
        V = self.enriching
        C_cat = V.category
        unit = V.unit

        # Collect all V-morphisms  I → hom(A,B)  for each pair
        underlying_morphisms = {}
        for A in self.objects:
            for B in self.objects:
                hAB = self.hom(A, B)
                for mor, (src, dst) in C_cat.morphisms.items():
                    if src == unit and dst == hAB:
                        label = f"({A!r}→{B!r})[{mor!r}]"
                        underlying_morphisms[label] = (A, B)

        # Identities: j_A : I → C(A,A)
        identities = {}
        for A in self.objects:
            j = self.identity_elements[A]
            label = f"({A!r}→{A!r})[{j!r}]"
            identities[A] = label

        # Composition: compose by applying ∘_{A,B,C} to the tensor of two morphisms
        composition = {}
        for label_g, (B, C) in underlying_morphisms.items():
            g_mor = label_g.split("[")[1].rstrip("]").strip("'\"")
            for label_f, (A, B2) in underlying_morphisms.items():
                if B2 != B:
                    continue
                f_mor = label_f.split("[")[1].rstrip("]").strip("'\"")
                if g_mor not in C_cat.morphisms or f_mor not in C_cat.morphisms:
                    continue
                # tensor f_mor ⊗ g_mor (reversed: ∘ takes C(B,C) ⊗ C(A,B))
                try:
                    fg = V.tensor_mor(g_mor, f_mor)
                    result_mor = C_cat.compose(self.compose(A, B, C), fg)
                    result_label = f"({A!r}→{C!r})[{result_mor!r}]"
                    if result_label in underlying_morphisms:
                        composition[(label_g, label_f)] = result_label
                except (ValueError, KeyError):
                    continue

        # Fill identity compositions
        for label, (A, B) in underlying_morphisms.items():
            id_A = identities.get(A)
            id_B = identities.get(B)
            if id_A:
                composition.setdefault((label, id_A), label)
            if id_B:
                composition.setdefault((id_B, label), label)

        return FiniteCategory(
            objects=self.objects,
            morphisms=underlying_morphisms,
            identities=identities,
            composition=composition,
        )


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def discrete_enriched_category(
    objects,
    enriching: FiniteMonoidalCategory,
    hom_matrix,
) -> FiniteEnrichedCategory:
    """
    Build a **discrete V-enriched category** where the composition morphisms
    are uniquely determined by the monoidal structure.

    In a discrete enriched category the only morphisms are identities,
    so  ∘_{A,B,C} = the unique V-morphism  hom(B,C) ⊗ hom(A,B) → hom(A,C)
    that is compatible with unitality (here: the composition in V).

    Parameters
    ----------
    objects   : iterable of object labels
    enriching : FiniteMonoidalCategory  (V)
    hom_matrix : dict[(A, B), V-object]
        The hom-objects. For a discrete category,  hom(A, A)  should be the
        monoidal unit I and  hom(A, B)  for A ≠ B can be any V-object.

    Returns
    -------
    FiniteEnrichedCategory
    """
    V = enriching
    C_cat = V.category
    objs = list(objects)

    # Composition morphisms: C(B,C)⊗C(A,B) → C(A,C)
    # We pick the (unique if discrete) morphism between these objects
    compositions = {}
    for A in objs:
        for B in objs:
            for C in objs:
                src_obj = V.tensor_obj(hom_matrix[(B, C)], hom_matrix[(A, B)])
                dst_obj = hom_matrix[(A, C)]
                # Find a morphism src_obj → dst_obj in V
                candidates = C_cat.hom(src_obj, dst_obj)
                if not candidates:
                    raise ValueError(
                        f"No V-morphism {src_obj!r} → {dst_obj!r} "
                        f"for composition ∘_({A},{B},{C})."
                    )
                compositions[(A, B, C)] = candidates[0]

    # Identity elements: I → C(A,A)
    identity_elements = {}
    for A in objs:
        dst_obj = hom_matrix[(A, A)]
        candidates = C_cat.hom(V.unit, dst_obj)
        if not candidates:
            raise ValueError(
                f"No V-morphism I → {dst_obj!r} for identity j_{A!r}."
            )
        identity_elements[A] = candidates[0]

    return FiniteEnrichedCategory(
        objects=objs,
        enriching=enriching,
        hom_objects=hom_matrix,
        compositions=compositions,
        identity_elements=identity_elements,
    )
