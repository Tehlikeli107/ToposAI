"""
Finite monoidal and symmetric monoidal categories.

A monoidal category (C, ⊗, I, α, λ, ρ) equips a category with a tensor
product bifunctor and coherent natural isomorphisms.  A symmetric monoidal
category adds a braiding γ satisfying the hexagon axiom.

All structures are represented explicitly and validated at construction time.
"""

from __future__ import annotations

from .formal_category import FiniteCategory


class FiniteMonoidalCategory:
    """
    Finite monoidal category (C, ⊗, I, α, λ, ρ).

    Parameters
    ----------
    category : FiniteCategory
        The underlying category C.
    tensor_objects : dict[(str, str), str]
        Maps (A, B) to A ⊗ B for every pair of objects.
    tensor_morphisms : dict[(str, str), str]
        Maps (f, g) to f ⊗ g for every pair of morphisms.
        For f: A → A' and g: B → B', f ⊗ g : A⊗B → A'⊗B'.
    unit : str
        The monoidal unit object I.
    associator : dict[(str, str, str), str]
        Maps (A, B, C) to α_{A,B,C} : (A⊗B)⊗C → A⊗(B⊗C).
    left_unitor : dict[str, str]
        Maps A to λ_A : I⊗A → A.
    right_unitor : dict[str, str]
        Maps A to ρ_A : A⊗I → A.

    Validated axioms
    ----------------
    - Bifunctoriality of ⊗ (identity and composition interchange)
    - Naturality of α, λ, ρ
    - Pentagon coherence
    - Triangle coherence
    """

    def __init__(
        self,
        category: FiniteCategory,
        tensor_objects,
        tensor_morphisms,
        unit: str,
        associator,
        left_unitor,
        right_unitor,
    ):
        self.category = category
        self.tensor_objects = dict(tensor_objects)
        self.tensor_morphisms = dict(tensor_morphisms)
        self.unit = unit
        self.associator = dict(associator)
        self.left_unitor = dict(left_unitor)
        self.right_unitor = dict(right_unitor)
        self.validate()

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def tensor_obj(self, A: str, B: str) -> str:
        """Return A ⊗ B."""
        try:
            return self.tensor_objects[(A, B)]
        except KeyError as exc:
            raise ValueError(f"Tensor product {A!r} ⊗ {B!r} is not defined.") from exc

    def tensor_mor(self, f: str, g: str) -> str:
        """Return f ⊗ g."""
        try:
            return self.tensor_morphisms[(f, g)]
        except KeyError as exc:
            raise ValueError(f"Tensor product of morphisms {f!r} ⊗ {g!r} is not defined.") from exc

    def alpha(self, A: str, B: str, C: str) -> str:
        """Return α_{A,B,C} : (A⊗B)⊗C → A⊗(B⊗C)."""
        try:
            return self.associator[(A, B, C)]
        except KeyError as exc:
            raise ValueError(f"Associator α_{{{A},{B},{C}}} is not defined.") from exc

    def lambda_(self, A: str) -> str:
        """Return λ_A : I⊗A → A."""
        try:
            return self.left_unitor[A]
        except KeyError as exc:
            raise ValueError(f"Left unitor λ_{A} is not defined.") from exc

    def rho(self, A: str) -> str:
        """Return ρ_A : A⊗I → A."""
        try:
            return self.right_unitor[A]
        except KeyError as exc:
            raise ValueError(f"Right unitor ρ_{A} is not defined.") from exc

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        self._check_tensor_totality()
        self._check_bifunctoriality()
        self._check_associator_types()
        self._check_unitor_types()
        self._check_associator_naturality()
        self._check_unitor_naturality()
        self._check_pentagon()
        self._check_triangle()
        return True

    def _check_tensor_totality(self):
        C = self.category
        if self.unit not in C.object_set:
            raise ValueError(f"Unit {self.unit!r} is not an object of the category.")
        for A in C.objects:
            for B in C.objects:
                AB = self.tensor_obj(A, B)
                if AB not in C.object_set:
                    raise ValueError(f"Tensor {A}⊗{B} = {AB!r} is not a category object.")
        for f, (src_f, dst_f) in C.morphisms.items():
            for g, (src_g, dst_g) in C.morphisms.items():
                fg = self.tensor_mor(f, g)
                if fg not in C.morphisms:
                    raise ValueError(f"Tensor morphism {f}⊗{g} = {fg!r} is not a morphism.")
                expected = (self.tensor_obj(src_f, src_g), self.tensor_obj(dst_f, dst_g))
                if C.morphisms[fg] != expected:
                    raise ValueError(
                        f"Tensor morphism {f}⊗{g} has type {C.morphisms[fg]}, expected {expected}."
                    )

    def _check_bifunctoriality(self):
        C = self.category
        # id_A ⊗ id_B = id_{A⊗B}
        for A in C.objects:
            for B in C.objects:
                if self.tensor_mor(C.identities[A], C.identities[B]) != C.identities[self.tensor_obj(A, B)]:
                    raise ValueError(f"Bifunctoriality: id_{A} ⊗ id_{B} ≠ id_{{A⊗B}}.")
        # (h∘f) ⊗ (k∘g) = (h⊗k) ∘ (f⊗g)
        for h, f in C.composable_pairs():
            for k, g in C.composable_pairs():
                if C.compose(self.tensor_mor(h, k), self.tensor_mor(f, g)) != self.tensor_mor(
                    C.compose(h, f), C.compose(k, g)
                ):
                    raise ValueError(
                        f"Bifunctoriality interchange fails for h={h}, f={f}, k={k}, g={g}."
                    )

    def _check_associator_types(self):
        C = self.category
        for A in C.objects:
            for B in C.objects:
                for Cv in C.objects:
                    a = self.alpha(A, B, Cv)
                    if a not in C.morphisms:
                        raise ValueError(f"Associator α_{{{A},{B},{Cv}}} is not a morphism.")
                    AB_C = self.tensor_obj(self.tensor_obj(A, B), Cv)
                    A_BC = self.tensor_obj(A, self.tensor_obj(B, Cv))
                    if C.morphisms[a] != (AB_C, A_BC):
                        raise ValueError(
                            f"Associator α_{{{A},{B},{Cv}}} has type {C.morphisms[a]}, "
                            f"expected ({AB_C} → {A_BC})."
                        )

    def _check_unitor_types(self):
        C = self.category
        I = self.unit
        for A in C.objects:
            lam = self.lambda_(A)
            IA = self.tensor_obj(I, A)
            if C.morphisms.get(lam) != (IA, A):
                raise ValueError(f"Left unitor λ_{A}: expected {IA} → {A}, got {C.morphisms.get(lam)}.")
            rho = self.rho(A)
            AI = self.tensor_obj(A, I)
            if C.morphisms.get(rho) != (AI, A):
                raise ValueError(f"Right unitor ρ_{A}: expected {AI} → {A}, got {C.morphisms.get(rho)}.")

    def _check_associator_naturality(self):
        """α_{A',B',C'} ∘ ((f⊗g)⊗h) = (f⊗(g⊗h)) ∘ α_{A,B,C}"""
        C = self.category
        for f, (sf, df) in C.morphisms.items():
            for g, (sg, dg) in C.morphisms.items():
                for h, (sh, dh) in C.morphisms.items():
                    lhs = C.compose(
                        self.alpha(df, dg, dh),
                        self.tensor_mor(self.tensor_mor(f, g), h),
                    )
                    rhs = C.compose(
                        self.tensor_mor(f, self.tensor_mor(g, h)),
                        self.alpha(sf, sg, sh),
                    )
                    if lhs != rhs:
                        raise ValueError(f"Associator naturality fails for f={f}, g={g}, h={h}.")

    def _check_unitor_naturality(self):
        """λ_B ∘ (id_I ⊗ f) = f ∘ λ_A   and   ρ_B ∘ (f ⊗ id_I) = f ∘ ρ_A"""
        C = self.category
        id_I = C.identities[self.unit]
        for f, (src_f, dst_f) in C.morphisms.items():
            if C.compose(self.lambda_(dst_f), self.tensor_mor(id_I, f)) != C.compose(f, self.lambda_(src_f)):
                raise ValueError(f"Left unitor naturality fails for f={f}.")
            if C.compose(self.rho(dst_f), self.tensor_mor(f, id_I)) != C.compose(f, self.rho(src_f)):
                raise ValueError(f"Right unitor naturality fails for f={f}.")

    def _check_pentagon(self):
        """
        Pentagon: α_{A,B,C⊗D} ∘ α_{A⊗B,C,D}
                = (id_A ⊗ α_{B,C,D}) ∘ α_{A,B⊗C,D} ∘ (α_{A,B,C} ⊗ id_D)

        Both sides: ((A⊗B)⊗C)⊗D → A⊗(B⊗(C⊗D)).
        """
        C = self.category
        for A in C.objects:
            for B in C.objects:
                for Cv in C.objects:
                    for D in C.objects:
                        lhs = C.compose(
                            self.alpha(A, B, self.tensor_obj(Cv, D)),
                            self.alpha(self.tensor_obj(A, B), Cv, D),
                        )
                        rhs = C.compose(
                            self.tensor_mor(C.identities[A], self.alpha(B, Cv, D)),
                            C.compose(
                                self.alpha(A, self.tensor_obj(B, Cv), D),
                                self.tensor_mor(self.alpha(A, B, Cv), C.identities[D]),
                            ),
                        )
                        if lhs != rhs:
                            raise ValueError(
                                f"Pentagon coherence fails for A={A}, B={B}, C={Cv}, D={D}."
                            )

    def _check_triangle(self):
        """
        Triangle: (id_A ⊗ λ_B) ∘ α_{A,I,B} = ρ_A ⊗ id_B

        Both sides: (A⊗I)⊗B → A⊗B.
        """
        C = self.category
        I = self.unit
        for A in C.objects:
            for B in C.objects:
                lhs = C.compose(
                    self.tensor_mor(C.identities[A], self.lambda_(B)),
                    self.alpha(A, I, B),
                )
                rhs = self.tensor_mor(self.rho(A), C.identities[B])
                if lhs != rhs:
                    raise ValueError(f"Triangle coherence fails for A={A}, B={B}.")


class FiniteSymmetricMonoidalCategory(FiniteMonoidalCategory):
    """
    Finite symmetric monoidal category (C, ⊗, I, α, λ, ρ, γ).

    Adds a braiding γ_{A,B} : A⊗B → B⊗A satisfying:
    - Type correctness
    - Involutivity: γ_{B,A} ∘ γ_{A,B} = id_{A⊗B}
    - Naturality: γ_{A',B'} ∘ (f⊗g) = (g⊗f) ∘ γ_{A,B}
    - Hexagon: α_{B,C,A} ∘ γ_{A,B⊗C} ∘ α_{A,B,C}
               = (id_B ⊗ γ_{A,C}) ∘ α_{B,A,C} ∘ (γ_{A,B} ⊗ id_C)

    The hexagon is stated without α⁻¹ — both paths go
    (A⊗B)⊗C → B⊗(C⊗A).
    """

    def __init__(
        self,
        category: FiniteCategory,
        tensor_objects,
        tensor_morphisms,
        unit: str,
        associator,
        left_unitor,
        right_unitor,
        braiding,
    ):
        self.braiding = dict(braiding)
        super().__init__(
            category, tensor_objects, tensor_morphisms, unit,
            associator, left_unitor, right_unitor,
        )
        self._check_braiding_types()
        self._check_braiding_involutivity()
        self._check_braiding_naturality()
        self._check_hexagon()

    def gamma(self, A: str, B: str) -> str:
        """Return γ_{A,B} : A⊗B → B⊗A."""
        try:
            return self.braiding[(A, B)]
        except KeyError as exc:
            raise ValueError(f"Braiding γ_{{{A},{B}}} is not defined.") from exc

    def _check_braiding_types(self):
        C = self.category
        for A in C.objects:
            for B in C.objects:
                gam = self.gamma(A, B)
                AB = self.tensor_obj(A, B)
                BA = self.tensor_obj(B, A)
                if C.morphisms.get(gam) != (AB, BA):
                    raise ValueError(
                        f"Braiding γ_{{{A},{B}}} has type {C.morphisms.get(gam)}, "
                        f"expected ({AB} → {BA})."
                    )

    def _check_braiding_involutivity(self):
        """γ_{B,A} ∘ γ_{A,B} = id_{A⊗B}"""
        C = self.category
        for A in C.objects:
            for B in C.objects:
                AB = self.tensor_obj(A, B)
                if C.compose(self.gamma(B, A), self.gamma(A, B)) != C.identities[AB]:
                    raise ValueError(f"Braiding involutivity fails for A={A}, B={B}.")

    def _check_braiding_naturality(self):
        """γ_{A',B'} ∘ (f⊗g) = (g⊗f) ∘ γ_{A,B}"""
        C = self.category
        for f, (sf, df) in C.morphisms.items():
            for g, (sg, dg) in C.morphisms.items():
                if C.compose(self.gamma(df, dg), self.tensor_mor(f, g)) != C.compose(
                    self.tensor_mor(g, f), self.gamma(sf, sg)
                ):
                    raise ValueError(f"Braiding naturality fails for f={f}, g={g}.")

    def _check_hexagon(self):
        """
        Hexagon: α_{B,C,A} ∘ γ_{A,B⊗C} ∘ α_{A,B,C}
               = (id_B ⊗ γ_{A,C}) ∘ α_{B,A,C} ∘ (γ_{A,B} ⊗ id_C)

        Both sides: (A⊗B)⊗C → B⊗(C⊗A).
        """
        C = self.category
        for A in C.objects:
            for B in C.objects:
                for Cv in C.objects:
                    BC = self.tensor_obj(B, Cv)
                    lhs = C.compose(
                        self.alpha(B, Cv, A),
                        C.compose(self.gamma(A, BC), self.alpha(A, B, Cv)),
                    )
                    rhs = C.compose(
                        self.tensor_mor(C.identities[B], self.gamma(A, Cv)),
                        C.compose(
                            self.alpha(B, A, Cv),
                            self.tensor_mor(self.gamma(A, B), C.identities[Cv]),
                        ),
                    )
                    if lhs != rhs:
                        raise ValueError(
                            f"Hexagon coherence fails for A={A}, B={B}, C={Cv}."
                        )


def strict_monoidal_from_monoid(objects, tensor_table, unit, name_tensor=None):
    """
    Build a strict symmetric monoidal discrete category from a commutative monoid.

    A discrete category has only identity morphisms, so all structure maps are
    identities and all coherence conditions hold trivially.

    Parameters
    ----------
    objects : sequence of str
        The objects (= elements of the monoid).
    tensor_table : dict[(str, str), str]
        The monoid multiplication table A ⊗ B.
    unit : str
        The monoid identity element.
    name_tensor : callable, optional
        How to name the tensor morphism id_A ⊗ id_B. Defaults to
        ``lambda f, g: f + "_x_" + g``.

    Returns
    -------
    FiniteSymmetricMonoidalCategory
    """
    objects = tuple(objects)
    if name_tensor is None:
        def name_tensor(f, g):
            return f"{f}_x_{g}"

    morphisms = {f"id_{o}": (o, o) for o in objects}
    identities = {o: f"id_{o}" for o in objects}
    composition = {(f"id_{o}", f"id_{o}"): f"id_{o}" for o in objects}
    category = FiniteCategory(objects, morphisms, identities, composition)

    tensor_objects = dict(tensor_table)
    tensor_morphisms = {
        (f"id_{A}", f"id_{B}"): f"id_{tensor_table[(A, B)]}"
        for A in objects
        for B in objects
    }
    associator = {
        (A, B, C): f"id_{tensor_table[(tensor_table[(A, B)], C)]}"
        for A in objects
        for B in objects
        for C in objects
    }
    left_unitor = {A: f"id_{A}" for A in objects}
    right_unitor = {A: f"id_{A}" for A in objects}
    braiding = {
        (A, B): f"id_{tensor_table[(A, B)]}"
        for A in objects
        for B in objects
    }

    return FiniteSymmetricMonoidalCategory(
        category=category,
        tensor_objects=tensor_objects,
        tensor_morphisms=tensor_morphisms,
        unit=unit,
        associator=associator,
        left_unitor=left_unitor,
        right_unitor=right_unitor,
        braiding=braiding,
    )
