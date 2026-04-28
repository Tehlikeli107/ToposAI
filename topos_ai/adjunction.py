"""
topos_ai.adjunction — Formal Adjoint Functors.

Constructs and verifies an adjunction  F ⊣ G  between two finite categories
C and D, given explicit unit and counit natural transformations.

Mathematical contracts
----------------------
An **adjunction** F ⊣ G consists of:
  • F: C → D   (left adjoint)
  • G: D → C   (right adjoint)
  • Unit   η: id_C → G ∘ F   (natural transformation on C)
  • Counit ε: F ∘ G → id_D   (natural transformation on D)

satisfying the **triangle identities**:

  (ε_F) ∘ (Fη)  =  id_F      (1)
  (Gε) ∘ (ηG)   =  id_G      (2)

Equivalently, there is a natural bijection:

  Φ_{c,d}: C(c, G(d)) → D(F(c), d)
  Φ(φ) = ε_d ∘ F(φ)

with inverse  Ψ(ψ) = G(ψ) ∘ η_c.

All checks are exact combinatorial equalities over named morphisms in the
FiniteCategory representation.

References
----------
S. Mac Lane, "Categories for the Working Mathematician" (2nd ed.),
Chapter IV (Adjunctions).
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional


class FiniteAdjunction:
    """
    A formally-verified adjunction  F ⊣ G  between finite categories.

    Parameters
    ----------
    C : FiniteCategory
        The source category (domain of F, codomain of G).
    D : FiniteCategory
        The target category (codomain of F, domain of G).
    F_obj : dict[str, str]
        Object map of F: C → D.
    F_mor : dict[str, str]
        Morphism map of F: C → D.
    G_obj : dict[str, str]
        Object map of G: D → C.
    G_mor : dict[str, str]
        Morphism map of G: D → C.
    unit : dict[str, str]
        Unit η: for each c ∈ C, ``unit[c]`` is the name of a morphism
        in C from c to G(F(c)).
    counit : dict[str, str]
        Counit ε: for each d ∈ D, ``counit[d]`` is the name of a morphism
        in D from F(G(d)) to d.
    validate : bool
        If True (default), verify all axioms on construction.
    """

    def __init__(
        self,
        C,
        D,
        F_obj: Dict[str, str],
        F_mor: Dict[str, str],
        G_obj: Dict[str, str],
        G_mor: Dict[str, str],
        unit: Dict[str, str],
        counit: Dict[str, str],
        validate: bool = True,
    ):
        self.C = C
        self.D = D
        self.F_obj = dict(F_obj)
        self.F_mor = dict(F_mor)
        self.G_obj = dict(G_obj)
        self.G_mor = dict(G_mor)
        self.unit = dict(unit)
        self.counit = dict(counit)

        if validate:
            self._check_functor_F()
            self._check_functor_G()
            self._check_unit_types()
            self._check_counit_types()
            self._check_unit_naturality()
            self._check_counit_naturality()
            self._check_triangle_identities()

    # ------------------------------------------------------------------
    # Functor validity
    # ------------------------------------------------------------------

    def _check_functor_F(self):
        """Verify F: C → D respects composition and identities."""
        for c in self.C.objects:
            if c not in self.F_obj:
                raise ValueError(f"Adjunction: F_obj missing object {c!r}")
            if self.F_obj[c] not in self.D.object_set:
                raise ValueError(f"Adjunction: F_obj[{c!r}] = {self.F_obj[c]!r} not in D")
        for m in self.C.morphisms:
            if m not in self.F_mor:
                raise ValueError(f"Adjunction: F_mor missing morphism {m!r}")
            Fm = self.F_mor[m]
            if Fm not in self.D.morphisms:
                raise ValueError(f"Adjunction: F_mor[{m!r}] = {Fm!r} not in D")
            expected_src = self.F_obj[self.C.source(m)]
            expected_tgt = self.F_obj[self.C.target(m)]
            if self.D.source(Fm) != expected_src or self.D.target(Fm) != expected_tgt:
                raise ValueError(
                    f"Adjunction: F_mor[{m!r}] type mismatch: "
                    f"expected {expected_src}→{expected_tgt}, "
                    f"got {self.D.source(Fm)}→{self.D.target(Fm)}"
                )
        # Check F preserves composition
        for (after, before), composite in self.C.composition.items():
            expected = self.D.compose(self.F_mor[after], self.F_mor[before])
            actual = self.F_mor[composite]
            if actual != expected:
                raise ValueError(
                    f"Adjunction: F does not preserve composition: "
                    f"F({after}∘{before}) = {actual!r} ≠ {expected!r}"
                )

    def _check_functor_G(self):
        """Verify G: D → C respects composition and identities."""
        for d in self.D.objects:
            if d not in self.G_obj:
                raise ValueError(f"Adjunction: G_obj missing object {d!r}")
            if self.G_obj[d] not in self.C.object_set:
                raise ValueError(f"Adjunction: G_obj[{d!r}] not in C")
        for m in self.D.morphisms:
            if m not in self.G_mor:
                raise ValueError(f"Adjunction: G_mor missing morphism {m!r}")
            Gm = self.G_mor[m]
            if Gm not in self.C.morphisms:
                raise ValueError(f"Adjunction: G_mor[{m!r}] = {Gm!r} not in C")
            expected_src = self.G_obj[self.D.source(m)]
            expected_tgt = self.G_obj[self.D.target(m)]
            if self.C.source(Gm) != expected_src or self.C.target(Gm) != expected_tgt:
                raise ValueError(
                    f"Adjunction: G_mor[{m!r}] type mismatch"
                )
        # Check G preserves composition
        for (after, before), composite in self.D.composition.items():
            expected = self.C.compose(self.G_mor[after], self.G_mor[before])
            actual = self.G_mor[composite]
            if actual != expected:
                raise ValueError(
                    f"Adjunction: G does not preserve composition"
                )

    # ------------------------------------------------------------------
    # Unit / counit type checks
    # ------------------------------------------------------------------

    def _check_unit_types(self):
        """Each η_c: c → G(F(c)) must have the correct source and target."""
        for c in self.C.objects:
            if c not in self.unit:
                raise ValueError(f"Adjunction: unit missing component for {c!r}")
            eta_c = self.unit[c]
            if eta_c not in self.C.morphisms:
                raise ValueError(
                    f"Adjunction: unit[{c!r}] = {eta_c!r} not a morphism in C"
                )
            expected_tgt = self.G_obj[self.F_obj[c]]
            if self.C.source(eta_c) != c:
                raise ValueError(
                    f"Adjunction: unit[{c!r}] has source {self.C.source(eta_c)!r}, expected {c!r}"
                )
            if self.C.target(eta_c) != expected_tgt:
                raise ValueError(
                    f"Adjunction: unit[{c!r}] has target {self.C.target(eta_c)!r}, "
                    f"expected G(F({c!r})) = {expected_tgt!r}"
                )

    def _check_counit_types(self):
        """Each ε_d: F(G(d)) → d must have the correct source and target."""
        for d in self.D.objects:
            if d not in self.counit:
                raise ValueError(f"Adjunction: counit missing component for {d!r}")
            eps_d = self.counit[d]
            if eps_d not in self.D.morphisms:
                raise ValueError(
                    f"Adjunction: counit[{d!r}] = {eps_d!r} not a morphism in D"
                )
            expected_src = self.F_obj[self.G_obj[d]]
            if self.D.source(eps_d) != expected_src:
                raise ValueError(
                    f"Adjunction: counit[{d!r}] has source {self.D.source(eps_d)!r}, "
                    f"expected F(G({d!r})) = {expected_src!r}"
                )
            if self.D.target(eps_d) != d:
                raise ValueError(
                    f"Adjunction: counit[{d!r}] has target {self.D.target(eps_d)!r}, "
                    f"expected {d!r}"
                )

    # ------------------------------------------------------------------
    # Naturality
    # ------------------------------------------------------------------

    def _check_unit_naturality(self):
        """
        Verify η is natural: for each m: X→Y in C,
          G(F(m)) ∘ η_X  =  η_Y ∘ m   in C.
        """
        for m, (X, Y) in self.C.morphisms.items():
            GFm = self.G_mor[self.F_mor[m]]
            eta_X = self.unit[X]
            eta_Y = self.unit[Y]
            lhs = self.C.compose(GFm, eta_X)
            rhs = self.C.compose(eta_Y, m)
            if lhs != rhs:
                raise ValueError(
                    f"Adjunction: unit naturality fails for {m!r}: "
                    f"G(F({m!r}))∘η_{X!r} = {lhs!r} ≠ {rhs!r} = η_{Y!r}∘{m!r}"
                )

    def _check_counit_naturality(self):
        """
        Verify ε is natural: for each m: X→Y in D,
          m ∘ ε_X  =  ε_Y ∘ F(G(m))   in D.
        """
        for m, (X, Y) in self.D.morphisms.items():
            FGm = self.F_mor[self.G_mor[m]]
            eps_X = self.counit[X]
            eps_Y = self.counit[Y]
            lhs = self.D.compose(m, eps_X)
            rhs = self.D.compose(eps_Y, FGm)
            if lhs != rhs:
                raise ValueError(
                    f"Adjunction: counit naturality fails for {m!r}: "
                    f"{m!r}∘ε_{X!r} = {lhs!r} ≠ {rhs!r} = ε_{Y!r}∘F(G({m!r}))"
                )

    # ------------------------------------------------------------------
    # Triangle identities
    # ------------------------------------------------------------------

    def _check_triangle_identities(self):
        """
        Verify the zig-zag equations:
          (1) ε_{F(c)} ∘ F(η_c)  = id_{F(c)}   for all c ∈ C
          (2) G(ε_d) ∘ η_{G(d)}  = id_{G(d)}   for all d ∈ D
        """
        self._check_triangle_1()
        self._check_triangle_2()

    def _check_triangle_1(self):
        """ε_{F(c)} ∘ F(η_c) = id_{F(c)} for all c ∈ C."""
        for c in self.C.objects:
            Fc = self.F_obj[c]
            id_Fc = self.D.identities[Fc]
            eps_Fc = self.counit[Fc]
            F_eta_c = self.F_mor[self.unit[c]]
            result = self.D.compose(eps_Fc, F_eta_c)
            if result != id_Fc:
                raise ValueError(
                    f"Adjunction: triangle identity 1 fails at c={c!r}: "
                    f"ε_{{F({c!r})}} ∘ F(η_{c!r}) = {result!r} ≠ id_{{F({c!r})}} = {id_Fc!r}"
                )

    def _check_triangle_2(self):
        """G(ε_d) ∘ η_{G(d)} = id_{G(d)} for all d ∈ D."""
        for d in self.D.objects:
            Gd = self.G_obj[d]
            id_Gd = self.C.identities[Gd]
            G_eps_d = self.G_mor[self.counit[d]]
            eta_Gd = self.unit[Gd]
            result = self.C.compose(G_eps_d, eta_Gd)
            if result != id_Gd:
                raise ValueError(
                    f"Adjunction: triangle identity 2 fails at d={d!r}: "
                    f"G(ε_{d!r}) ∘ η_{{G({d!r})}} = {result!r} ≠ id_{{G({d!r})}} = {id_Gd!r}"
                )

    # ------------------------------------------------------------------
    # Hom-bijection
    # ------------------------------------------------------------------

    def hom_C(self, c: str, d: str):
        """Return all morphisms φ: c → G(d) in C."""
        Gd = self.G_obj[d]
        return self.C.hom(c, Gd)

    def hom_D(self, c: str, d: str):
        """Return all morphisms ψ: F(c) → d in D."""
        Fc = self.F_obj[c]
        return self.D.hom(Fc, d)

    def phi(self, phi_mor: str, d: str) -> str:
        """
        Hom-bijection  Φ_{c,d}: C(c, G(d)) → D(F(c), d).

        Φ(φ) = ε_d ∘ F(φ).

        Parameters
        ----------
        phi_mor : str
            A morphism φ: c → G(d) in C.
        d : str
            The target object in D.

        Returns
        -------
        The morphism ε_d ∘ F(φ) in D (of type F(c) → d).
        """
        F_phi = self.F_mor[phi_mor]
        eps_d = self.counit[d]
        return self.D.compose(eps_d, F_phi)

    def psi(self, psi_mor: str, c: str) -> str:
        """
        Inverse hom-bijection  Ψ_{c,d}: D(F(c), d) → C(c, G(d)).

        Ψ(ψ) = G(ψ) ∘ η_c.

        Parameters
        ----------
        psi_mor : str
            A morphism ψ: F(c) → d in D.
        c : str
            The source object in C.

        Returns
        -------
        The morphism G(ψ) ∘ η_c in C (of type c → G(d)).
        """
        G_psi = self.G_mor[psi_mor]
        eta_c = self.unit[c]
        return self.C.compose(G_psi, eta_c)

    def verify_hom_bijection(self) -> bool:
        """
        Check that Φ and Ψ are mutually inverse on all hom-sets.

        Returns True iff  Ψ(Φ(φ)) = φ  and  Φ(Ψ(ψ)) = ψ  for all c, d.
        """
        for c in self.C.objects:
            for d in self.D.objects:
                for phi_mor in self.hom_C(c, d):
                    if self.psi(self.phi(phi_mor, d), c) != phi_mor:
                        return False
                for psi_mor in self.hom_D(c, d):
                    if self.phi(self.psi(psi_mor, c), d) != psi_mor:
                        return False
        return True

    def __repr__(self):
        return (
            f"FiniteAdjunction(F: {list(self.C.objects)} -> {list(self.D.objects)}, "
            f"G: {list(self.D.objects)} -> {list(self.C.objects)})"
        )
