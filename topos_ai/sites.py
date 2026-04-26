"""
topos_ai.sites — Grothendieck Sites and Sheaf Condition.

Implements Grothendieck topologies on finite categories and the exact
sheaf condition (equalizer / unique gluing) for set-valued presheaves.

Mathematical contracts
----------------------
**Sieve** on d ∈ C:
  A set S of morphisms with codomain d, closed under pre-composition:
  if f: e → d ∈ S and h: e' → e, then  f ∘ h ∈ S.

**Grothendieck topology** J on C:
  An assignment  d ↦ J(d)  (covering sieves on d) satisfying:
  • Maximality:  the maximal sieve M(d) = all morphisms to d is in J(d)
  • Stability:   if S ∈ J(d) and f: e → d, then  f*S ∈ J(e)
                 where  f*S = {h: ? → e | f ∘ h ∈ S}
  • Transitivity: if S ∈ J(d) and a sieve R on d satisfies
                 f*R ∈ J(source(f)) for all f ∈ S, then R ∈ J(d)

**Presheaf** F: C^op → FinSet:
  • objects_map: d ↦ F(d)
  • restriction_map: f: e→d ↦ F(f): F(d) → F(e)  (contravariant!)

**Sheaf condition** for F on (C, J):
  For every d ∈ C and covering sieve S ∈ J(d):
  A *matching family* is a family  {s_f ∈ F(source(f)) | f ∈ S} such that
  for all f ∈ S and all composable h: F(h)(s_f) = s_{f ∘ h}.
  F is a sheaf iff every matching family has a *unique* amalgamation
  a ∈ F(d) with  F(f)(a) = s_f  for all f ∈ S.

References
----------
SGA 4 (Artin, Grothendieck, Verdier); Mac Lane & Moerdijk "Sheaves in
Geometry and Logic" (1992), Chapters II–III.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Sieve
# ---------------------------------------------------------------------------

class Sieve:
    """
    A sieve on an object in a finite category.

    Parameters
    ----------
    on : str
        The object d this sieve lives on.
    morphisms : set[str]
        Names of morphisms with codomain d.
    category : FiniteCategory
        The ambient category.
    validate : bool
        If True, check that all given morphisms have codomain ``on``
        and that the set is closed under pre-composition.
    """

    def __init__(self, on: str, morphisms: Set[str], category, validate: bool = True):
        self.on = on
        self.morphisms: FrozenSet[str] = frozenset(morphisms)
        self.category = category
        if validate:
            self._validate()

    # ------------------------------------------------------------------
    def _validate(self):
        cat = self.category
        for m in self.morphisms:
            if m not in cat.morphisms:
                raise ValueError(f"Sieve: morphism {m!r} not in category")
            if cat.target(m) != self.on:
                raise ValueError(
                    f"Sieve: morphism {m!r} has codomain {cat.target(m)!r}, "
                    f"expected {self.on!r}"
                )
        # Closure: for every f ∈ S and every h composable with f on the right,
        # f∘h must be in S.
        for f in self.morphisms:
            f_src = cat.source(f)
            for h, (h_src, h_tgt) in cat.morphisms.items():
                if h_tgt != f_src:
                    continue
                try:
                    fh = cat.compose(f, h)
                except ValueError:
                    continue
                if fh not in self.morphisms:
                    raise ValueError(
                        f"Sieve on {self.on!r} is not closed: "
                        f"{f!r} ∈ S but {f!r} ∘ {h!r} = {fh!r} ∉ S"
                    )

    # ------------------------------------------------------------------
    def __contains__(self, m: str) -> bool:
        return m in self.morphisms

    def __len__(self) -> int:
        return len(self.morphisms)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Sieve):
            return NotImplemented
        return self.on == other.on and self.morphisms == other.morphisms

    def __hash__(self):
        return hash((self.on, self.morphisms))

    def __repr__(self):
        return f"Sieve(on={self.on!r}, morphisms={set(self.morphisms)!r})"

    # ------------------------------------------------------------------
    def pullback(self, f: str) -> "Sieve":
        """
        Base-change (pullback) of this sieve along  f: e → self.on.

        f*S  =  { h: ? → e  |  self.on-morphism (f ∘ h) ∈ S }

        The result is a sieve on  source(f).
        """
        cat = self.category
        if cat.target(f) != self.on:
            raise ValueError(
                f"pullback: morphism {f!r} has codomain {cat.target(f)!r}, "
                f"expected {self.on!r}"
            )
        e = cat.source(f)
        pulled: Set[str] = set()
        for h, (h_src, h_tgt) in cat.morphisms.items():
            if h_tgt != e:
                continue
            try:
                fh = cat.compose(f, h)
            except ValueError:
                continue
            if fh in self.morphisms:
                pulled.add(h)
        return Sieve(on=e, morphisms=pulled, category=cat, validate=False)

    # ------------------------------------------------------------------
    def is_closed(self) -> bool:
        """Return True if the morphism set is closed under pre-composition."""
        try:
            self._validate()
            return True
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def maximal_sieve(category, d: str) -> Sieve:
    """Return the maximal sieve on d: all morphisms with codomain d."""
    mors = {m for m, (_, tgt) in category.morphisms.items() if tgt == d}
    return Sieve(on=d, morphisms=mors, category=category, validate=False)


def empty_sieve(category, d: str) -> Sieve:
    """Return the empty sieve on d (no morphisms)."""
    return Sieve(on=d, morphisms=set(), category=category, validate=True)


# ---------------------------------------------------------------------------
# GrothendieckTopology
# ---------------------------------------------------------------------------

class GrothendieckTopology:
    """
    A Grothendieck topology on a finite category.

    Parameters
    ----------
    category : FiniteCategory
        The site's underlying category C.
    covering_sieves : dict[str, list[Sieve]]
        Maps each object d to its covering sieves J(d).
    validate : bool
        If True (default), check maximality, stability, and transitivity.
    """

    def __init__(self, category, covering_sieves: Dict[str, List[Sieve]], validate: bool = True):
        self.category = category
        self.covering_sieves: Dict[str, List[Sieve]] = dict(covering_sieves)
        if validate:
            self._validate()

    # ------------------------------------------------------------------
    def covers(self, d: str, sieve: Sieve) -> bool:
        """Return True if ``sieve`` is a covering sieve on d."""
        return any(sieve == s for s in self.covering_sieves.get(d, []))

    # ------------------------------------------------------------------
    def _validate(self):
        cat = self.category
        self._check_maximality()
        self._check_stability()
        self._check_transitivity()

    def _check_maximality(self):
        """Every maximal sieve must be covering."""
        cat = self.category
        for d in cat.objects:
            max_s = maximal_sieve(cat, d)
            if not self.covers(d, max_s):
                raise ValueError(
                    f"GrothendieckTopology: maximality fails — maximal sieve on "
                    f"{d!r} is not covering"
                )

    def _check_stability(self):
        """Covering sieves must be stable under base change."""
        cat = self.category
        for d in cat.objects:
            for s in self.covering_sieves.get(d, []):
                # For every morphism f: e → d
                for f, (e, tgt) in cat.morphisms.items():
                    if tgt != d:
                        continue
                    pulled = s.pullback(f)
                    if not self.covers(e, pulled):
                        raise ValueError(
                            f"GrothendieckTopology: stability fails — pullback of "
                            f"covering sieve {s} along {f!r} is not covering on {e!r}"
                        )

    def _check_transitivity(self):
        """
        Check transitivity: if S ∈ J(d) and a sieve R on d satisfies
        f*R ∈ J(source(f)) for all f ∈ S, then R ∈ J(d).
        """
        cat = self.category
        for d in cat.objects:
            all_sieves_on_d = _all_sieves(cat, d)
            for s in self.covering_sieves.get(d, []):
                for r in all_sieves_on_d:
                    if self.covers(d, r):
                        continue  # already covering, nothing to check
                    # Check if f*R ∈ J(source(f)) for all f ∈ S
                    all_pull_cover = all(
                        self.covers(cat.source(f), r.pullback(f))
                        for f in s.morphisms
                    )
                    if all_pull_cover:
                        raise ValueError(
                            f"GrothendieckTopology: transitivity fails — sieve "
                            f"{r} should be covering on {d!r} by transitivity "
                            f"(via covering sieve {s})"
                        )

    # ------------------------------------------------------------------
    def __repr__(self):
        summary = {d: len(sieves) for d, sieves in self.covering_sieves.items()}
        return f"GrothendieckTopology(covers={summary})"


def _all_sieves(category, d: str) -> List[Sieve]:
    """Enumerate all sieves on d by taking all subsets closed under pre-composition."""
    cat = category
    # Candidates: all morphisms with codomain d
    candidates = [m for m, (_, tgt) in cat.morphisms.items() if tgt == d]
    result = []
    # Check every subset
    for mask in range(1 << len(candidates)):
        subset = {candidates[i] for i in range(len(candidates)) if mask & (1 << i)}
        s = Sieve(on=d, morphisms=subset, category=cat, validate=False)
        if s.is_closed():
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# GrothendieckSite
# ---------------------------------------------------------------------------

class GrothendieckSite:
    """
    A Grothendieck site: a category equipped with a Grothendieck topology.

    Parameters
    ----------
    category : FiniteCategory
    topology : GrothendieckTopology
    """

    def __init__(self, category, topology: GrothendieckTopology):
        self.category = category
        self.topology = topology

    def __repr__(self):
        return (
            f"GrothendieckSite(objects={list(self.category.objects)}, "
            f"topology={self.topology})"
        )


def trivial_topology(category) -> GrothendieckTopology:
    """
    The trivial (or coarsest) Grothendieck topology: only maximal sieves cover.

    This makes every presheaf a sheaf.
    """
    covering = {d: [maximal_sieve(category, d)] for d in category.objects}
    return GrothendieckTopology(category, covering, validate=False)


def discrete_topology(category) -> GrothendieckTopology:
    """
    The discrete (or finest) Grothendieck topology: all sieves cover.

    Only the terminal presheaf (constant singleton) is a sheaf.
    """
    covering = {d: _all_sieves(category, d) for d in category.objects}
    return GrothendieckTopology(category, covering, validate=False)


# ---------------------------------------------------------------------------
# Presheaf on a site
# ---------------------------------------------------------------------------

class FinitePresheaf:
    """
    A set-valued presheaf F: C^op → FinSet.

    Parameters
    ----------
    category : FiniteCategory
        The underlying category C.
    objects_map : dict[str, frozenset]
        F(d) for each object d ∈ C.
    restriction_map : dict[str, dict]
        For each morphism f: e → d,  F(f): F(d) → F(e)  (restriction).
        Key is the morphism name; value is a dict  {x ∈ F(d): F(f)(x) ∈ F(e)}.
    validate : bool
        If True (default), run type-checking.
    """

    def __init__(
        self,
        *,
        category,
        objects_map: Dict[str, FrozenSet],
        restriction_map: Dict[str, Dict],
        validate: bool = True,
    ):
        self.category = category
        self.objects_map: Dict[str, FrozenSet] = {k: frozenset(v) for k, v in objects_map.items()}
        self.restriction_map: Dict[str, Dict] = dict(restriction_map)
        if validate:
            self._validate()

    # ------------------------------------------------------------------
    def _validate(self):
        cat = self.category
        for d in cat.objects:
            if d not in self.objects_map:
                raise ValueError(f"FinitePresheaf: missing objects_map entry for {d!r}")
        for mor in cat.morphisms:
            if mor not in self.restriction_map:
                raise ValueError(f"FinitePresheaf: missing restriction_map entry for {mor!r}")
            src = cat.source(mor)   # e
            tgt = cat.target(mor)   # d  (f: e → d)
            res = self.restriction_map[mor]
            for x in self.objects_map[tgt]:   # domain is F(d)
                if x not in res:
                    raise ValueError(
                        f"FinitePresheaf: restriction_map[{mor!r}] missing element {x!r}"
                    )
                if res[x] not in self.objects_map[src]:
                    raise ValueError(
                        f"FinitePresheaf: restriction_map[{mor!r}]({x!r}) ∉ F({src!r})"
                    )
        # Check functoriality: F(g∘f) = F(f)∘F(g)
        for (after, before), composite in cat.composition.items():
            f_res = self.restriction_map[before]   # F(before): F(tgt(before)) → F(src(before))
            g_res = self.restriction_map[after]     # F(after):  F(tgt(after))  → F(src(after))
            composite_res = self.restriction_map[composite]
            tgt_comp = cat.target(composite)        # = cat.target(after)
            for x in self.objects_map[tgt_comp]:
                lhs = f_res.get(g_res.get(x))
                rhs = composite_res.get(x)
                if lhs != rhs:
                    raise ValueError(
                        f"FinitePresheaf: functoriality fails on ({after!r} ∘ {before!r}): "
                        f"F({before!r})(F({after!r})({x!r})) = {lhs!r} ≠ {rhs!r}"
                    )

    # ------------------------------------------------------------------
    def restrict(self, mor: str, x: Any) -> Any:
        """Apply the restriction map F(mor) to element x ∈ F(target(mor))."""
        return self.restriction_map[mor][x]

    def sections(self, d: str) -> FrozenSet:
        """Return F(d)."""
        return self.objects_map[d]

    def __repr__(self):
        sizes = {d: len(s) for d, s in self.objects_map.items()}
        return f"FinitePresheaf(sections={sizes})"


# ---------------------------------------------------------------------------
# Sheaf condition
# ---------------------------------------------------------------------------

def matching_families(presheaf: FinitePresheaf, sieve: Sieve) -> List[Dict[str, Any]]:
    """
    Enumerate all matching families for (presheaf, sieve).

    A matching family is a dict  {f: s_f ∈ F(source(f)) | f ∈ sieve}  such
    that for every f ∈ sieve and every composable h (f∘h ∈ sieve),
    F(h)(s_f) = s_{f∘h}.

    Returns a list of such dicts.
    """
    cat = presheaf.category
    d = sieve.on
    mor_list = sorted(sieve.morphisms)  # stable order

    if not mor_list:
        return [{}]

    domains = [sorted(presheaf.sections(cat.source(m)), key=str) for m in mor_list]

    result = []
    for assignment in product(*domains):
        s = dict(zip(mor_list, assignment))
        valid = True
        for f in mor_list:
            for h, (_, h_tgt) in cat.morphisms.items():
                if h_tgt != cat.source(f):
                    continue
                try:
                    fh = cat.compose(f, h)
                except ValueError:
                    continue
                if fh not in sieve:
                    continue
                # F(h)(s_f) must equal s_{f∘h}
                Fh_sf = presheaf.restrict(h, s[f])
                if Fh_sf != s[fh]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            result.append(s)
    return result


def amalgamations(
    presheaf: FinitePresheaf, sieve: Sieve, family: Dict[str, Any]
) -> List[Any]:
    """
    Find all amalgamations of a matching family.

    An amalgamation is an element  a ∈ F(sieve.on)  such that
    F(f)(a) = s_f  for every f ∈ sieve.

    Returns the list of all such amalgamations.
    """
    cat = presheaf.category
    d = sieve.on
    result = []
    for a in presheaf.sections(d):
        ok = True
        for f in sieve.morphisms:
            if presheaf.restrict(f, a) != family[f]:
                ok = False
                break
        if ok:
            result.append(a)
    return result


def is_sheaf(presheaf: FinitePresheaf, site: GrothendieckSite) -> bool:
    """
    Check whether ``presheaf`` is a sheaf on ``site``.

    For every covering sieve S ∈ J(d) and every matching family for (F, S),
    there must be exactly one amalgamation in F(d).

    Returns True iff F is a sheaf.
    """
    cat = site.category
    for d in cat.objects:
        for sieve in site.topology.covering_sieves.get(d, []):
            for family in matching_families(presheaf, sieve):
                amals = amalgamations(presheaf, sieve, family)
                if len(amals) != 1:
                    return False
    return True


def sheaf_condition_failure(
    presheaf: FinitePresheaf, site: GrothendieckSite
) -> Optional[dict]:
    """
    If ``presheaf`` is not a sheaf, return a dict describing the first failure:
    ``{"object": d, "sieve": S, "family": f, "amalgamations": [...]}``
    Otherwise return None.
    """
    cat = site.category
    for d in cat.objects:
        for sieve in site.topology.covering_sieves.get(d, []):
            for family in matching_families(presheaf, sieve):
                amals = amalgamations(presheaf, sieve, family)
                if len(amals) != 1:
                    return {
                        "object": d,
                        "sieve": sieve,
                        "family": family,
                        "amalgamations": amals,
                    }
    return None


# ---------------------------------------------------------------------------
# Subobject classifier Ω for the trivial site
# ---------------------------------------------------------------------------

def omega_presheaf(category) -> FinitePresheaf:
    """
    Build the subobject classifier Ω for the trivial topology.

    Ω(d) = { all sieves on d }  (as frozenset labels).
    The restriction map Ω(f): Ω(d) → Ω(e)  for f: e → d
    sends a sieve S on d to the pullback  f*S  on e.

    This is the canonical Ω for any Grothendieck site with the trivial
    topology; it is always a sheaf.
    """
    cat = category
    all_s: Dict[str, List[Sieve]] = {d: _all_sieves(cat, d) for d in cat.objects}
    # Represent each sieve by its morphism frozenset (hashable label)
    obj_map: Dict[str, FrozenSet] = {
        d: frozenset(s.morphisms for s in sieves)
        for d, sieves in all_s.items()
    }
    # Build restriction maps
    res_map: Dict[str, Dict] = {}
    for mor, (e, d) in cat.morphisms.items():
        # Ω(mor): Ω(d) → Ω(e)
        sieve_by_label = {s.morphisms: s for s in all_s[d]}
        res: Dict[FrozenSet, FrozenSet] = {}
        for lbl, s in sieve_by_label.items():
            pulled = s.pullback(mor)
            res[lbl] = pulled.morphisms
        res_map[mor] = res
    return FinitePresheaf(
        category=cat,
        objects_map=obj_map,
        restriction_map=res_map,
        validate=False,
    )
