from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except ImportError:
    torch = None
    _Module = object


@dataclass(frozen=True)
class FiniteHorn:
    """A finite horn Lambda^n_k -> X, given by all faces except `k`."""

    dimension: int
    missing_face: int
    faces: dict


class FiniteSimplicialSet:
    """
    Finite simplicial-set skeleton with explicit face maps.

    This supports computational quasi-category checks by enumerating finite
    inner horns and asking whether each has at least one filler.
    """

    def __init__(self, simplices, faces, degeneracies=None):
        self.simplices = {int(dim): tuple(values) for dim, values in simplices.items()}
        self.max_dimension = max(self.simplices, default=-1)
        for dimension in range(self.max_dimension + 1):
            self.simplices.setdefault(dimension, ())
        self.simplex_sets = {dimension: frozenset(values) for dimension, values in self.simplices.items()}
        self.faces = dict(faces)
        self.degeneracies = dict(degeneracies or {})
        self.validate()

    def face(self, dimension, simplex, index):
        """Return `d_index(simplex)` for an n-simplex."""
        try:
            return self.faces[(dimension, simplex, index)]
        except KeyError as exc:
            raise ValueError(f"Missing face d_{index} for {dimension}-simplex {simplex!r}.") from exc

    def degeneracy(self, dimension, simplex, index):
        """Return `s_index(simplex)` for an n-simplex when degeneracies are supplied."""
        try:
            return self.degeneracies[(dimension, simplex, index)]
        except KeyError as exc:
            raise ValueError(f"Missing degeneracy s_{index} for {dimension}-simplex {simplex!r}.") from exc

    def validate(self):
        if 0 not in self.simplices:
            raise ValueError("A finite simplicial set needs a 0-simplex level.")

        for dimension, values in self.simplices.items():
            if dimension < 0:
                raise ValueError("Simplicial dimensions must be non-negative.")
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate simplex labels in dimension {dimension}.")

        for dimension in range(1, self.max_dimension + 1):
            for simplex in self.simplices[dimension]:
                for index in range(dimension + 1):
                    value = self.face(dimension, simplex, index)
                    if value not in self.simplex_sets[dimension - 1]:
                        raise ValueError(f"Face d_{index} of {simplex!r} is not a {dimension - 1}-simplex.")

        expected_faces = {
            (dimension, simplex, index)
            for dimension in range(1, self.max_dimension + 1)
            for simplex in self.simplices[dimension]
            for index in range(dimension + 1)
        }
        if set(self.faces) != expected_faces:
            missing = expected_faces - set(self.faces)
            extra = set(self.faces) - expected_faces
            raise ValueError(f"Face table mismatch. missing={missing}, extra={extra}")

        if self.degeneracies:
            expected_degeneracies = {
                (dimension, simplex, index)
                for dimension in range(self.max_dimension)
                for simplex in self.simplices[dimension]
                for index in range(dimension + 1)
            }
            if set(self.degeneracies) != expected_degeneracies:
                missing = expected_degeneracies - set(self.degeneracies)
                extra = set(self.degeneracies) - expected_degeneracies
                raise ValueError(f"Degeneracy table mismatch. missing={missing}, extra={extra}")

            for (dimension, simplex, index), value in self.degeneracies.items():
                if dimension < 0 or index < 0 or index > dimension:
                    raise ValueError(f"Invalid degeneracy index s_{index} for dimension {dimension}.")
                if simplex not in self.simplex_sets.get(dimension, frozenset()):
                    raise ValueError(f"Degeneracy source {simplex!r} is not a {dimension}-simplex.")
                if value not in self.simplex_sets.get(dimension + 1, frozenset()):
                    raise ValueError(f"Degeneracy target {value!r} is not a {dimension + 1}-simplex.")

        self.validate_face_identities()
        if self.degeneracies:
            self.validate_degeneracy_identities()
        return True

    def validate_face_identities(self):
        """Check `d_i d_j = d_{j-1} d_i` for all `i < j`."""
        for dimension in range(2, self.max_dimension + 1):
            for simplex in self.simplices[dimension]:
                for i in range(dimension + 1):
                    for j in range(i + 1, dimension + 1):
                        left = self.face(dimension - 1, self.face(dimension, simplex, j), i)
                        right = self.face(dimension - 1, self.face(dimension, simplex, i), j - 1)
                        if left != right:
                            raise ValueError(f"Face identity d_{i} d_{j} fails on {simplex!r}.")
        return True

    def validate_degeneracy_identities(self):
        """Check face-degeneracy and degeneracy-degeneracy simplicial identities."""
        if not self.degeneracies:
            return True

        for dimension in range(self.max_dimension):
            for simplex in self.simplices[dimension]:
                for j in range(dimension + 1):
                    degenerate = self.degeneracy(dimension, simplex, j)
                    for i in range(dimension + 2):
                        left = self.face(dimension + 1, degenerate, i)
                        if i < j:
                            right = self.degeneracy(dimension - 1, self.face(dimension, simplex, i), j - 1)
                        elif i == j or i == j + 1:
                            right = simplex
                        else:
                            right = self.degeneracy(dimension - 1, self.face(dimension, simplex, i - 1), j)
                        if left != right:
                            raise ValueError(f"Face-degeneracy identity d_{i} s_{j} fails on {simplex!r}.")

        for dimension in range(self.max_dimension - 1):
            for simplex in self.simplices[dimension]:
                for i in range(dimension + 1):
                    for j in range(i, dimension + 1):
                        left = self.degeneracy(
                            dimension + 1,
                            self.degeneracy(dimension, simplex, j),
                            i,
                        )
                        right = self.degeneracy(
                            dimension + 1,
                            self.degeneracy(dimension, simplex, i),
                            j + 1,
                        )
                        if left != right:
                            raise ValueError(f"Degeneracy identity s_{i} s_{j} fails on {simplex!r}.")

        return True

    def _validate_horn(self, horn: FiniteHorn):
        dimension = horn.dimension
        missing_face = horn.missing_face
        if dimension < 1:
            raise ValueError("Horns start in dimension 1.")
        if missing_face < 0 or missing_face > dimension:
            raise ValueError("Horn missing face must be between 0 and n.")
        expected_faces = set(range(dimension + 1)) - {missing_face}
        if set(horn.faces) != expected_faces:
            raise ValueError("Horn must provide exactly all faces except the missing one.")
        for index, face in horn.faces.items():
            if face not in self.simplex_sets.get(dimension - 1, frozenset()):
                raise ValueError(f"Horn face {index} is not a {dimension - 1}-simplex.")
        for i in expected_faces:
            for j in expected_faces:
                if i >= j:
                    continue
                left = self.face(dimension - 1, horn.faces[j], i)
                right = self.face(dimension - 1, horn.faces[i], j - 1)
                if left != right:
                    raise ValueError(f"Horn compatibility fails for faces {i} and {j}.")
        return True

    def horn_from_simplex(self, dimension, simplex, missing_face):
        """Return the horn obtained by deleting one face from a simplex boundary."""
        if simplex not in self.simplex_sets.get(dimension, frozenset()):
            raise ValueError(f"{simplex!r} is not a {dimension}-simplex.")
        return FiniteHorn(
            dimension=dimension,
            missing_face=missing_face,
            faces={
                index: self.face(dimension, simplex, index)
                for index in range(dimension + 1)
                if index != missing_face
            },
        )

    def horn_fillers(self, horn: FiniteHorn):
        """Enumerate n-simplices whose boundary fills the supplied horn."""
        self._validate_horn(horn)
        fillers = []
        for simplex in self.simplices.get(horn.dimension, ()):
            if all(self.face(horn.dimension, simplex, index) == face for index, face in horn.faces.items()):
                fillers.append(simplex)
        return tuple(fillers)

    def compatible_horns(self, dimension, missing_face):
        """Enumerate compatible finite horns of a fixed shape."""
        if dimension < 1:
            return ()
        face_indices = tuple(index for index in range(dimension + 1) if index != missing_face)
        horns = []
        for face_values in product(self.simplices.get(dimension - 1, ()), repeat=len(face_indices)):
            horn = FiniteHorn(
                dimension=dimension,
                missing_face=missing_face,
                faces=dict(zip(face_indices, face_values)),
            )
            try:
                self._validate_horn(horn)
            except ValueError:
                continue
            horns.append(horn)
        return tuple(horns)

    def missing_inner_horns(self, max_dimension=None):
        """Return inner horns with no filler, up to `max_dimension`."""
        if max_dimension is None:
            max_dimension = self.max_dimension
        missing = []
        for dimension in range(2, min(max_dimension, self.max_dimension) + 1):
            for missing_face in range(1, dimension):
                for horn in self.compatible_horns(dimension, missing_face):
                    if not self.horn_fillers(horn):
                        missing.append(horn)
        return tuple(missing)

    def is_inner_kan(self, max_dimension=None):
        """Finite quasi-category check: every enumerated inner horn has a filler."""
        return len(self.missing_inner_horns(max_dimension=max_dimension)) == 0

    def has_unique_inner_horn_fillers(self, max_dimension=None):
        """Return True when every enumerated inner horn has exactly one filler."""
        if max_dimension is None:
            max_dimension = self.max_dimension
        for dimension in range(2, min(max_dimension, self.max_dimension) + 1):
            for missing_face in range(1, dimension):
                for horn in self.compatible_horns(dimension, missing_face):
                    if len(self.horn_fillers(horn)) != 1:
                        return False
        return True


def nerve_2_skeleton(category):
    """
    Build the 2-skeleton of the nerve of a finite 1-category.

    Inner 2-horn fillers encode categorical composition: a composable pair
    `f: x -> y`, `g: y -> z` is filled by the 2-simplex `(g, f)`.
    """
    simplices = {
        0: tuple(("obj", obj) for obj in category.objects),
        1: tuple(("mor", morphism) for morphism in category.morphisms),
        2: tuple(("comp", after, before) for after, before in category.composable_pairs()),
    }
    faces = {}
    for morphism, (src, dst) in category.morphisms.items():
        edge = ("mor", morphism)
        faces[(1, edge, 0)] = ("obj", dst)
        faces[(1, edge, 1)] = ("obj", src)

    for after, before in category.composable_pairs():
        triangle = ("comp", after, before)
        faces[(2, triangle, 0)] = ("mor", after)
        faces[(2, triangle, 1)] = ("mor", category.compose(after, before))
        faces[(2, triangle, 2)] = ("mor", before)

    degeneracies = {
        (0, ("obj", obj), 0): ("mor", identity)
        for obj, identity in category.identities.items()
    }
    for morphism, (src, dst) in category.morphisms.items():
        edge = ("mor", morphism)
        degeneracies[(1, edge, 0)] = ("comp", morphism, category.identities[src])
        degeneracies[(1, edge, 1)] = ("comp", category.identities[dst], morphism)

    return FiniteSimplicialSet(simplices=simplices, faces=faces, degeneracies=degeneracies)


def nerve_3_skeleton(category):
    """
    Build the 3-skeleton of the nerve of a finite 1-category.

    The 3-simplices encode associativity coherence for triples of composable
    arrows; their middle faces compare the two parenthesizations.
    """
    triples = []
    for first in category.morphisms:
        for second in category.morphisms:
            if category.target(first) != category.source(second):
                continue
            for third in category.morphisms:
                if category.target(second) == category.source(third):
                    triples.append(("assoc", third, second, first))

    simplices = {
        0: tuple(("obj", obj) for obj in category.objects),
        1: tuple(("mor", morphism) for morphism in category.morphisms),
        2: tuple(("comp", after, before) for after, before in category.composable_pairs()),
        3: tuple(triples),
    }
    faces = {}
    for morphism, (src, dst) in category.morphisms.items():
        edge = ("mor", morphism)
        faces[(1, edge, 0)] = ("obj", dst)
        faces[(1, edge, 1)] = ("obj", src)

    for after, before in category.composable_pairs():
        triangle = ("comp", after, before)
        faces[(2, triangle, 0)] = ("mor", after)
        faces[(2, triangle, 1)] = ("mor", category.compose(after, before))
        faces[(2, triangle, 2)] = ("mor", before)

    for _tag, third, second, first in simplices[3]:
        tetrahedron = ("assoc", third, second, first)
        faces[(3, tetrahedron, 0)] = ("comp", third, second)
        faces[(3, tetrahedron, 1)] = ("comp", third, category.compose(second, first))
        faces[(3, tetrahedron, 2)] = ("comp", category.compose(third, second), first)
        faces[(3, tetrahedron, 3)] = ("comp", second, first)

    degeneracies = {
        (0, ("obj", obj), 0): ("mor", identity)
        for obj, identity in category.identities.items()
    }
    for morphism, (src, dst) in category.morphisms.items():
        edge = ("mor", morphism)
        degeneracies[(1, edge, 0)] = ("comp", morphism, category.identities[src])
        degeneracies[(1, edge, 1)] = ("comp", category.identities[dst], morphism)

    for after, before in category.composable_pairs():
        triangle = ("comp", after, before)
        src = category.source(before)
        middle = category.target(before)
        dst = category.target(after)
        degeneracies[(2, triangle, 0)] = ("assoc", after, before, category.identities[src])
        degeneracies[(2, triangle, 1)] = ("assoc", after, category.identities[middle], before)
        degeneracies[(2, triangle, 2)] = ("assoc", category.identities[dst], after, before)

    return FiniteSimplicialSet(simplices=simplices, faces=faces, degeneracies=degeneracies)


class SimplicialComplexBuilder:
    """Build an epsilon-neighborhood simplicial complex from a point cloud."""

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def build_complex(self, X):
        """
        Return edges, triangles, and an edge index map.

        This is a finite simplicial-complex proxy. It is inspired by higher
        categorical language but does not model infinity-categorical coherence
        data such as horn fillers or higher composition laws.
        """
        dist_matrix = torch.cdist(X, X, p=2)
        adj = (dist_matrix < self.epsilon).float()
        adj.fill_diagonal_(0.0)

        edges = torch.nonzero(torch.triu(adj)).tolist()
        edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}

        triangles = []
        for i, j in edges:
            common_neighbors = torch.nonzero(adj[i] * adj[j]).squeeze(-1).tolist()
            for k in common_neighbors:
                if k > j:
                    triangles.append((i, j, k))

        return edges, triangles, edge_to_idx


class HodgeLaplacianEngine:
    """Boundary matrices and Hodge Laplacians for a finite complex."""

    def __init__(self, num_nodes, edges, triangles, edge_to_idx):
        self.V = num_nodes
        self.E = len(edges)
        self.T = len(triangles)

        self.B1 = torch.zeros((self.V, self.E), dtype=torch.float32)
        self.B2 = torch.zeros((self.E, self.T), dtype=torch.float32)

        self._build_boundaries(edges, triangles, edge_to_idx)

    def _build_boundaries(self, edges, triangles, edge_to_idx):
        for e_idx, (i, j) in enumerate(edges):
            self.B1[i, e_idx] = -1.0
            self.B1[j, e_idx] = 1.0

        for t_idx, (i, j, k) in enumerate(triangles):
            e_ij = edge_to_idx[(i, j)]
            e_jk = edge_to_idx[(j, k)]
            e_ik = edge_to_idx[(i, k)]
            self.B2[e_ij, t_idx] = 1.0
            self.B2[e_jk, t_idx] = 1.0
            self.B2[e_ik, t_idx] = -1.0

    def get_laplacians(self):
        L0 = self.B1 @ self.B1.T
        L1 = self.B1.T @ self.B1 + self.B2 @ self.B2.T
        return L0, L1


class FormalInfinityCategoryValidator:
    """
    Strict Formal Infinity-Category Engine (Quasi-Category inner horn filler).

    Replaces the old continuous/Hodge message passing approximation with 
    100% pure, discrete mathematical logic. Evaluates if a given Simplicial 
    Complex (a set of 0-simplices, 1-simplices, and 2-simplices) satisfies 
    the Inner Kan Extension condition for quasi-categories.

    Specifically, it verifies that every "inner horn" (Λ^2_1) - which is two 
    composable arrows f: x -> y and g: y -> z - has a valid "filler" (a 
    2-simplex that acts as the composition g o f : x -> z).
    """
    def __init__(self, simplices_0, simplices_1, simplices_2):
        self.s0 = set(simplices_0) # e.g. { 'x', 'y', 'z' }
        self.s1 = set(simplices_1) # e.g. { ('x','y'), ('y','z') }
        self.s2 = set(simplices_2) # e.g. { ('x','y','z') }

    def find_missing_inner_horns(self):
        """
        Finds all pairs of composable 1-simplices (f: x->y, g: y->z) that 
        DO NOT have a corresponding 2-simplex filler (x, y, z) verifying 
        their composition.
        """
        missing_horns = []
        for f in self.s1:
            for g in self.s1:
                x, y1 = f
                y2, z = g
                if y1 == y2: # They are composable!
                    # The inner horn is (f, g) forming Λ^2_1(x, y, z)
                    filler_candidate = (x, y1, z)
                    if filler_candidate not in self.s2:
                        missing_horns.append((f, g))
        return missing_horns

    def is_valid_infinity_category(self):
        """
        An infinity-category (quasi-category) must have fillers for all inner horns.
        Returns True if the composition law holds strictly across all 2-simplices.
        """
        return len(self.find_missing_inner_horns()) == 0

    def enforce_strict_composition(self):
        """
        Autonomously generates (fills) the missing 1-simplices (transitive closure) 
        and 2-simplices (homotopies) to make the space a valid Infinity-Category.
        """
        added_1_simplices = 0
        added_2_simplices = 0

        changed = True
        while changed:
            changed = False
            missing = self.find_missing_inner_horns()
            for (x, y), (_, z) in missing:
                # Add the missing 1-simplex composition (h: x -> z)
                if (x, z) not in self.s1:
                    self.s1.add((x, z))
                    added_1_simplices += 1

                # Add the missing 2-simplex filler (x, y, z) representing the homotopy h ~ g o f
                filler = (x, y, z)
                if filler not in self.s2:
                    self.s2.add(filler)
                    added_2_simplices += 1
                    changed = True

        return added_1_simplices, added_2_simplices


# ------------------------------------------------------------------ #
# Quasi-category composition                                           #
# ------------------------------------------------------------------ #

def compose_in_quasicategory(
    simplicial_set: FiniteSimplicialSet,
    f,
    g,
) -> Tuple[object, ...]:
    """
    Compose two composable 1-simplices in a quasi-category.

    In a quasi-category X every inner Λ²₁-horn has at least one filler.
    Given 1-simplices  f : x → y  and  g : y → z  (with d₁(f) = d₀(g)),
    this function finds all 2-simplices σ such that:

        d₂(σ) = f   (tail morphism)
        d₀(σ) = g   (head morphism)

    and returns the list of *composites*  d₁(σ)  for each such filler.
    In a quasi-category these composites are all pairwise homotopic, so
    the composite is well-defined *up to homotopy*.

    Convention
    ----------
    The face maps in ``nerve_2_skeleton`` satisfy:
        d₀(comp, after, before) = after   (head)
        d₁(comp, after, before) = after∘before  (composite)
        d₂(comp, after, before) = before  (tail)

    Parameters
    ----------
    simplicial_set : FiniteSimplicialSet  (should be a quasi-category)
    f : label of a 1-simplex  (tail, i.e. the "before" morphism)
    g : label of a 1-simplex  (head, i.e. the "after" morphism)

    Returns
    -------
    composites : tuple of 1-simplex labels  [d₁(σ) for each inner horn filler σ]

    Raises
    ------
    ValueError
        If f or g are not 1-simplices, or if they are not composable
        (d₀(f) ≠ d₁(g) in the standard orientation), or if no filler exists.
    """
    if f not in simplicial_set.simplex_sets.get(1, frozenset()):
        raise ValueError(f"{f!r} is not a 1-simplex in the simplicial set.")
    if g not in simplicial_set.simplex_sets.get(1, frozenset()):
        raise ValueError(f"{g!r} is not a 1-simplex in the simplicial set.")

    # Check composability: d₀(f) must equal d₁(g)
    # d₀ of a 1-simplex is its "target" in our convention (index 0 = head)
    # d₁ of a 1-simplex is its "source" (index 1 = tail)
    f_target = simplicial_set.face(1, f, 0)   # d₀(f) = codomain of f
    g_source = simplicial_set.face(1, g, 1)   # d₁(g) = domain of g
    if f_target != g_source:
        raise ValueError(
            f"Morphisms are not composable: d₀({f!r}) = {f_target!r} ≠ d₁({g!r}) = {g_source!r}."
        )

    # Find all 2-simplices σ with d₂(σ) = f and d₀(σ) = g
    composites = []
    for sigma in simplicial_set.simplices.get(2, ()):
        if (
            simplicial_set.face(2, sigma, 2) == f
            and simplicial_set.face(2, sigma, 0) == g
        ):
            composite = simplicial_set.face(2, sigma, 1)
            composites.append(composite)

    if not composites:
        raise ValueError(
            f"No inner horn filler found for (d₂=={f!r}, d₀=={g!r}). "
            "The simplicial set may not be a quasi-category, or the required "
            "2-simplex is missing from this finite skeleton."
        )

    return tuple(composites)


def morphisms_are_homotopic(
    simplicial_set: FiniteSimplicialSet,
    f,
    g,
) -> bool:
    """
    Test whether two parallel 1-simplices are homotopic in a quasi-category.

    Two morphisms  f, g : x → y  are *homotopic* (rel endpoints) if there
    exists a 2-simplex  σ  satisfying one of the two standard conditions:

    Left homotopy:   d₀(σ) = s₀(y),  d₁(σ) = g,  d₂(σ) = f
    Right homotopy:  d₀(σ) = g,       d₁(σ) = f,  d₂(σ) = s₀(x)

    Since degeneracy maps may not be present, this implementation checks
    for a 2-simplex σ with d₂(σ) = f, d₀(σ) = g (left) or d₂(σ) = g,
    d₀(σ) = f (right), where the 1-simplices share both endpoints.

    Parameters
    ----------
    simplicial_set : FiniteSimplicialSet
    f, g           : labels of 1-simplices with the same source and target

    Returns
    -------
    bool
    """
    if f == g:
        return True

    # Verify parallel
    f_src = simplicial_set.face(1, f, 1)
    f_tgt = simplicial_set.face(1, f, 0)
    g_src = simplicial_set.face(1, g, 1)
    g_tgt = simplicial_set.face(1, g, 0)
    if f_src != g_src or f_tgt != g_tgt:
        return False

    for sigma in simplicial_set.simplices.get(2, ()):
        d0 = simplicial_set.face(2, sigma, 0)
        d1 = simplicial_set.face(2, sigma, 1)
        d2 = simplicial_set.face(2, sigma, 2)
        # Left homotopy: d₂=f, d₀=g (and d₁ ∈ hom(x, z) but here x=z=y boundary)
        if d2 == f and d0 == g:
            return True
        # Right homotopy: d₂=g, d₀=f
        if d2 == g and d0 == f:
            return True

    return False


def _homotopy_classes(
    simplicial_set: FiniteSimplicialSet,
    one_simplices,
) -> Dict[object, List[object]]:
    """
    Partition a list of 1-simplices into homotopy classes.

    Uses union-find to group morphisms that are pairwise homotopic.
    Returns a dict: representative → list of all equivalent members.
    """
    members = list(one_simplices)
    parent = {m: m for m in members}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, m1 in enumerate(members):
        for m2 in members[i + 1:]:
            if morphisms_are_homotopic(simplicial_set, m1, m2):
                union(m1, m2)

    classes: Dict[object, List[object]] = {}
    for m in members:
        root = find(m)
        classes.setdefault(root, []).append(m)
    return classes


def homotopy_category(
    simplicial_set: FiniteSimplicialSet,
) -> "FiniteCategory":  # noqa: F821
    """
    Construct the **homotopy category** Ho(X) of a quasi-category X.

    Ho(X) is the ordinary 1-category whose:
      - objects  = 0-simplices of X
      - morphisms = homotopy classes of 1-simplices of X
      - composition = [f] ∘ [g]  :=  [d₁(σ)]  for any inner 2-horn filler σ
                                      with  d₂(σ) ∈ [f],  d₀(σ) ∈ [g]

    Homotopies and composition are computed from the 2-skeleton of ``simplicial_set``.

    Parameters
    ----------
    simplicial_set : FiniteSimplicialSet  (quasi-category, inner-Kan complex)

    Returns
    -------
    FiniteCategory  representing Ho(X).

    Notes
    -----
    *Degenerate 1-simplices* serve as identity morphisms.  If degeneracy maps
    are available, the identity at object ``x`` is ``s₀(x)``.  Otherwise, any
    1-simplex  e : x → x  with  d₀(e) = d₁(e) = x  that arises as the
    composite of a morphism with its reverse is used.  As a last resort the
    first available endomorphism is chosen.
    """
    from .formal_category import FiniteCategory  # local import to avoid cycles

    objects_raw = list(simplicial_set.simplices.get(0, ()))
    one_simps = list(simplicial_set.simplices.get(1, ()))

    # Build source / target maps
    src_of = {e: simplicial_set.face(1, e, 1) for e in one_simps}
    tgt_of = {e: simplicial_set.face(1, e, 0) for e in one_simps}

    # Homotopy classes of morphisms between each (src, tgt) pair
    mor_classes: Dict[Tuple, List] = {}  # (src, tgt) → list of representatives
    all_classes = _homotopy_classes(simplicial_set, one_simps)

    # Canonical representative for each class: the first member (sorted for determinism)
    repr_of: Dict[object, object] = {}  # any 1-simplex → its class representative
    for rep, members in all_classes.items():
        for m in members:
            repr_of[m] = rep

    # For each (src, tgt) pair collect the set of class representatives
    hom_sets: Dict[Tuple, List] = {}
    for rep in all_classes:
        s = src_of[rep]
        t = tgt_of[rep]
        hom_sets.setdefault((s, t), []).append(rep)

    # Build morphism table for FiniteCategory
    # Morphism labels: (src_obj, tgt_obj, rep_simplex) — unique by construction
    morphisms: Dict[str, Tuple] = {}
    mor_label: Dict[object, str] = {}  # class rep → label string

    for rep in all_classes:
        s = src_of[rep]
        t = tgt_of[rep]
        label = f"[{rep!r}]"
        morphisms[label] = (s, t)
        mor_label[rep] = label

    # Identity morphisms
    # Prefer degenerate 1-simplices s₀(x) if degeneracy maps exist
    identities: Dict[object, str] = {}
    for x in objects_raw:
        # Try degeneracy-based identity first
        if simplicial_set.degeneracies:
            try:
                id_simp = simplicial_set.degeneracy(0, x, 0)
                id_rep = repr_of.get(id_simp, id_simp)
                if id_rep in mor_label:
                    identities[x] = mor_label[id_rep]
                    continue
            except (ValueError, KeyError):
                pass
        # Fall back: find the first endomorphism at x
        for rep in hom_sets.get((x, x), []):
            identities[x] = mor_label[rep]
            break
        if x not in identities:
            # No endomorphism found — create a formal identity
            label = f"[id_{x!r}]"
            morphisms[label] = (x, x)
            identities[x] = label

    # Composition table: compose from 2-simplices
    composition: Dict[Tuple[str, str], str] = {}

    # For each pair of composable representatives, find the composite class
    for rep_g in all_classes:
        src_g = src_of[rep_g]
        tgt_g = tgt_of[rep_g]
        for rep_f in all_classes:
            src_f = src_of[rep_f]
            tgt_f = tgt_of[rep_f]
            if tgt_f != src_g:
                continue
            # Try to fill the horn: find σ with d₂=rep_f, d₀=rep_g
            try:
                composites = compose_in_quasicategory(simplicial_set, rep_f, rep_g)
            except ValueError:
                continue
            # Take the class representative of the first composite
            comp_simp = composites[0]
            comp_rep = repr_of.get(comp_simp, comp_simp)
            if comp_rep not in mor_label:
                # The composite might not be a representative but should be in our table
                continue
            lbl_g = mor_label[rep_g]
            lbl_f = mor_label[rep_f]
            composition[(lbl_g, lbl_f)] = mor_label[comp_rep]

    # Fill identity composition axioms for any missing entries
    for x in objects_raw:
        id_x = identities[x]
        for rep in all_classes:
            lbl = mor_label[rep]
            src_r = src_of[rep]
            tgt_r = tgt_of[rep]
            if src_r == x:
                composition.setdefault((lbl, id_x), lbl)
            if tgt_r == x:
                composition.setdefault((id_x, lbl), lbl)
        composition.setdefault((id_x, id_x), id_x)

    return FiniteCategory(
        objects=objects_raw,
        morphisms=morphisms,
        identities=identities,
        composition=composition,
    )
