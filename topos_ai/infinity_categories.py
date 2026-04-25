from dataclasses import dataclass
from itertools import product

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


class InfinityCategoryLayer(_Module):
    """
    Hodge message-passing layer on nodes and edges.

    The class name is kept for API compatibility, but the implemented math is
    finite simplicial/Hodge processing rather than a full infinity-category.
    """

    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        self.W0 = nn.Linear(node_dim, out_dim)
        self.W1 = nn.Linear(edge_dim, out_dim)

    def forward(self, H0, H1, L0, L1):
        H0_new = torch.relu(self.W0(L0 @ H0))
        H1_new = torch.relu(self.W1(L1 @ H1))
        return H0_new, H1_new