try:
    import torch
except ImportError:
    torch = None


class FinitePathGroupoid:
    """
    Finite groupoid semantics for 1-truncated HoTT identity types.

    Objects are terms/types-as-points, paths are identity proofs, and path
    composition follows the convention `compose(q, p) = q o p`.
    """

    def __init__(self, objects, paths, identities, inverses, composition):
        self.objects = tuple(objects)
        self.object_set = frozenset(self.objects)
        self.paths = dict(paths)
        self.identities = dict(identities)
        self.inverses = dict(inverses)
        self.composition = dict(composition)
        self.validate_groupoid_laws()

    def source(self, path):
        return self.paths[path][0]

    def target(self, path):
        return self.paths[path][1]

    def refl(self, obj):
        """Return the reflexivity proof `refl_obj : obj = obj`."""
        return self.identities[obj]

    def identity_type(self, source, target):
        """Return the finite identity type `Id(source, target)`."""
        return frozenset(path for path, (src, dst) in self.paths.items() if src == source and dst == target)

    def inverse(self, path):
        """Return the inverse path."""
        return self.inverses[path]

    def composable_pairs(self):
        for before, (_src_before, dst_before) in self.paths.items():
            for after, (src_after, _dst_after) in self.paths.items():
                if dst_before == src_after:
                    yield after, before

    def compose(self, after, before):
        if self.target(before) != self.source(after):
            raise ValueError(f"Paths are not composable: {after} o {before}.")
        try:
            return self.composition[(after, before)]
        except KeyError as exc:
            raise ValueError(f"Missing path composition for {after} o {before}.") from exc

    def validate_groupoid_laws(self):
        if set(self.identities) != set(self.objects):
            raise ValueError("Every groupoid object needs a reflexivity path.")
        if set(self.inverses) != set(self.paths):
            raise ValueError("Every path needs an inverse.")
        if set(self.composition) != set(self.composable_pairs()):
            raise ValueError("Path composition table must contain exactly the composable pairs.")

        for path, (src, dst) in self.paths.items():
            if src not in self.object_set or dst not in self.object_set:
                raise ValueError(f"Path {path!r} has an endpoint outside the object set.")

        for obj, identity in self.identities.items():
            if identity not in self.paths or self.paths[identity] != (obj, obj):
                raise ValueError(f"Reflexivity path for {obj!r} must have type {obj!r} -> {obj!r}.")

        for path, inverse in self.inverses.items():
            if inverse not in self.paths:
                raise ValueError(f"Inverse {inverse!r} is not a declared path.")
            if self.paths[inverse] != (self.target(path), self.source(path)):
                raise ValueError(f"Inverse {inverse!r} has the wrong endpoints for {path!r}.")
            if self.inverses[inverse] != path:
                raise ValueError(f"Inverse involution fails for {path!r}.")

        for (after, before), result in self.composition.items():
            if result not in self.paths:
                raise ValueError(f"Composite path {result!r} is not declared.")
            expected_type = (self.source(before), self.target(after))
            if self.paths[result] != expected_type:
                raise ValueError(f"Composite path {result!r} has type {self.paths[result]}, expected {expected_type}.")

        for path, (src, dst) in self.paths.items():
            if self.compose(path, self.identities[src]) != path:
                raise ValueError(f"Right identity path law fails for {path!r}.")
            if self.compose(self.identities[dst], path) != path:
                raise ValueError(f"Left identity path law fails for {path!r}.")
            if self.compose(self.inverse(path), path) != self.identities[src]:
                raise ValueError(f"Left inverse path law fails for {path!r}.")
            if self.compose(path, self.inverse(path)) != self.identities[dst]:
                raise ValueError(f"Right inverse path law fails for {path!r}.")

        for first in self.paths:
            for second in self.paths:
                if self.target(first) != self.source(second):
                    continue
                for third in self.paths:
                    if self.target(second) != self.source(third):
                        continue
                    left = self.compose(third, self.compose(second, first))
                    right = self.compose(self.compose(third, second), first)
                    if left != right:
                        raise ValueError(f"Path associativity fails for ({third}, {second}, {first}).")

        return True


class PathFamily:
    """
    Dependent family over a finite path groupoid.

    A transport map is supplied for every path and validated as a functor from
    the path groupoid into finite sets.
    """

    def __init__(self, base: FinitePathGroupoid, fibers, transports):
        self.base = base
        self.fibers = {obj: frozenset(values) for obj, values in fibers.items()}
        self.transports = {path: dict(mapping) for path, mapping in transports.items()}
        self.validate_functorial_transport()

    def transport(self, path, value):
        """Transport a value along a path proof."""
        try:
            return self.transports[path][value]
        except KeyError as exc:
            raise ValueError(f"No transport value for {path!r} at {value!r}.") from exc

    def validate_functorial_transport(self):
        if set(self.fibers) != set(self.base.objects):
            raise ValueError("A dependent family needs one fiber over every base object.")
        if set(self.transports) != set(self.base.paths):
            raise ValueError("A dependent family needs one transport map for every path.")

        for path, (src, dst) in self.base.paths.items():
            mapping = self.transports[path]
            if set(mapping) != set(self.fibers[src]):
                raise ValueError(f"Transport along {path!r} must be defined on the whole source fiber.")
            if not set(mapping.values()).issubset(self.fibers[dst]):
                raise ValueError(f"Transport along {path!r} must land in the target fiber.")

        for obj, identity in self.base.identities.items():
            for value in self.fibers[obj]:
                if self.transport(identity, value) != value:
                    raise ValueError(f"Transport along refl({obj!r}) must be identity.")

        for after, before in self.base.composable_pairs():
            composite = self.base.compose(after, before)
            src = self.base.source(before)
            for value in self.fibers[src]:
                left = self.transport(composite, value)
                right = self.transport(after, self.transport(before, value))
                if left != right:
                    raise ValueError(f"Transport functoriality fails for {after!r} o {before!r}.")

        return True

    def transport_equivalence(self, path):
        """Return the forward and inverse transport maps witnessed by a path."""
        inverse = self.base.inverse(path)
        return dict(self.transports[path]), dict(self.transports[inverse])

    def validate_transport_equivalences(self):
        """Check every path transport is a bijection with inverse-path transport."""
        self.validate_functorial_transport()
        for path in self.base.paths:
            forward, backward = self.transport_equivalence(path)
            src = self.base.source(path)
            dst = self.base.target(path)

            if set(forward) != set(self.fibers[src]) or set(backward) != set(self.fibers[dst]):
                return False
            if set(forward.values()) != set(self.fibers[dst]):
                return False
            if set(backward.values()) != set(self.fibers[src]):
                return False

            for value in self.fibers[src]:
                if backward[forward[value]] != value:
                    return False
            for value in self.fibers[dst]:
                if forward[backward[value]] != value:
                    return False

        return True


class FormalHomotopyEquivalence:
    """
    Strict Formal Homotopy Equivalence (Equivalence of Categories).

    Replaces the old continuous/numerical approximation with 100% pure, 
    discrete mathematical logic. Evaluates if two Finite Categories 
    (or Groupoids) are formally equivalent by finding an invertible 
    Functor (Strict Isomorphism) between them.
    
    This is an exact combinatorial proof engine for Univalence 
    (Identity is Equivalence) on finite structures.
    """
    
    def __init__(self, cat_A, cat_B):
        self.cat_A = cat_A
        self.cat_B = cat_B

    def find_strict_isomorphism(self):
        """
        Attempts to find a strict categorical isomorphism (F: A -> B, G: B -> A)
        where F and G are bijective on both objects and morphisms, and perfectly
        preserve the composition laws of the categories.
        
        Returns the object and morphism mapping dictionaries if successful, 
        else None. (O(N!) combinatorial search for small exact categories).
        """
        import itertools
        
        objects_A = list(self.cat_A.objects)
        objects_B = list(self.cat_B.objects)
        
        if len(objects_A) != len(objects_B):
            return None # Sizes must match for strict isomorphism
            
        morphisms_A = list(self.cat_A.paths if hasattr(self.cat_A, 'paths') else self.cat_A.morphisms.keys())
        morphisms_B = list(self.cat_B.paths if hasattr(self.cat_B, 'paths') else self.cat_B.morphisms.keys())
        
        if len(morphisms_A) != len(morphisms_B):
            return None

        # Helper to get src/dst based on whether it's a Groupoid or FiniteCategory
        def get_endpoints(cat, mor):
            if hasattr(cat, 'paths'):
                return cat.paths[mor]
            return cat.morphisms[mor]

        def get_comp(cat, f, g):
            if hasattr(cat, 'compose'):
                try: return cat.compose(f, g)
                except ValueError: return None
            return cat.composition.get((f, g))

        # Brute force search over all object bijections
        for obj_perm in itertools.permutations(objects_B):
            obj_map = dict(zip(objects_A, obj_perm))
            
            # For each valid object mapping, try to find a valid morphism mapping
            for mor_perm in itertools.permutations(morphisms_B):
                mor_map = dict(zip(morphisms_A, mor_perm))
                
                # Check Functoriality: F(f: X -> Y) must be F(f): F(X) -> F(Y)
                functorial = True
                for f in morphisms_A:
                    src_A, dst_A = get_endpoints(self.cat_A, f)
                    src_B, dst_B = get_endpoints(self.cat_B, mor_map[f])
                    
                    if obj_map[src_A] != src_B or obj_map[dst_A] != dst_B:
                        functorial = False
                        break
                
                if not functorial:
                    continue
                    
                # Check Composition Preservation: F(f o g) == F(f) o F(g)
                preserves_comp = True
                for f in morphisms_A:
                    for g in morphisms_A:
                        comp_A = get_comp(self.cat_A, f, g)
                        if comp_A is not None:
                            comp_B_expected = mor_map[comp_A]
                            comp_B_actual = get_comp(self.cat_B, mor_map[f], mor_map[g])
                            if comp_B_actual != comp_B_expected:
                                preserves_comp = False
                                break
                    if not preserves_comp:
                        break
                        
                if preserves_comp:
                    return obj_map, mor_map
                    
        return None

    def is_univalent_equivalent(self):
        """Returns True if the spaces are formally isomorphic."""
        return self.find_strict_isomorphism() is not None


class HomotopyEquivalence:
    """
    Numerical homotopy equivalence between two finite point clouds.

    Finds the rigid-body transformation (rotation R, translation t) that
    minimises  ‖R A + t - B‖_F  via singular value decomposition (Kabsch
    algorithm), and transports points along the resulting path.

    This is a *numerical* companion to ``FormalHomotopyEquivalence``: it does
    not validate categorical axioms but provides differentiable alignment.
    """

    def find_homotopy_path(self, space_a, space_b):
        """
        Compute rotation R and translation t such that  R @ A.T + t ≈ B.T.

        Parameters
        ----------
        space_a, space_b : Tensor of shape (n, d)

        Returns
        -------
        R : Tensor (d, d)
        t : Tensor (d,)

        Raises
        ------
        ValueError
            If the two spaces do not have the same shape.
        """
        if space_a.shape != space_b.shape:
            raise ValueError(
                "Both spaces must have the same shape to find a homotopy path."
            )
        # Centre the clouds
        centroid_a = space_a.mean(dim=0)
        centroid_b = space_b.mean(dim=0)
        a_c = space_a - centroid_a
        b_c = space_b - centroid_b

        # Kabsch: optimal rotation via SVD of covariance matrix
        H = a_c.T @ b_c
        U, _S, Vt = torch.linalg.svd(H)
        # Correct for reflections
        d = torch.det(Vt.T @ U.T)
        D = torch.diag(
            torch.cat([torch.ones(space_a.shape[1] - 1, device=space_a.device),
                       d.unsqueeze(0)])
        )
        R = Vt.T @ D @ U.T
        t = centroid_b - R @ centroid_a
        return R, t

    def transport_along_path(self, space_a, R, t):
        """
        Apply the rigid transformation: return R @ A.T transposed + t.

        Parameters
        ----------
        space_a : Tensor (n, d)
        R       : Tensor (d, d)
        t       : Tensor (d,)

        Returns
        -------
        Tensor (n, d)
        """
        return (R @ space_a.T).T + t
