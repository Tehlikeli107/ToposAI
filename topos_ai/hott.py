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


class HomotopyEquivalence:
    """
    Orthogonal Procrustes alignment inspired by HoTT path language.

    Given two point clouds with the same shape, this class finds an orthogonal
    map and translation that best align them in least-squares sense. It is a
    numerical alignment tool, not a proof of univalence or equivalence of
    arbitrary mathematical objects.
    """

    def find_homotopy_path(self, space_A, space_B):
        """
        Return (R, translation) so that space_A @ R.T + translation approximates
        space_B.
        """
        if space_A.shape != space_B.shape:
            raise ValueError("space_A and space_B must have the same shape.")

        mean_A = torch.mean(space_A, dim=0, keepdim=True)
        mean_B = torch.mean(space_B, dim=0, keepdim=True)

        centered_A = space_A - mean_A
        centered_B = space_B - mean_B

        covariance = centered_B.t() @ centered_A
        U, _, Vh = torch.linalg.svd(covariance, full_matrices=False)
        V = Vh.t()
        R = U @ V.t()

        if torch.det(R) < 0:
            U = U.clone()
            U[:, -1] *= -1.0
            R = U @ V.t()

        translation = mean_B.t() - R @ mean_A.t()
        return R, translation.squeeze()

    def transport_along_path(self, space_A, R, translation):
        """Apply the fitted alignment to space_A."""
        return space_A @ R.t() + translation.unsqueeze(0)
