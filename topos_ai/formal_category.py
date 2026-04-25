from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product


def _powerset(values):
    values = tuple(values)
    for size in range(len(values) + 1):
        for subset in combinations(values, size):
            yield frozenset(subset)


def _all_functions(domain, codomain):
    domain = tuple(domain)
    codomain = tuple(codomain)
    if not domain:
        return ({},)
    if not codomain:
        return ()
    return tuple(dict(zip(domain, outputs)) for outputs in product(codomain, repeat=len(domain)))


def _equivalence_class_map(elements, generating_pairs):
    elements = tuple(elements)
    parent = {element: element for element in elements}

    def find(element):
        while parent[element] != element:
            parent[element] = parent[parent[element]]
            element = parent[element]
        return element

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in generating_pairs:
        if left not in parent or right not in parent:
            raise ValueError("Equivalence generators must be elements of the quotient source set.")
        union(left, right)

    classes = {}
    for element in elements:
        classes.setdefault(find(element), set()).add(element)

    frozen_classes = {root: frozenset(values) for root, values in classes.items()}
    return {element: frozen_classes[find(element)] for element in elements}


class FiniteCategory:
    """
    Small finite category with explicit objects, morphisms, identities, and composition.

    Morphisms are named by hashable labels. Composition uses the convention
    `compose(g, f) = g o f`, so f must target the source of g.
    """

    def __init__(self, objects, morphisms, identities, composition):
        self.objects = tuple(objects)
        self.object_set = frozenset(self.objects)
        self.morphisms = dict(morphisms)
        self.identities = dict(identities)
        self.composition = dict(composition)
        self.validate_laws()

    def source(self, morphism):
        return self.morphisms[morphism][0]

    def target(self, morphism):
        return self.morphisms[morphism][1]

    def hom(self, source, target):
        return tuple(name for name, (src, dst) in self.morphisms.items() if src == source and dst == target)

    def arrows_to(self, target):
        return tuple(name for name, (_src, dst) in self.morphisms.items() if dst == target)

    def arrows_from(self, source):
        return tuple(name for name, (src, _dst) in self.morphisms.items() if src == source)

    def composable_pairs(self):
        for before, (_src_before, dst_before) in self.morphisms.items():
            for after, (src_after, _dst_after) in self.morphisms.items():
                if dst_before == src_after:
                    yield after, before

    def compose(self, after, before):
        if self.target(before) != self.source(after):
            raise ValueError(f"Morphisms are not composable: {after} o {before}.")
        try:
            return self.composition[(after, before)]
        except KeyError as exc:
            raise ValueError(f"Missing composition entry for {after} o {before}.") from exc

    def validate_laws(self):
        if set(self.identities) != set(self.objects):
            raise ValueError("Every object must have exactly one declared identity morphism.")

        for obj, identity in self.identities.items():
            if identity not in self.morphisms:
                raise ValueError(f"Identity morphism {identity!r} is not declared.")
            if self.morphisms[identity] != (obj, obj):
                raise ValueError(f"Identity {identity!r} must have type {obj} -> {obj}.")

        required_pairs = set(self.composable_pairs())
        if set(self.composition) != required_pairs:
            missing = required_pairs - set(self.composition)
            extra = set(self.composition) - required_pairs
            raise ValueError(f"Composition table mismatch. missing={missing}, extra={extra}")

        for (after, before), result in self.composition.items():
            if result not in self.morphisms:
                raise ValueError(f"Composite {result!r} is not declared as a morphism.")
            expected_type = (self.source(before), self.target(after))
            if self.morphisms[result] != expected_type:
                raise ValueError(f"Composite {result!r} has type {self.morphisms[result]}, expected {expected_type}.")

        for morphism, (src, dst) in self.morphisms.items():
            if self.compose(morphism, self.identities[src]) != morphism:
                raise ValueError(f"Right identity law fails for {morphism!r}.")
            if self.compose(self.identities[dst], morphism) != morphism:
                raise ValueError(f"Left identity law fails for {morphism!r}.")

        for f in self.morphisms:
            for g in self.morphisms:
                if self.target(f) != self.source(g):
                    continue
                for h in self.morphisms:
                    if self.target(g) != self.source(h):
                        continue
                    left = self.compose(h, self.compose(g, f))
                    right = self.compose(self.compose(h, g), f)
                    if left != right:
                        raise ValueError(f"Associativity fails for ({h}, {g}, {f}): {left!r} != {right!r}.")

        return True


class FiniteFunctor:
    """Functor between finite categories with explicit object and morphism maps."""

    def __init__(self, source: FiniteCategory, target: FiniteCategory, object_map, morphism_map):
        self.source = source
        self.target = target
        self.object_map = dict(object_map)
        self.morphism_map = dict(morphism_map)
        self.validate()

    def map_object(self, obj):
        return self.object_map[obj]

    def map_morphism(self, morphism):
        return self.morphism_map[morphism]

    def validate(self):
        if set(self.object_map) != set(self.source.objects):
            raise ValueError("Functor must map every source object.")
        if set(self.morphism_map) != set(self.source.morphisms):
            raise ValueError("Functor must map every source morphism.")
        if not set(self.object_map.values()).issubset(self.target.object_set):
            raise ValueError("Functor object map must land in the target category.")

        for morphism, (src, dst) in self.source.morphisms.items():
            mapped = self.morphism_map[morphism]
            if mapped not in self.target.morphisms:
                raise ValueError(f"Functor maps {morphism!r} to an undeclared target morphism.")
            expected_type = (self.object_map[src], self.object_map[dst])
            if self.target.morphisms[mapped] != expected_type:
                raise ValueError(f"Functor maps {morphism!r} to {self.target.morphisms[mapped]}, expected {expected_type}.")

        for obj, identity in self.source.identities.items():
            if self.morphism_map[identity] != self.target.identities[self.object_map[obj]]:
                raise ValueError(f"Functor does not preserve the identity on {obj!r}.")

        for after, before in self.source.composable_pairs():
            mapped_composite = self.morphism_map[self.source.compose(after, before)]
            composite_of_maps = self.target.compose(self.morphism_map[after], self.morphism_map[before])
            if mapped_composite != composite_of_maps:
                raise ValueError(f"Functor does not preserve composition for {after!r} o {before!r}.")

        return True


class Presheaf:
    """
    Finite presheaf on a finite category, represented as a functor C^op -> FinSet.

    For a morphism f: A -> B in C, `restrictions[f]` is the function
    F(B) -> F(A).
    """

    def __init__(self, category: FiniteCategory, sets, restrictions):
        self.category = category
        self.sets = {obj: frozenset(values) for obj, values in sets.items()}
        self.restrictions = {morphism: dict(mapping) for morphism, mapping in restrictions.items()}
        self.validate_functor_laws()

    def restrict(self, morphism, element):
        try:
            return self.restrictions[morphism][element]
        except KeyError as exc:
            raise ValueError(f"No restriction value for {morphism!r} at {element!r}.") from exc

    def validate_functor_laws(self):
        if set(self.sets) != set(self.category.objects):
            raise ValueError("Presheaf must assign a set to every object.")
        if set(self.restrictions) != set(self.category.morphisms):
            raise ValueError("Presheaf must assign a restriction map to every morphism.")

        for morphism, (src, dst) in self.category.morphisms.items():
            mapping = self.restrictions[morphism]
            if set(mapping) != set(self.sets[dst]):
                raise ValueError(f"Restriction {morphism!r} must be defined on F({dst}).")
            if not set(mapping.values()).issubset(self.sets[src]):
                raise ValueError(f"Restriction {morphism!r} must land in F({src}).")

        for obj, identity in self.category.identities.items():
            for element in self.sets[obj]:
                if self.restrict(identity, element) != element:
                    raise ValueError(f"Identity functor law fails on {obj!r}.")

        for first, second in self.category.composable_pairs():
            composite = self.category.compose(first, second)
            for element in self.sets[self.category.target(first)]:
                left = self.restrict(composite, element)
                right = self.restrict(second, self.restrict(first, element))
                if left != right:
                    raise ValueError(f"Contravariant functor law fails for {first} o {second}.")

        return True


@dataclass(frozen=True)
class NaturalTransformation:
    """Natural transformation between finite presheaves."""

    source: Presheaf
    target: Presheaf
    components: dict

    def apply(self, obj, element):
        return self.components[obj][element]

    def validate_naturality(self):
        if self.source.category is not self.target.category:
            raise ValueError("Natural transformations require the same base category object.")

        category = self.source.category
        if set(self.components) != set(category.objects):
            raise ValueError("A component is required for every object.")

        for obj in category.objects:
            mapping = self.components[obj]
            if set(mapping) != set(self.source.sets[obj]):
                raise ValueError(f"Component at {obj!r} must be defined on the whole source set.")
            if not set(mapping.values()).issubset(self.target.sets[obj]):
                raise ValueError(f"Component at {obj!r} must land in the target set.")

        for morphism, (src, dst) in category.morphisms.items():
            for element in self.source.sets[dst]:
                left = self.target.restrict(morphism, self.apply(dst, element))
                right = self.apply(src, self.source.restrict(morphism, element))
                if left != right:
                    raise ValueError(f"Naturality square fails for morphism {morphism!r}.")

        return True


def _freeze_components(category: FiniteCategory, components):
    frozen = []
    for obj in category.objects:
        items = tuple(sorted(components[obj].items(), key=lambda item: (repr(item[0]), repr(item[1]))))
        frozen.append((obj, items))
    return tuple(frozen)


@dataclass(frozen=True)
class FrozenNaturalTransformation:
    """
    Hashable natural-transformation value for finite exponential presheaves.

    It keeps source and target by object identity and stores component maps as
    tuples so transformations can be elements of finite sets.
    """

    source: Presheaf
    target: Presheaf
    components: tuple

    @classmethod
    def from_transformation(cls, transformation: NaturalTransformation):
        return cls(
            source=transformation.source,
            target=transformation.target,
            components=_freeze_components(transformation.source.category, transformation.components),
        )

    def _component_mapping(self, obj):
        for component_obj, items in self.components:
            if component_obj == obj:
                return dict(items)
        raise KeyError(obj)

    def apply(self, obj, element):
        return self._component_mapping(obj)[element]

    def thaw(self):
        components = {obj: dict(items) for obj, items in self.components}
        transformation = NaturalTransformation(source=self.source, target=self.target, components=components)
        transformation.validate_naturality()
        return transformation


def natural_transformations(source: Presheaf, target: Presheaf):
    """Enumerate all natural transformations between two finite presheaves."""
    if source.category is not target.category:
        raise ValueError("Natural transformations require the same base category object.")

    category = source.category
    component_choices = []
    for obj in category.objects:
        choices = _all_functions(source.sets[obj], target.sets[obj])
        if not choices:
            return ()
        component_choices.append((obj, choices))

    transformations = []
    for choice_tuple in product(*(choices for _obj, choices in component_choices)):
        components = {obj: mapping for (obj, _choices), mapping in zip(component_choices, choice_tuple)}
        transformation = NaturalTransformation(source=source, target=target, components=components)
        try:
            transformation.validate_naturality()
        except ValueError:
            continue
        transformations.append(transformation)

    return tuple(transformations)


def representable_presheaf(category: FiniteCategory, obj):
    """Return the Yoneda presheaf y(obj) = Hom(-, obj)."""
    sets = {source: frozenset(category.hom(source, obj)) for source in category.objects}
    restrictions = {}

    for morphism, (src, dst) in category.morphisms.items():
        mapping = {}
        for arrow_to_obj in sets[dst]:
            mapping[arrow_to_obj] = category.compose(arrow_to_obj, morphism)
        restrictions[morphism] = mapping

    return Presheaf(category, sets, restrictions)


def yoneda_element_to_transformation(category: FiniteCategory, obj, presheaf: Presheaf, element):
    """
    Map an element x in F(obj) to the natural transformation y(obj) -> F.

    At stage A, an arrow h: A -> obj is sent to F(h)(x).
    """
    if element not in presheaf.sets[obj]:
        raise ValueError(f"{element!r} is not an element of F({obj}).")

    source = representable_presheaf(category, obj)
    components = {}
    for stage in category.objects:
        components[stage] = {arrow: presheaf.restrict(arrow, element) for arrow in source.sets[stage]}

    transformation = NaturalTransformation(source=source, target=presheaf, components=components)
    transformation.validate_naturality()
    return transformation


def yoneda_transformation_to_element(obj, transformation: NaturalTransformation):
    """Evaluate alpha: y(obj) -> F at id_obj, yielding an element of F(obj)."""
    identity = transformation.source.category.identities[obj]
    return transformation.apply(obj, identity)


def yoneda_lemma_bijection(category: FiniteCategory, obj, presheaf: Presheaf):
    """
    Enumerate the finite Yoneda bijection Nat(y(obj), F) ~= F(obj).

    Returns a dictionary mapping each element of F(obj) to its corresponding
    natural transformation y(obj) -> F.
    """
    source = representable_presheaf(category, obj)
    transformations = natural_transformations(source, presheaf)
    by_element = {yoneda_transformation_to_element(obj, alpha): alpha for alpha in transformations}

    if set(by_element) != set(presheaf.sets[obj]) or len(by_element) != len(transformations):
        raise ValueError("Yoneda map failed to produce a bijection for this finite presheaf.")

    return by_element


def category_of_elements(presheaf: Presheaf):
    """
    Build the finite category of elements `int F` and its projection to `C`.

    Objects are pairs `(c, x)` with `x in F(c)`. A morphism
    `(c, F(f)(y)) -> (d, y)` is represented by `("element", f, y)`, where
    `f: c -> d` in the base category.
    """
    category = presheaf.category
    objects = tuple((obj, element) for obj in category.objects for element in presheaf.sets[obj])
    morphisms = {}

    for arrow, (src, dst) in category.morphisms.items():
        for target_element in presheaf.sets[dst]:
            source_element = presheaf.restrict(arrow, target_element)
            label = ("element", arrow, target_element)
            morphisms[label] = ((src, source_element), (dst, target_element))

    identities = {
        (obj, element): ("element", category.identities[obj], element)
        for obj, element in objects
    }

    composition = {}
    for before, (_before_src, before_dst) in morphisms.items():
        for after, (after_src, _after_dst) in morphisms.items():
            if before_dst != after_src:
                continue
            composite = ("element", category.compose(after[1], before[1]), after[2])
            composition[(after, before)] = composite

    elements = FiniteCategory(
        objects=objects,
        morphisms=morphisms,
        identities=identities,
        composition=composition,
    )
    projection = FiniteFunctor(
        source=elements,
        target=category,
        object_map={(obj, element): obj for obj, element in objects},
        morphism_map={label: label[1] for label in morphisms},
    )
    return elements, projection


def yoneda_density_colimit(presheaf: Presheaf):
    """
    Reconstruct `F` as the finite colimit of representables over `int F`.

    This is the density theorem form of Yoneda:
    `F ~= colim_{(c, x) in int F} y(c)`.
    """
    category = presheaf.category
    elements, _projection = category_of_elements(presheaf)

    class_maps = {}
    sets = {}
    for stage in category.objects:
        raw_elements = tuple(
            ((obj, element), arrow_to_obj)
            for obj, element in elements.objects
            for arrow_to_obj in category.hom(stage, obj)
        )
        generators = []
        for element_arrow, ((src_obj, source_element), (dst_obj, target_element)) in elements.morphisms.items():
            base_arrow = element_arrow[1]
            for arrow_to_src in category.hom(stage, src_obj):
                generators.append(
                    (
                        ((src_obj, source_element), arrow_to_src),
                        ((dst_obj, target_element), category.compose(base_arrow, arrow_to_src)),
                    )
                )

        class_maps[stage] = _equivalence_class_map(raw_elements, generators)
        sets[stage] = frozenset(class_maps[stage].values())

    restrictions = {}
    for morphism, (src, dst) in category.morphisms.items():
        mapping = {}
        for equivalence_class in sets[dst]:
            restricted_classes = {
                class_maps[src][((obj, element), category.compose(arrow_to_obj, morphism))]
                for (obj, element), arrow_to_obj in equivalence_class
            }
            if len(restricted_classes) != 1:
                raise ValueError("Yoneda density restriction is not well-defined.")
            mapping[equivalence_class] = next(iter(restricted_classes))
        restrictions[morphism] = mapping

    density = Presheaf(category=category, sets=sets, restrictions=restrictions)

    to_components = {}
    for stage in category.objects:
        component = {}
        for equivalence_class in density.sets[stage]:
            values = {
                presheaf.restrict(arrow_to_obj, element)
                for (obj, element), arrow_to_obj in equivalence_class
            }
            if len(values) != 1:
                raise ValueError("Yoneda density counit is not well-defined.")
            component[equivalence_class] = next(iter(values))
        to_components[stage] = component

    from_components = {
        stage: {
            element: class_maps[stage][((stage, element), category.identities[stage])]
            for element in presheaf.sets[stage]
        }
        for stage in category.objects
    }

    to_presheaf = NaturalTransformation(source=density, target=presheaf, components=to_components)
    from_presheaf = NaturalTransformation(source=presheaf, target=density, components=from_components)
    to_presheaf.validate_naturality()
    from_presheaf.validate_naturality()
    return density, to_presheaf, from_presheaf


class Subpresheaf:
    """Subobject of a finite presheaf, represented objectwise and checked under restrictions."""

    def __init__(self, parent: Presheaf, subsets):
        self.parent = parent
        self.subsets = {obj: frozenset(values) for obj, values in subsets.items()}
        self.validate()

    def validate(self):
        if set(self.subsets) != set(self.parent.category.objects):
            raise ValueError("Subpresheaf must provide a subset for every object.")

        for obj, subset in self.subsets.items():
            if not subset.issubset(self.parent.sets[obj]):
                raise ValueError(f"Subset at {obj!r} is not contained in the parent presheaf.")

        for morphism, (src, dst) in self.parent.category.morphisms.items():
            for element in self.subsets[dst]:
                restricted = self.parent.restrict(morphism, element)
                if restricted not in self.subsets[src]:
                    raise ValueError(f"Subpresheaf is not closed under restriction {morphism!r}.")

        return True


class PresheafTopos:
    """
    Finite fragment of the presheaf topos Set^(C^op).

    It implements sieves, the subobject classifier Omega, characteristic maps,
    and pullback along truth for finite presheaves.
    """

    def __init__(self, category: FiniteCategory):
        self.category = category

    def _require_presheaf(self, presheaf: Presheaf):
        if presheaf.category is not self.category:
            raise ValueError("Presheaf must be over this topos base category.")

    def natural_transformation(self, source: Presheaf, target: Presheaf, components):
        """Construct and validate a natural transformation in this presheaf category."""
        self._require_presheaf(source)
        self._require_presheaf(target)
        transformation = NaturalTransformation(source=source, target=target, components=components)
        transformation.validate_naturality()
        return transformation

    def reindex_presheaf(self, functor: FiniteFunctor, presheaf: Presheaf):
        """
        Inverse-image/reindexing of a presheaf along a finite functor.

        For u: C -> D and F in Set^(D^op), this returns u*F in Set^(C^op)
        with `(u*F)(c) = F(u c)` and restrictions induced by F(u f).
        """
        if functor.source is not self.category:
            raise ValueError("Reindexing must be called on the functor source topos.")
        if presheaf.category is not functor.target:
            raise ValueError("Presheaf must live on the functor target category.")

        return Presheaf(
            category=self.category,
            sets={obj: presheaf.sets[functor.map_object(obj)] for obj in self.category.objects},
            restrictions={
                morphism: dict(presheaf.restrictions[functor.map_morphism(morphism)])
                for morphism in self.category.morphisms
            },
        )

    def reindex_transformation(self, functor: FiniteFunctor, transformation: NaturalTransformation):
        """Reindex a natural transformation along a finite functor."""
        if transformation.source.category is not functor.target or transformation.target.category is not functor.target:
            raise ValueError("Transformation must live on the functor target category.")
        source = self.reindex_presheaf(functor, transformation.source)
        target = self.reindex_presheaf(functor, transformation.target)
        return self.natural_transformation(
            source=source,
            target=target,
            components={
                obj: dict(transformation.components[functor.map_object(obj)])
                for obj in self.category.objects
            },
        )

    def _left_kan_index(self, functor: FiniteFunctor, target_obj):
        return tuple(
            (source_obj, arrow)
            for source_obj in functor.source.objects
            for arrow in self.category.hom(target_obj, functor.map_object(source_obj))
        )

    def _left_kan_index_morphisms(self, functor: FiniteFunctor, target_obj):
        index = self._left_kan_index(functor, target_obj)
        for source_obj, arrow in index:
            for restricted_obj, restricted_arrow in index:
                for morphism in functor.source.hom(restricted_obj, source_obj):
                    if self.category.compose(functor.map_morphism(morphism), restricted_arrow) == arrow:
                        yield source_obj, arrow, restricted_obj, restricted_arrow, morphism

    def left_kan_extension_presheaf(self, functor: FiniteFunctor, presheaf: Presheaf):
        """
        Left adjoint to reindexing, computed as finite left Kan extension.

        For `u: C -> D` and `F: C^op -> FinSet`, this computes
        `Lan_{u^op}(F): D^op -> FinSet` by quotienting matching comma-indexed
        elements under the colimit identifications.
        """
        if functor.target is not self.category:
            raise ValueError("Left Kan extension must be called on the functor target topos.")
        if presheaf.category is not functor.source:
            raise ValueError("Presheaf must live on the functor source category.")

        raw_by_obj = {}
        class_maps = {}
        sets = {}
        for obj in self.category.objects:
            raw_elements = tuple(
                (source_obj, arrow, element)
                for source_obj, arrow in self._left_kan_index(functor, obj)
                for element in presheaf.sets[source_obj]
            )
            generators = []
            for source_obj, arrow, restricted_obj, restricted_arrow, morphism in self._left_kan_index_morphisms(functor, obj):
                for element in presheaf.sets[source_obj]:
                    generators.append(
                        (
                            (source_obj, arrow, element),
                            (restricted_obj, restricted_arrow, presheaf.restrict(morphism, element)),
                        )
                    )
            raw_by_obj[obj] = raw_elements
            class_maps[obj] = _equivalence_class_map(raw_elements, generators)
            sets[obj] = frozenset(class_maps[obj].values())

        restrictions = {}
        for morphism, (src, dst) in self.category.morphisms.items():
            mapping = {}
            for equivalence_class in sets[dst]:
                restricted_classes = {
                    class_maps[src][(source_obj, self.category.compose(arrow, morphism), element)]
                    for source_obj, arrow, element in equivalence_class
                }
                if len(restricted_classes) != 1:
                    raise ValueError("Left Kan extension restriction is not well-defined.")
                mapping[equivalence_class] = next(iter(restricted_classes))
            restrictions[morphism] = mapping

        return Presheaf(category=self.category, sets=sets, restrictions=restrictions)

    def _right_kan_index(self, functor: FiniteFunctor, target_obj):
        return tuple(
            (source_obj, arrow)
            for source_obj in functor.source.objects
            for arrow in self.category.hom(functor.map_object(source_obj), target_obj)
        )

    def _right_kan_index_morphisms(self, functor: FiniteFunctor, target_obj):
        index = self._right_kan_index(functor, target_obj)
        for source_obj, arrow in index:
            for restricted_obj, restricted_arrow in index:
                for morphism in functor.source.hom(restricted_obj, source_obj):
                    if self.category.compose(arrow, functor.map_morphism(morphism)) == restricted_arrow:
                        yield source_obj, arrow, restricted_obj, restricted_arrow, morphism

    def right_kan_extension_presheaf(self, functor: FiniteFunctor, presheaf: Presheaf):
        """
        Right adjoint to reindexing, computed as finite right Kan extension.

        Elements are compatible families over the comma category `(u(-) -> d)`.
        """
        if functor.target is not self.category:
            raise ValueError("Right Kan extension must be called on the functor target topos.")
        if presheaf.category is not functor.source:
            raise ValueError("Presheaf must live on the functor source category.")

        indexes = {obj: self._right_kan_index(functor, obj) for obj in self.category.objects}
        sets = {}
        for obj, index in indexes.items():
            choices = [tuple(presheaf.sets[source_obj]) for source_obj, _arrow in index]
            values = []
            for picked in product(*choices) if choices else ((),):
                assignment = dict(zip(index, picked))
                is_compatible = True
                for source_obj, arrow, restricted_obj, restricted_arrow, morphism in self._right_kan_index_morphisms(functor, obj):
                    if presheaf.restrict(morphism, assignment[(source_obj, arrow)]) != assignment[(restricted_obj, restricted_arrow)]:
                        is_compatible = False
                        break
                if is_compatible:
                    values.append(frozenset(assignment.items()))
            sets[obj] = frozenset(values)

        restrictions = {}
        for morphism, (src, dst) in self.category.morphisms.items():
            mapping = {}
            for assignment_items in sets[dst]:
                assignment = dict(assignment_items)
                restricted = frozenset(
                    ((source_obj, arrow), assignment[(source_obj, self.category.compose(morphism, arrow))])
                    for source_obj, arrow in indexes[src]
                )
                mapping[assignment_items] = restricted
            restrictions[morphism] = mapping

        return Presheaf(category=self.category, sets=sets, restrictions=restrictions)

    def identity_transformation(self, presheaf: Presheaf):
        """Identity natural transformation on a presheaf."""
        self._require_presheaf(presheaf)
        return self.natural_transformation(
            source=presheaf,
            target=presheaf,
            components={obj: {element: element for element in presheaf.sets[obj]} for obj in self.category.objects},
        )

    def compose_transformations(self, after: NaturalTransformation, before: NaturalTransformation):
        """Compose natural transformations as `after o before`."""
        if before.target is not after.source:
            raise ValueError("Natural transformation composition requires matching middle object.")
        self._require_presheaf(before.source)
        self._require_presheaf(before.target)
        self._require_presheaf(after.target)
        return self.natural_transformation(
            source=before.source,
            target=after.target,
            components={
                obj: {
                    element: after.apply(obj, before.apply(obj, element))
                    for element in before.source.sets[obj]
                }
                for obj in self.category.objects
            },
        )

    def is_monomorphism(self, transformation: NaturalTransformation):
        """Return True when a map of presheaves is objectwise injective."""
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        for obj in self.category.objects:
            image = [transformation.apply(obj, element) for element in transformation.source.sets[obj]]
            if len(set(image)) != len(image):
                return False
        return True

    def is_epimorphism(self, transformation: NaturalTransformation):
        """Return True when a map of presheaves is objectwise surjective."""
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        for obj in self.category.objects:
            image = {transformation.apply(obj, element) for element in transformation.source.sets[obj]}
            if image != set(transformation.target.sets[obj]):
                return False
        return True

    def terminal_presheaf(self):
        """Terminal object of Set^(C^op), computed pointwise as a singleton."""
        singleton = "*"
        return Presheaf(
            category=self.category,
            sets={obj: {singleton} for obj in self.category.objects},
            restrictions={morphism: {singleton: singleton} for morphism in self.category.morphisms},
        )

    def initial_presheaf(self):
        """Initial object of Set^(C^op), computed pointwise as the empty set."""
        return Presheaf(
            category=self.category,
            sets={obj: set() for obj in self.category.objects},
            restrictions={morphism: {} for morphism in self.category.morphisms},
        )

    def product_presheaf(self, left: Presheaf, right: Presheaf):
        """
        Binary product in the presheaf topos, computed pointwise.

        Returns `(product, pi_left, pi_right)`.
        """
        self._require_presheaf(left)
        self._require_presheaf(right)

        sets = {
            obj: frozenset(product(left.sets[obj], right.sets[obj]))
            for obj in self.category.objects
        }
        restrictions = {}
        for morphism, (_src, dst) in self.category.morphisms.items():
            restrictions[morphism] = {
                pair: (left.restrict(morphism, pair[0]), right.restrict(morphism, pair[1]))
                for pair in sets[dst]
            }

        product_object = Presheaf(self.category, sets, restrictions)
        pi_left = self.natural_transformation(
            source=product_object,
            target=left,
            components={obj: {pair: pair[0] for pair in product_object.sets[obj]} for obj in self.category.objects},
        )
        pi_right = self.natural_transformation(
            source=product_object,
            target=right,
            components={obj: {pair: pair[1] for pair in product_object.sets[obj]} for obj in self.category.objects},
        )
        return product_object, pi_left, pi_right

    def coproduct_presheaf(self, left: Presheaf, right: Presheaf):
        """
        Binary coproduct in the presheaf topos, computed pointwise.

        Returns `(coproduct, in_left, in_right)`.
        """
        self._require_presheaf(left)
        self._require_presheaf(right)

        sets = {}
        for obj in self.category.objects:
            left_values = {("left", value) for value in left.sets[obj]}
            right_values = {("right", value) for value in right.sets[obj]}
            sets[obj] = frozenset(left_values | right_values)

        restrictions = {}
        for morphism, (_src, dst) in self.category.morphisms.items():
            mapping = {}
            for tag, value in sets[dst]:
                if tag == "left":
                    mapping[(tag, value)] = (tag, left.restrict(morphism, value))
                else:
                    mapping[(tag, value)] = (tag, right.restrict(morphism, value))
            restrictions[morphism] = mapping

        coproduct_object = Presheaf(self.category, sets, restrictions)
        in_left = self.natural_transformation(
            source=left,
            target=coproduct_object,
            components={obj: {value: ("left", value) for value in left.sets[obj]} for obj in self.category.objects},
        )
        in_right = self.natural_transformation(
            source=right,
            target=coproduct_object,
            components={obj: {value: ("right", value) for value in right.sets[obj]} for obj in self.category.objects},
        )
        return coproduct_object, in_left, in_right

    def pullback(self, first: NaturalTransformation, second: NaturalTransformation):
        """Pullback/fiber product of two maps with a common codomain."""
        if first.target is not second.target:
            raise ValueError("Pullback requires natural transformations with the same target.")
        self._require_presheaf(first.source)
        self._require_presheaf(second.source)
        self._require_presheaf(first.target)

        sets = {}
        for obj in self.category.objects:
            sets[obj] = frozenset(
                (left_value, right_value)
                for left_value in first.source.sets[obj]
                for right_value in second.source.sets[obj]
                if first.apply(obj, left_value) == second.apply(obj, right_value)
            )

        restrictions = {}
        for morphism, (_src, dst) in self.category.morphisms.items():
            restrictions[morphism] = {
                pair: (first.source.restrict(morphism, pair[0]), second.source.restrict(morphism, pair[1]))
                for pair in sets[dst]
            }

        pullback_object = Presheaf(self.category, sets, restrictions)
        pi_first = self.natural_transformation(
            source=pullback_object,
            target=first.source,
            components={obj: {pair: pair[0] for pair in pullback_object.sets[obj]} for obj in self.category.objects},
        )
        pi_second = self.natural_transformation(
            source=pullback_object,
            target=second.source,
            components={obj: {pair: pair[1] for pair in pullback_object.sets[obj]} for obj in self.category.objects},
        )
        return pullback_object, pi_first, pi_second

    def equalizer(self, first: NaturalTransformation, second: NaturalTransformation):
        """Equalizer subpresheaf of two parallel natural transformations."""
        if first.source is not second.source or first.target is not second.target:
            raise ValueError("Equalizer requires parallel natural transformations.")
        self._require_presheaf(first.source)
        self._require_presheaf(first.target)

        subsets = {}
        for obj in self.category.objects:
            subsets[obj] = {
                element
                for element in first.source.sets[obj]
                if first.apply(obj, element) == second.apply(obj, element)
            }
        return Subpresheaf(parent=first.source, subsets=subsets)

    def subpresheaf_object(self, subobject: Subpresheaf):
        """View a subobject as its own presheaf with inherited restrictions."""
        self._require_presheaf(subobject.parent)
        restrictions = {}
        for morphism, (_src, dst) in self.category.morphisms.items():
            restrictions[morphism] = {
                element: subobject.parent.restrict(morphism, element)
                for element in subobject.subsets[dst]
            }
        return Presheaf(category=self.category, sets=subobject.subsets, restrictions=restrictions)

    def subpresheaf_inclusion(self, subobject: Subpresheaf):
        """Canonical monomorphism from a subpresheaf object into its parent."""
        subobject_presheaf = self.subpresheaf_object(subobject)
        inclusion = self.natural_transformation(
            source=subobject_presheaf,
            target=subobject.parent,
            components={
                obj: {element: element for element in subobject_presheaf.sets[obj]}
                for obj in self.category.objects
            },
        )
        return subobject_presheaf, inclusion

    def inverse_image(self, transformation: NaturalTransformation, subobject: Subpresheaf):
        """Pull back a subobject along a natural transformation."""
        if transformation.target is not subobject.parent:
            raise ValueError("Inverse image requires a subobject of the transformation target.")
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        return Subpresheaf(
            parent=transformation.source,
            subsets={
                obj: {
                    element
                    for element in transformation.source.sets[obj]
                    if transformation.apply(obj, element) in subobject.subsets[obj]
                }
                for obj in self.category.objects
            },
        )

    def forcing_sieve(self, subobject: Subpresheaf, obj, element):
        """
        Kripke-Joyal truth value of `element in subobject` as a sieve on obj.

        The returned sieve contains arrows along which the element restricts
        into the subobject.
        """
        self._require_presheaf(subobject.parent)
        if element not in subobject.parent.sets[obj]:
            raise ValueError(f"{element!r} is not an element over {obj!r}.")
        return self._characteristic_sieve(subobject, obj, element)

    def forces(self, subobject: Subpresheaf, obj, element):
        """Return True exactly when the forcing sieve is maximal."""
        return self.forcing_sieve(subobject, obj, element) == self.maximal_sieve(obj)

    def truth_value(self, subobject: Subpresheaf):
        """Truth value of a closed proposition, represented as a subobject of 1."""
        self._require_presheaf(subobject.parent)
        if any(len(subobject.parent.sets[obj]) != 1 for obj in self.category.objects):
            raise ValueError("Truth values are defined for subobjects of a terminal presheaf.")
        return self.characteristic_map(subobject)

    def diagonal_transformation(self, presheaf: Presheaf):
        """Diagonal morphism `F -> F x F`, the internal equality classifying map."""
        self._require_presheaf(presheaf)
        product_object, _pi_left, _pi_right = self.product_presheaf(presheaf, presheaf)
        return self.natural_transformation(
            source=presheaf,
            target=product_object,
            components={
                obj: {element: (element, element) for element in presheaf.sets[obj]}
                for obj in self.category.objects
            },
        )

    def equality_subobject(self, presheaf: Presheaf):
        """Internal equality predicate `Eq_F <= F x F`, the image of the diagonal."""
        diagonal = self.diagonal_transformation(presheaf)
        return self.image(diagonal)

    def equality_truth(self, presheaf: Presheaf, obj, left, right):
        """Kripke-Joyal truth value of `left = right` over an object."""
        equality = self.equality_subobject(presheaf)
        return self.forcing_sieve(equality, obj, (left, right))

    def exists_along(self, transformation: NaturalTransformation, subobject: Subpresheaf):
        """Existential quantification along a map, as image of the restricted map."""
        if subobject.parent is not transformation.source:
            raise ValueError("Existential quantification requires a subobject of the map source.")
        subobject_presheaf, inclusion = self.subpresheaf_inclusion(subobject)
        restricted_map = self.compose_transformations(transformation, inclusion)
        return self.image(restricted_map)

    def forall_along(self, transformation: NaturalTransformation, subobject: Subpresheaf):
        """
        Universal quantification along a map, computed as the right adjoint to pullback.

        It is the greatest subobject T of the codomain such that f*T <= S.
        """
        if subobject.parent is not transformation.source:
            raise ValueError("Universal quantification requires a subobject of the map source.")
        self._require_presheaf(transformation.target)
        result = self.subobject_bottom(transformation.target)
        for candidate in self.subobjects(transformation.target):
            if self.subobject_leq(self.inverse_image(transformation, candidate), subobject):
                result = self.subobject_join(result, candidate)
        return result

    def validate_quantifier_adjunctions(self, transformation: NaturalTransformation):
        """Check finite adjunction laws `exists_f -| f* -| forall_f`."""
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        source_subobjects = self.subobjects(transformation.source)
        target_subobjects = self.subobjects(transformation.target)
        for source_subobject in source_subobjects:
            exists_source = self.exists_along(transformation, source_subobject)
            forall_source = self.forall_along(transformation, source_subobject)
            for target_subobject in target_subobjects:
                pullback_target = self.inverse_image(transformation, target_subobject)
                if self.subobject_leq(exists_source, target_subobject) != self.subobject_leq(source_subobject, pullback_target):
                    return False
                if self.subobject_leq(pullback_target, source_subobject) != self.subobject_leq(target_subobject, forall_source):
                    return False
        return True

    def validate_frobenius_reciprocity(self, transformation: NaturalTransformation):
        """Check `exists_f(S meet f*T) = exists_f(S) meet T` for all finite subobjects."""
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        for source_subobject in self.subobjects(transformation.source):
            exists_source = self.exists_along(transformation, source_subobject)
            for target_subobject in self.subobjects(transformation.target):
                pulled_target = self.inverse_image(transformation, target_subobject)
                left = self.exists_along(
                    transformation,
                    self.subobject_meet(source_subobject, pulled_target),
                )
                right = self.subobject_meet(exists_source, target_subobject)
                if left.subsets != right.subsets:
                    return False
        return True

    def validate_beck_chevalley(self, transformation: NaturalTransformation, base_change: NaturalTransformation):
        """
        Check Beck-Chevalley for a pullback square.

        For maps `f: X -> Y` and `g: Z -> Y`, this validates both
        `g* exists_f = exists_q p*` and `g* forall_f = forall_q p*`, where
        `P = X x_Y Z`, `p: P -> X`, and `q: P -> Z`.
        """
        if transformation.target is not base_change.target:
            raise ValueError("Beck-Chevalley requires maps with a common codomain.")
        pullback_object, pullback_to_source, pullback_to_base = self.pullback(transformation, base_change)
        for source_subobject in self.subobjects(transformation.source):
            pulled_source = self.inverse_image(pullback_to_source, source_subobject)
            left_exists = self.inverse_image(base_change, self.exists_along(transformation, source_subobject))
            right_exists = self.exists_along(pullback_to_base, pulled_source)
            if left_exists.subsets != right_exists.subsets:
                return False

            left_forall = self.inverse_image(base_change, self.forall_along(transformation, source_subobject))
            right_forall = self.forall_along(pullback_to_base, pulled_source)
            if left_forall.subsets != right_forall.subsets:
                return False

        return pullback_object.validate_functor_laws()

    def exponential_presheaf(self, exponent: Presheaf, base: Presheaf):
        """
        Exponential object `base^exponent`.

        For each object c, `(base^exponent)(c)` is
        `Nat(y(c) x exponent, base)`.
        """
        self._require_presheaf(exponent)
        self._require_presheaf(base)

        sources = {}
        sets = {}
        for obj in self.category.objects:
            representable = representable_presheaf(self.category, obj)
            source, _pi_y, _pi_exponent = self.product_presheaf(representable, exponent)
            sources[obj] = source
            sets[obj] = frozenset(
                FrozenNaturalTransformation.from_transformation(alpha)
                for alpha in natural_transformations(source, base)
            )

        restrictions = {}
        for morphism, (src, dst) in self.category.morphisms.items():
            source_src = sources[src]
            mapping = {}
            for alpha in sets[dst]:
                components = {}
                for stage in self.category.objects:
                    stage_map = {}
                    for arrow_to_src, exponent_value in source_src.sets[stage]:
                        arrow_to_dst = self.category.compose(morphism, arrow_to_src)
                        stage_map[(arrow_to_src, exponent_value)] = alpha.apply(stage, (arrow_to_dst, exponent_value))
                    components[stage] = stage_map
                restricted = NaturalTransformation(source=source_src, target=base, components=components)
                restricted.validate_naturality()
                mapping[alpha] = FrozenNaturalTransformation.from_transformation(restricted)
            restrictions[morphism] = mapping

        exponential = Presheaf(self.category, sets, restrictions)
        exponential.exponent = exponent
        exponential.base = base
        exponential.exponential_sources = sources
        return exponential

    def evaluation_map(self, exponent: Presheaf, base: Presheaf, power: Presheaf | None = None):
        """
        Evaluation morphism `base^exponent x exponent -> base`.

        The exponential is represented by
        `(base^exponent)(c) = Nat(y(c) x exponent, base)`, so evaluation at
        stage c simply applies a natural transformation to `(id_c, x)`.
        """
        self._require_presheaf(exponent)
        self._require_presheaf(base)
        if power is None:
            power = self.exponential_presheaf(exponent, base)
        else:
            self._require_presheaf(power)
            if getattr(power, "exponent", None) is not exponent or getattr(power, "base", None) is not base:
                raise ValueError("Evaluation requires the exponential object base^exponent.")

        evaluation_product, _pi_power, _pi_exponent = self.product_presheaf(power, exponent)
        components = {}
        for obj in self.category.objects:
            identity = self.category.identities[obj]
            components[obj] = {
                (alpha, exponent_value): alpha.apply(obj, (identity, exponent_value))
                for alpha, exponent_value in evaluation_product.sets[obj]
            }

        evaluation = self.natural_transformation(source=evaluation_product, target=base, components=components)
        return power, evaluation_product, evaluation

    def transpose(
        self,
        product_map: NaturalTransformation,
        domain: Presheaf,
        exponent: Presheaf,
        base: Presheaf,
        power: Presheaf | None = None,
    ):
        """
        Curry a map `domain x exponent -> base` into `domain -> base^exponent`.

        At c, an element h in domain(c) names a natural transformation
        `y(c) x exponent -> base`; at stage d and arrow u: d -> c, it sends
        `(u, x)` to `product_map_d(domain(u)(h), x)`.
        """
        self._require_presheaf(product_map.source)
        self._require_presheaf(product_map.target)
        self._require_presheaf(domain)
        self._require_presheaf(exponent)
        self._require_presheaf(base)
        if product_map.target is not base:
            raise ValueError("Transpose target must be the requested exponential base.")
        if power is None:
            power = self.exponential_presheaf(exponent, base)
        else:
            self._require_presheaf(power)
            if getattr(power, "exponent", None) is not exponent or getattr(power, "base", None) is not base:
                raise ValueError("Transpose requires the exponential object base^exponent.")

        components = {}
        for obj in self.category.objects:
            exponential_source = power.exponential_sources[obj]
            obj_component = {}
            for domain_element in domain.sets[obj]:
                alpha_components = {}
                for stage in self.category.objects:
                    stage_map = {}
                    for arrow_to_obj, exponent_value in exponential_source.sets[stage]:
                        restricted_domain_element = domain.restrict(arrow_to_obj, domain_element)
                        stage_map[(arrow_to_obj, exponent_value)] = product_map.apply(
                            stage,
                            (restricted_domain_element, exponent_value),
                        )
                    alpha_components[stage] = stage_map

                alpha = NaturalTransformation(source=exponential_source, target=base, components=alpha_components)
                alpha.validate_naturality()
                obj_component[domain_element] = FrozenNaturalTransformation.from_transformation(alpha)
            components[obj] = obj_component

        return power, self.natural_transformation(source=domain, target=power, components=components)

    def _require_same_parent(self, left: Subpresheaf, right: Subpresheaf):
        if left.parent is not right.parent:
            raise ValueError("Subobject operations require the same parent presheaf object.")
        self._require_presheaf(left.parent)

    def subobject_bottom(self, parent: Presheaf):
        """Bottom subobject of a presheaf."""
        self._require_presheaf(parent)
        return Subpresheaf(parent=parent, subsets={obj: set() for obj in self.category.objects})

    def subobject_top(self, parent: Presheaf):
        """Top subobject of a presheaf."""
        self._require_presheaf(parent)
        return Subpresheaf(parent=parent, subsets={obj: set(parent.sets[obj]) for obj in self.category.objects})

    def subobject_meet(self, left: Subpresheaf, right: Subpresheaf):
        """Meet/intersection in the subobject Heyting algebra."""
        self._require_same_parent(left, right)
        return Subpresheaf(
            parent=left.parent,
            subsets={obj: left.subsets[obj] & right.subsets[obj] for obj in self.category.objects},
        )

    def subobject_join(self, left: Subpresheaf, right: Subpresheaf):
        """Join/union in the subobject Heyting algebra."""
        self._require_same_parent(left, right)
        return Subpresheaf(
            parent=left.parent,
            subsets={obj: left.subsets[obj] | right.subsets[obj] for obj in self.category.objects},
        )

    def subobject_implication(self, antecedent: Subpresheaf, consequent: Subpresheaf):
        """
        Heyting implication between subobjects of the same presheaf.

        An element x in F(c) is included when every pullback of x along every
        arrow d -> c that lands in the antecedent also lands in the consequent.
        """
        self._require_same_parent(antecedent, consequent)
        parent = antecedent.parent
        subsets = {}

        for obj in self.category.objects:
            valid_elements = set()
            for element in parent.sets[obj]:
                is_valid = True
                for arrow in self.category.arrows_to(obj):
                    restricted = parent.restrict(arrow, element)
                    source = self.category.source(arrow)
                    if restricted in antecedent.subsets[source] and restricted not in consequent.subsets[source]:
                        is_valid = False
                        break
                if is_valid:
                    valid_elements.add(element)
            subsets[obj] = valid_elements

        return Subpresheaf(parent=parent, subsets=subsets)

    def subobject_negation(self, subobject: Subpresheaf):
        """Heyting negation of a subobject: A => bottom."""
        return self.subobject_implication(subobject, self.subobject_bottom(subobject.parent))

    def subobject_leq(self, left: Subpresheaf, right: Subpresheaf):
        """Order relation in a subobject lattice."""
        self._require_same_parent(left, right)
        return all(left.subsets[obj].issubset(right.subsets[obj]) for obj in self.category.objects)

    def subobjects(self, parent: Presheaf):
        """Enumerate all subobjects of a finite presheaf."""
        self._require_presheaf(parent)
        subset_choices = [
            tuple(_powerset(parent.sets[obj]))
            for obj in self.category.objects
        ]
        subobjects = []
        for choices in product(*subset_choices):
            try:
                subobjects.append(Subpresheaf(parent=parent, subsets=dict(zip(self.category.objects, choices))))
            except ValueError:
                continue
        return tuple(subobjects)

    def j_closed_subobjects(self, parent: Presheaf, topology):
        """Enumerate subobjects fixed by J-closure."""
        self._require_presheaf(parent)
        self._require_topology(topology)
        return tuple(
            subobject
            for subobject in self.subobjects(parent)
            if self.is_j_closed_subobject(subobject, topology)
        )

    def subobject_j_meet(self, left: Subpresheaf, right: Subpresheaf, topology):
        """Meet of J-closed subobjects."""
        self._require_topology(topology)
        return self.subobject_meet(
            self.subobject_closure(left, topology),
            self.subobject_closure(right, topology),
        )

    def subobject_j_join(self, left: Subpresheaf, right: Subpresheaf, topology):
        """Join of J-closed subobjects, closed after ordinary union."""
        self._require_topology(topology)
        return self.subobject_closure(self.subobject_join(left, right), topology)

    def subobject_j_implication(self, antecedent: Subpresheaf, consequent: Subpresheaf, topology):
        """Implication in the J-closed subobject Heyting algebra."""
        self._require_topology(topology)
        return self.subobject_closure(self.subobject_implication(antecedent, consequent), topology)

    def subobject_j_negation(self, subobject: Subpresheaf, topology):
        """Negation in the J-closed subobject Heyting algebra."""
        return self.subobject_j_implication(subobject, self.subobject_bottom(subobject.parent), topology)

    def validate_j_subobject_heyting_laws(self, parent: Presheaf, topology):
        """Check finite Heyting adjunction laws for J-closed subobjects."""
        self._require_presheaf(parent)
        self._require_topology(topology)
        closed = self.j_closed_subobjects(parent, topology)
        for antecedent in closed:
            for consequent in closed:
                implication = self.subobject_j_implication(antecedent, consequent, topology)
                if not self.is_j_closed_subobject(implication, topology):
                    return False
                for probe in closed:
                    meet_below = self.subobject_leq(
                        self.subobject_j_meet(probe, antecedent, topology),
                        consequent,
                    )
                    adjoint_below = self.subobject_leq(probe, implication)
                    if meet_below != adjoint_below:
                        return False
        return True

    def power_object(self, presheaf: Presheaf):
        """Power object P(F) = Omega^F."""
        self._require_presheaf(presheaf)
        omega = self.omega()
        power = self.exponential_presheaf(presheaf, omega)
        power.power_parent = presheaf
        power.power_omega = omega
        return power

    def _characteristic_sieve(self, subobject: Subpresheaf, obj, element):
        parent = subobject.parent
        return frozenset(
            arrow
            for arrow in self.category.arrows_to(obj)
            if parent.restrict(arrow, element) in subobject.subsets[self.category.source(arrow)]
        )

    def name_subobject(self, subobject: Subpresheaf):
        """
        Return the classifying name 1 -> P(F) of a subobject S <= F.

        This is the transpose of the characteristic map F -> Omega.
        """
        self._require_presheaf(subobject.parent)
        terminal = self.terminal_presheaf()
        power = self.power_object(subobject.parent)
        omega = power.power_omega

        components = {}
        for obj in self.category.objects:
            source = power.exponential_sources[obj]
            alpha_components = {}
            for stage in self.category.objects:
                alpha_components[stage] = {
                    (arrow_to_obj, element): self._characteristic_sieve(subobject, stage, element)
                    for arrow_to_obj, element in source.sets[stage]
                }
            alpha = NaturalTransformation(source=source, target=omega, components=alpha_components)
            alpha.validate_naturality()
            named_element = FrozenNaturalTransformation.from_transformation(alpha)
            components[obj] = {next(iter(terminal.sets[obj])): named_element}

        return self.natural_transformation(source=terminal, target=power, components=components)

    def membership_relation(self, presheaf: Presheaf, power: Presheaf | None = None):
        """
        Return `(P(F), F x P(F), in_relation)` for the power object.

        A pair `(x, phi)` belongs to membership at c when
        `phi_c(id_c, x)` is the maximal sieve on c.
        """
        self._require_presheaf(presheaf)
        if power is None:
            power = self.power_object(presheaf)

        product_object, _pi_f, _pi_power = self.product_presheaf(presheaf, power)
        subsets = {}
        for obj in self.category.objects:
            identity = self.category.identities[obj]
            true_at_obj = self.maximal_sieve(obj)
            subsets[obj] = {
                (element, predicate)
                for element, predicate in product_object.sets[obj]
                if predicate.apply(obj, (identity, element)) == true_at_obj
            }

        return power, product_object, Subpresheaf(parent=product_object, subsets=subsets)

    def extension_of_name(self, parent: Presheaf, name: NaturalTransformation):
        """Recover a subobject of F from a global element 1 -> P(F)."""
        self._require_presheaf(parent)
        subsets = {}
        for obj in self.category.objects:
            identity = self.category.identities[obj]
            true_at_obj = self.maximal_sieve(obj)
            terminal_element = next(iter(name.source.sets[obj]))
            predicate = name.apply(obj, terminal_element)
            subsets[obj] = {
                element
                for element in parent.sets[obj]
                if predicate.apply(obj, (identity, element)) == true_at_obj
            }
        return Subpresheaf(parent=parent, subsets=subsets)

    def image(self, transformation: NaturalTransformation):
        """Image subpresheaf of a natural transformation."""
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        return Subpresheaf(
            parent=transformation.target,
            subsets={
                obj: {transformation.apply(obj, element) for element in transformation.source.sets[obj]}
                for obj in self.category.objects
            },
        )

    def image_factorization(self, transformation: NaturalTransformation):
        """Factor a map as an objectwise epimorphism followed by a monomorphism."""
        image = self.image(transformation)
        image_object, inclusion = self.subpresheaf_inclusion(image)
        quotient_to_image = self.natural_transformation(
            source=transformation.source,
            target=image_object,
            components={
                obj: {
                    element: transformation.apply(obj, element)
                    for element in transformation.source.sets[obj]
                }
                for obj in self.category.objects
            },
        )
        return image_object, quotient_to_image, inclusion

    def kernel_pair(self, transformation: NaturalTransformation):
        """Kernel pair of a natural transformation, implemented as its pullback along itself."""
        return self.pullback(transformation, transformation)

    def coequalizer(self, first: NaturalTransformation, second: NaturalTransformation):
        """
        Coequalizer of parallel natural transformations, computed pointwise.

        Quotient elements are represented as frozensets of original elements.
        """
        if first.source is not second.source or first.target is not second.target:
            raise ValueError("Coequalizer requires parallel natural transformations.")
        self._require_presheaf(first.source)
        self._require_presheaf(first.target)

        target = first.target
        class_maps = {}
        sets = {}
        for obj in self.category.objects:
            generators = [
                (first.apply(obj, element), second.apply(obj, element))
                for element in first.source.sets[obj]
            ]
            class_maps[obj] = _equivalence_class_map(target.sets[obj], generators)
            sets[obj] = frozenset(class_maps[obj].values())

        restrictions = {}
        for morphism, (src, dst) in self.category.morphisms.items():
            mapping = {}
            for equivalence_class in sets[dst]:
                restricted_classes = {
                    class_maps[src][target.restrict(morphism, representative)]
                    for representative in equivalence_class
                }
                if len(restricted_classes) != 1:
                    raise ValueError("Quotient restriction is not well-defined.")
                mapping[equivalence_class] = next(iter(restricted_classes))
            restrictions[morphism] = mapping

        quotient = Presheaf(self.category, sets, restrictions)
        projection = self.natural_transformation(
            source=target,
            target=quotient,
            components={
                obj: {element: class_maps[obj][element] for element in target.sets[obj]}
                for obj in self.category.objects
            },
        )
        return quotient, projection

    def _is_sieve(self, obj, arrows):
        arrows = frozenset(arrows)
        if not arrows.issubset(self.category.arrows_to(obj)):
            return False

        for arrow in arrows:
            for incoming in self.category.arrows_to(self.category.source(arrow)):
                if self.category.compose(arrow, incoming) not in arrows:
                    return False

        return True

    def sieves_on(self, obj):
        return frozenset(subset for subset in _powerset(self.category.arrows_to(obj)) if self._is_sieve(obj, subset))

    def maximal_sieve(self, obj):
        return frozenset(self.category.arrows_to(obj))

    def pullback_sieve(self, morphism, sieve):
        sieve = frozenset(sieve)
        source = self.category.source(morphism)
        return frozenset(
            arrow for arrow in self.category.arrows_to(source) if self.category.compose(morphism, arrow) in sieve
        )

    def matching_families(self, presheaf: Presheaf, obj, sieve):
        """
        Enumerate matching families for a presheaf over a sieve on obj.

        A family assigns x_f in F(domain f) to each f: domain f -> obj in the
        sieve, with F(g)(x_f) = x_(f o g) whenever f o g is still in the sieve.
        """
        self._require_presheaf(presheaf)
        sieve = frozenset(sieve)
        if not self._is_sieve(obj, sieve):
            raise ValueError("Matching families require a sieve on the requested object.")

        arrows = tuple(sieve)
        choices = [tuple(presheaf.sets[self.category.source(arrow)]) for arrow in arrows]
        families = []
        for values in product(*choices) if choices else ((),):
            family = dict(zip(arrows, values))
            is_matching = True
            for arrow in arrows:
                source = self.category.source(arrow)
                for incoming in self.category.arrows_to(source):
                    composite = self.category.compose(arrow, incoming)
                    if composite in family and presheaf.restrict(incoming, family[arrow]) != family[composite]:
                        is_matching = False
                        break
                if not is_matching:
                    break
            if is_matching:
                families.append(family)
        return tuple(families)

    def amalgamations(self, presheaf: Presheaf, obj, matching_family):
        """Return all global sections over obj that glue a matching family."""
        self._require_presheaf(presheaf)
        return tuple(
            element
            for element in presheaf.sets[obj]
            if all(presheaf.restrict(arrow, element) == value for arrow, value in matching_family.items())
        )

    def is_separated(self, presheaf: Presheaf, topology):
        """Check the separated presheaf condition for every covering sieve."""
        self._require_presheaf(presheaf)
        if topology.category is not self.category:
            raise ValueError("Topology must be defined on this topos base category.")

        for obj in self.category.objects:
            for cover in topology.covering_sieves[obj]:
                for family in self.matching_families(presheaf, obj, cover):
                    if len(self.amalgamations(presheaf, obj, family)) > 1:
                        return False
        return True

    def is_sheaf(self, presheaf: Presheaf, topology):
        """Check the finite sheaf condition: every matching family glues uniquely."""
        self._require_presheaf(presheaf)
        if topology.category is not self.category:
            raise ValueError("Topology must be defined on this topos base category.")

        for obj in self.category.objects:
            for cover in topology.covering_sieves[obj]:
                for family in self.matching_families(presheaf, obj, cover):
                    if len(self.amalgamations(presheaf, obj, family)) != 1:
                        return False
        return True

    def _matching_datum(self, cover, family):
        return (frozenset(cover), frozenset(family.items()))

    def _matching_datum_cover(self, datum):
        return datum[0]

    def _matching_datum_family(self, datum):
        return dict(datum[1])

    def _locally_equal_matching_datums(self, obj, left, right, topology):
        left_cover = self._matching_datum_cover(left)
        right_cover = self._matching_datum_cover(right)
        left_family = self._matching_datum_family(left)
        right_family = self._matching_datum_family(right)
        common_cover = left_cover & right_cover

        for refinement in topology.covering_sieves[obj]:
            if refinement.issubset(common_cover) and all(
                left_family[arrow] == right_family[arrow] for arrow in refinement
            ):
                return True
        return False

    def _restrict_matching_datum(self, presheaf: Presheaf, morphism, datum):
        cover = self._matching_datum_cover(datum)
        family = self._matching_datum_family(datum)
        pulled_cover = self.pullback_sieve(morphism, cover)
        pulled_family = {
            arrow: family[self.category.compose(morphism, arrow)]
            for arrow in pulled_cover
        }
        return self._matching_datum(pulled_cover, pulled_family)

    def plus_construction(self, presheaf: Presheaf, topology):
        """
        Finite plus construction for a presheaf on a site.

        `F+(c)` is the quotient of all matching families on covering sieves of c
        by local equality on a covering refinement. Applying plus twice gives
        the associated sheaf for the finite sites represented here.
        """
        self._require_presheaf(presheaf)
        if topology.category is not self.category:
            raise ValueError("Topology must be defined on this topos base category.")

        datums_by_obj = {}
        class_maps = {}
        sets = {}
        for obj in self.category.objects:
            datums = []
            for cover in topology.covering_sieves[obj]:
                for family in self.matching_families(presheaf, obj, cover):
                    datums.append(self._matching_datum(cover, family))
            datums_by_obj[obj] = tuple(dict.fromkeys(datums))

            generators = []
            for index, left in enumerate(datums_by_obj[obj]):
                for right in datums_by_obj[obj][index + 1:]:
                    if self._locally_equal_matching_datums(obj, left, right, topology):
                        generators.append((left, right))

            class_maps[obj] = _equivalence_class_map(datums_by_obj[obj], generators)
            sets[obj] = frozenset(class_maps[obj].values())

        restrictions = {}
        for morphism, (src, dst) in self.category.morphisms.items():
            mapping = {}
            for equivalence_class in sets[dst]:
                restricted_classes = {
                    class_maps[src][self._restrict_matching_datum(presheaf, morphism, datum)]
                    for datum in equivalence_class
                }
                if len(restricted_classes) != 1:
                    raise ValueError("Plus-construction restriction is not well-defined.")
                mapping[equivalence_class] = next(iter(restricted_classes))
            restrictions[morphism] = mapping

        plus = Presheaf(category=self.category, sets=sets, restrictions=restrictions)
        unit_components = {}
        for obj in self.category.objects:
            maximal = self.maximal_sieve(obj)
            unit_components[obj] = {}
            for element in presheaf.sets[obj]:
                family = {arrow: presheaf.restrict(arrow, element) for arrow in maximal}
                unit_components[obj][element] = class_maps[obj][self._matching_datum(maximal, family)]

        unit = self.natural_transformation(source=presheaf, target=plus, components=unit_components)
        plus.plus_source = presheaf
        plus.plus_topology = topology
        return plus, unit

    def extend_to_plus(
        self,
        transformation: NaturalTransformation,
        topology,
        plus: Presheaf | None = None,
    ):
        """
        Extend `F -> G` to `F+ -> G` when G is a sheaf.

        Each plus element is a matching family on a covering sieve; applying the
        transformation gives a matching family in G, which has a unique
        amalgamation because G is a sheaf.
        """
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        if topology.category is not self.category:
            raise ValueError("Topology must be defined on this topos base category.")
        if not self.is_sheaf(transformation.target, topology):
            raise ValueError("Maps can be extended along plus only into sheaves.")
        if plus is None:
            plus, _unit = self.plus_construction(transformation.source, topology)
        else:
            self._require_presheaf(plus)

        components = {}
        for obj in self.category.objects:
            obj_component = {}
            for equivalence_class in plus.sets[obj]:
                datum = next(iter(equivalence_class))
                family = self._matching_datum_family(datum)
                target_family = {
                    arrow: transformation.apply(self.category.source(arrow), value)
                    for arrow, value in family.items()
                }
                amalgamations = self.amalgamations(transformation.target, obj, target_family)
                if len(amalgamations) != 1:
                    raise ValueError("Target sheaf failed to provide a unique amalgamation.")
                obj_component[equivalence_class] = amalgamations[0]
            components[obj] = obj_component

        return self.natural_transformation(source=plus, target=transformation.target, components=components)

    def sheafification(self, presheaf: Presheaf, topology):
        """Associated sheaf via the finite plus-plus construction."""
        plus, unit = self.plus_construction(presheaf, topology)
        sheaf, plus_unit = self.plus_construction(plus, topology)
        return sheaf, self.compose_transformations(plus_unit, unit)

    def sheafification_factorization(self, transformation: NaturalTransformation, topology):
        """
        Factor a map from a presheaf to a sheaf through the associated sheaf.

        Returns `(aF, eta, unique_factor)` where `eta: F -> aF` and
        `unique_factor: aF -> G` satisfy `unique_factor o eta = transformation`.
        """
        self._require_presheaf(transformation.source)
        self._require_presheaf(transformation.target)
        plus, unit = self.plus_construction(transformation.source, topology)
        extended_once = self.extend_to_plus(transformation, topology, plus=plus)
        sheaf, plus_unit = self.plus_construction(plus, topology)
        sheaf_unit = self.compose_transformations(plus_unit, unit)
        factor = self.extend_to_plus(extended_once, topology, plus=sheaf)
        return sheaf, sheaf_unit, factor

    def _require_topology(self, topology):
        if topology.category is not self.category:
            raise ValueError("Topology must be defined on this topos base category.")

    def j_operator_on_sieve(self, topology, obj, sieve):
        """
        Lawvere-Tierney closure of a sieve induced by a Grothendieck topology.

        For a sieve S on c, j(S) contains f: d -> c when the pullback sieve
        f*S is a covering sieve of d.
        """
        self._require_topology(topology)
        sieve = frozenset(sieve)
        if not self._is_sieve(obj, sieve):
            raise ValueError("The local operator is defined on sieves.")

        closed = frozenset(
            arrow
            for arrow in self.category.arrows_to(obj)
            if self.pullback_sieve(arrow, sieve)
            in topology.covering_sieves[self.category.source(arrow)]
        )
        if not self._is_sieve(obj, closed):
            raise ValueError("The local operator did not produce a sieve.")
        return closed

    def lawvere_tierney_operator(self, topology):
        """Natural transformation `j: Omega -> Omega` induced by a site topology."""
        self._require_topology(topology)
        omega = self.omega()
        return self.natural_transformation(
            source=omega,
            target=omega,
            components={
                obj: {
                    sieve: self.j_operator_on_sieve(topology, obj, sieve)
                    for sieve in omega.sets[obj]
                }
                for obj in self.category.objects
            },
        )

    def validate_lawvere_tierney_axioms(self, topology):
        """Check the finite Lawvere-Tierney axioms for the topology-induced j."""
        self._require_topology(topology)
        for obj in self.category.objects:
            true_at_obj = self.maximal_sieve(obj)
            if self.j_operator_on_sieve(topology, obj, true_at_obj) != true_at_obj:
                return False

            for left in self.sieves_on(obj):
                j_left = self.j_operator_on_sieve(topology, obj, left)
                if self.j_operator_on_sieve(topology, obj, j_left) != j_left:
                    return False
                for right in self.sieves_on(obj):
                    j_right = self.j_operator_on_sieve(topology, obj, right)
                    meet = left & right
                    if self.j_operator_on_sieve(topology, obj, meet) != j_left & j_right:
                        return False
        return True

    def covering_sieves_from_lawvere_tierney_operator(self, operator: NaturalTransformation):
        """Recover covering sieves as those S with j(S) = true."""
        self._require_presheaf(operator.source)
        self._require_presheaf(operator.target)
        return {
            obj: frozenset(
                sieve
                for sieve in self.sieves_on(obj)
                if operator.apply(obj, sieve) == self.maximal_sieve(obj)
            )
            for obj in self.category.objects
        }

    def topology_from_lawvere_tierney_operator(self, operator: NaturalTransformation):
        """Recover the Grothendieck topology represented by a local operator."""
        return GrothendieckTopology(
            category=self.category,
            covering_sieves=self.covering_sieves_from_lawvere_tierney_operator(operator),
        )

    def sieve_implication(self, obj, antecedent, consequent):
        """
        Heyting implication between sieves on obj.

        An arrow f: d -> obj belongs when `f*antecedent` is contained in
        `f*consequent`.
        """
        antecedent = frozenset(antecedent)
        consequent = frozenset(consequent)
        if not self._is_sieve(obj, antecedent) or not self._is_sieve(obj, consequent):
            raise ValueError("Sieve implication requires sieves on the same object.")

        return frozenset(
            arrow
            for arrow in self.category.arrows_to(obj)
            if self.pullback_sieve(arrow, antecedent).issubset(self.pullback_sieve(arrow, consequent))
        )

    def sieve_j_meet(self, topology, obj, left, right):
        """Meet in the J-closed-sieve Heyting algebra."""
        left_closed = self.j_operator_on_sieve(topology, obj, left)
        right_closed = self.j_operator_on_sieve(topology, obj, right)
        return left_closed & right_closed

    def sieve_j_join(self, topology, obj, left, right):
        """Join in the J-closed-sieve Heyting algebra."""
        left_closed = self.j_operator_on_sieve(topology, obj, left)
        right_closed = self.j_operator_on_sieve(topology, obj, right)
        return self.j_operator_on_sieve(topology, obj, left_closed | right_closed)

    def sieve_j_implication(self, topology, obj, antecedent, consequent):
        """Implication in the J-closed-sieve Heyting algebra."""
        antecedent_closed = self.j_operator_on_sieve(topology, obj, antecedent)
        consequent_closed = self.j_operator_on_sieve(topology, obj, consequent)
        implication = self.sieve_implication(obj, antecedent_closed, consequent_closed)
        return self.j_operator_on_sieve(topology, obj, implication)

    def sieve_j_negation(self, topology, obj, sieve):
        """Negation in the J-closed-sieve Heyting algebra."""
        return self.sieve_j_implication(topology, obj, sieve, frozenset())

    def validate_omega_j_heyting_laws(self, topology):
        """Check finite Heyting adjunction laws for every Omega_J fiber."""
        self._require_topology(topology)
        omega_j = self.omega_j(topology)
        for obj in self.category.objects:
            for left in omega_j.sets[obj]:
                for right in omega_j.sets[obj]:
                    if self.sieve_j_meet(topology, obj, left, right) != left & right:
                        return False
                    if self.sieve_j_implication(topology, obj, left, right) not in omega_j.sets[obj]:
                        return False
                    for probe in omega_j.sets[obj]:
                        meet_below = self.sieve_j_meet(topology, obj, probe, left).issubset(right)
                        adjoint_below = probe.issubset(self.sieve_j_implication(topology, obj, left, right))
                        if meet_below != adjoint_below:
                            return False
        return True

    def omega_j(self, topology):
        """Subobject classifier of the sheaf topos, represented by J-closed sieves."""
        self._require_topology(topology)
        sets = {
            obj: frozenset(
                sieve
                for sieve in self.sieves_on(obj)
                if self.j_operator_on_sieve(topology, obj, sieve) == sieve
            )
            for obj in self.category.objects
        }
        restrictions = {}
        for morphism, (src, dst) in self.category.morphisms.items():
            restrictions[morphism] = {
                sieve: self.j_operator_on_sieve(topology, src, self.pullback_sieve(morphism, sieve))
                for sieve in sets[dst]
            }
        return Presheaf(self.category, sets, restrictions)

    def subobject_closure(self, subobject: Subpresheaf, topology):
        """J-closure of a subobject using the topology-induced local operator."""
        self._require_presheaf(subobject.parent)
        self._require_topology(topology)
        classifier = self.characteristic_map(subobject)
        subsets = {}
        for obj in self.category.objects:
            true_at_obj = self.maximal_sieve(obj)
            subsets[obj] = {
                element
                for element in subobject.parent.sets[obj]
                if self.j_operator_on_sieve(topology, obj, classifier.apply(obj, element)) == true_at_obj
            }
        return Subpresheaf(parent=subobject.parent, subsets=subsets)

    def truth_map_j(self, topology):
        """Truth morphism `true: 1 -> Omega_J` in the sheaf topos."""
        self._require_topology(topology)
        terminal = self.terminal_presheaf()
        omega_j = self.omega_j(topology)
        return self.natural_transformation(
            source=terminal,
            target=omega_j,
            components={
                obj: {
                    next(iter(terminal.sets[obj])): self.maximal_sieve(obj),
                }
                for obj in self.category.objects
            },
        )

    def characteristic_map_j(self, subobject: Subpresheaf, topology):
        """
        J-modal characteristic map `F -> Omega_J`.

        This is `j o chi_S`, so pulling back truth recovers the J-closure of S.
        If S is already J-closed, it is the sheaf-topos characteristic map.
        """
        self._require_presheaf(subobject.parent)
        self._require_topology(topology)
        classifier = self.characteristic_map(subobject)
        omega_j = self.omega_j(topology)
        return self.natural_transformation(
            source=subobject.parent,
            target=omega_j,
            components={
                obj: {
                    element: self.j_operator_on_sieve(topology, obj, classifier.apply(obj, element))
                    for element in subobject.parent.sets[obj]
                }
                for obj in self.category.objects
            },
        )

    def pullback_truth_j(self, classifier: NaturalTransformation, topology):
        """Recover the subobject classified by a map into `Omega_J`."""
        self._require_topology(topology)
        self._require_presheaf(classifier.source)
        self._require_presheaf(classifier.target)
        subsets = {}
        for obj in self.category.objects:
            true_at_obj = self.maximal_sieve(obj)
            subsets[obj] = {
                element
                for element in classifier.source.sets[obj]
                if classifier.apply(obj, element) == true_at_obj
            }
        return Subpresheaf(parent=classifier.source, subsets=subsets)

    def is_j_closed_subobject(self, subobject: Subpresheaf, topology):
        """Return True when a subobject is fixed by J-closure."""
        return self.subobject_closure(subobject, topology).subsets == subobject.subsets

    def is_dense_subobject(self, subobject: Subpresheaf, topology):
        """Return True when the J-closure of a subobject is the whole parent."""
        closure = self.subobject_closure(subobject, topology)
        return all(closure.subsets[obj] == subobject.parent.sets[obj] for obj in self.category.objects)

    def omega(self):
        """Return the subobject-classifier presheaf Omega of sieves."""
        sets = {obj: self.sieves_on(obj) for obj in self.category.objects}
        restrictions = {}
        for morphism, (_src, dst) in self.category.morphisms.items():
            restrictions[morphism] = {sieve: self.pullback_sieve(morphism, sieve) for sieve in sets[dst]}
        return Presheaf(self.category, sets, restrictions)

    def truth_map(self):
        """Truth morphism `true: 1 -> Omega`, selecting maximal sieves."""
        terminal = self.terminal_presheaf()
        omega = self.omega()
        return self.natural_transformation(
            source=terminal,
            target=omega,
            components={
                obj: {
                    next(iter(terminal.sets[obj])): self.maximal_sieve(obj),
                }
                for obj in self.category.objects
            },
        )

    def characteristic_map(self, subobject: Subpresheaf):
        """
        Return chi: F -> Omega classifying a subpresheaf S <= F.

        chi_A(x) is the sieve of arrows f: B -> A such that F(f)(x) is in S(B).
        """
        parent = subobject.parent
        omega = self.omega()
        components = {}

        for obj in self.category.objects:
            mapping = {}
            for element in parent.sets[obj]:
                sieve = frozenset(
                    arrow
                    for arrow in self.category.arrows_to(obj)
                    if parent.restrict(arrow, element) in subobject.subsets[self.category.source(arrow)]
                )
                if sieve not in omega.sets[obj]:
                    raise ValueError(f"Characteristic sieve at {obj!r} is not closed under precomposition.")
                mapping[element] = sieve
            components[obj] = mapping

        transformation = NaturalTransformation(source=parent, target=omega, components=components)
        transformation.validate_naturality()
        return transformation

    def pullback_truth(self, classifier: NaturalTransformation):
        """Recover the classified subpresheaf by pulling back true: 1 -> Omega."""
        subsets = {}
        for obj in self.category.objects:
            true_at_obj = self.maximal_sieve(obj)
            subsets[obj] = {element for element in classifier.source.sets[obj] if classifier.apply(obj, element) == true_at_obj}
        return Subpresheaf(parent=classifier.source, subsets=subsets)


class GrothendieckTopology:
    """Finite Grothendieck topology represented by covering sieves."""

    def __init__(self, category: FiniteCategory, covering_sieves):
        self.category = category
        self.covering_sieves = {
            obj: frozenset(frozenset(sieve) for sieve in covering_sieves.get(obj, ()))
            for obj in category.objects
        }
        self.validate()

    def validate(self):
        """Validate maximality, pullback stability, and transitivity axioms."""
        topos = PresheafTopos(self.category)

        for obj in self.category.objects:
            if topos.maximal_sieve(obj) not in self.covering_sieves[obj]:
                raise ValueError(f"Covering sieves at {obj!r} must include the maximal sieve.")
            for sieve in self.covering_sieves[obj]:
                if not topos._is_sieve(obj, sieve):
                    raise ValueError(f"Cover {sieve!r} is not a sieve on {obj!r}.")

        for obj in self.category.objects:
            for cover in self.covering_sieves[obj]:
                for morphism in self.category.arrows_to(obj):
                    source = self.category.source(morphism)
                    if topos.pullback_sieve(morphism, cover) not in self.covering_sieves[source]:
                        raise ValueError("Covering sieves must be stable under pullback.")

        for obj in self.category.objects:
            for sieve in topos.sieves_on(obj):
                for cover in self.covering_sieves[obj]:
                    pullbacks_cover = all(
                        topos.pullback_sieve(morphism, sieve)
                        in self.covering_sieves[self.category.source(morphism)]
                        for morphism in cover
                    )
                    if pullbacks_cover and sieve not in self.covering_sieves[obj]:
                        raise ValueError("Covering sieves must satisfy transitivity.")

        return True
