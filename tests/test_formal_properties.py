from topos_ai.formal_category import (
    FiniteCategory,
    Presheaf,
    PresheafTopos,
    Subpresheaf,
    category_of_elements,
    yoneda_density_colimit,
)
from topos_ai.infinity_categories import nerve_3_skeleton


def terminal_category():
    return FiniteCategory(
        objects=("*",),
        morphisms={"id": ("*", "*")},
        identities={"*": "id"},
        composition={("id", "id"): "id"},
    )


def walking_arrow_category():
    return FiniteCategory(
        objects=("0", "1"),
        morphisms={"id0": ("0", "0"), "id1": ("1", "1"), "up": ("0", "1")},
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0",
            ("id1", "id1"): "id1",
            ("up", "id0"): "up",
            ("id1", "up"): "up",
        },
    )


def thin_chain_category():
    objects = ("0", "1", "2", "3")
    morphisms = {}
    identities = {}
    for i, src in enumerate(objects):
        identity = f"id{src}"
        identities[src] = identity
        for dst in objects[i:]:
            label = identity if src == dst else f"m{src}{dst}"
            morphisms[label] = (src, dst)
    composition = {}
    for before, (src, middle) in morphisms.items():
        for after, (after_src, dst) in morphisms.items():
            if middle == after_src:
                composition[(after, before)] = identities[src] if src == dst else f"m{src}{dst}"
    return FiniteCategory(objects, morphisms, identities, composition)


def test_yoneda_density_round_trip_across_small_presheaves():
    fixtures = [
        Presheaf(
            terminal_category(),
            sets={"*": {"x", "y"}},
            restrictions={"id": {"x": "x", "y": "y"}},
        ),
        Presheaf(
            walking_arrow_category(),
            sets={"0": {"a", "b"}, "1": {"u"}},
            restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
        ),
    ]

    for presheaf in fixtures:
        topos = PresheafTopos(presheaf.category)
        elements, projection = category_of_elements(presheaf)
        density, to_presheaf, from_presheaf = yoneda_density_colimit(presheaf)

        assert len(elements.objects) == sum(len(values) for values in presheaf.sets.values())
        assert set(projection.object_map.values()).issubset(set(presheaf.category.objects))
        assert topos.compose_transformations(to_presheaf, from_presheaf).components == (
            topos.identity_transformation(presheaf).components
        )
        assert topos.compose_transformations(from_presheaf, to_presheaf).components == (
            topos.identity_transformation(density).components
        )


def test_subobject_heyting_adjunction_across_all_subobjects_of_walking_arrow():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    subobjects = topos.subobjects(presheaf)

    def leq(left, right):
        return all(left.subsets[obj].issubset(right.subsets[obj]) for obj in category.objects)

    for a in subobjects:
        for b in subobjects:
            implication = topos.subobject_implication(a, b)
            for x in subobjects:
                assert leq(topos.subobject_meet(x, a), b) == leq(x, implication)


def test_inverse_image_preserves_meet_for_all_subobjects():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    identity = topos.identity_transformation(presheaf)
    subobjects = topos.subobjects(presheaf)

    for left in subobjects:
        for right in subobjects:
            pulled_meet = topos.inverse_image(identity, topos.subobject_meet(left, right))
            meet_pulled = topos.subobject_meet(
                topos.inverse_image(identity, left),
                topos.inverse_image(identity, right),
            )
            assert pulled_meet.subsets == meet_pulled.subsets


def test_nerve_3_skeleton_inner_kan_for_thin_chain():
    nerve = nerve_3_skeleton(thin_chain_category())
    assert nerve.validate_face_identities() is True
    assert nerve.validate_degeneracy_identities() is True
    assert nerve.is_inner_kan(max_dimension=3) is True