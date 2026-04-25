from topos_ai.formal_category import (
    FiniteCategory,
    Presheaf,
    PresheafTopos,
    category_of_elements,
    yoneda_density_colimit,
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


def main():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    elements, projection = category_of_elements(presheaf)
    density, to_presheaf, from_presheaf = yoneda_density_colimit(presheaf)

    return {
        "objects_in_category_of_elements": len(elements.objects),
        "projection_target_objects": len(set(projection.object_map.values())),
        "round_trip_to_presheaf": topos.compose_transformations(
            to_presheaf,
            from_presheaf,
        ).components == topos.identity_transformation(presheaf).components,
        "round_trip_to_density": topos.compose_transformations(
            from_presheaf,
            to_presheaf,
        ).components == topos.identity_transformation(density).components,
    }


if __name__ == "__main__":
    print(main())
