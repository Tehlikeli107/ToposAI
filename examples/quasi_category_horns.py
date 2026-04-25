from topos_ai.formal_category import FiniteCategory
from topos_ai.infinity_categories import FiniteHorn, nerve_2_skeleton, nerve_3_skeleton


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


def main():
    category = thin_chain_category()
    nerve_2 = nerve_2_skeleton(category)
    nerve_3 = nerve_3_skeleton(category)
    horn = FiniteHorn(2, 1, {0: ("mor", "m12"), 2: ("mor", "m01")})

    return {
        "inner_kan_2": nerve_2.is_inner_kan(max_dimension=2),
        "inner_kan_3": nerve_3.is_inner_kan(max_dimension=3),
        "composition_filler": nerve_2.horn_fillers(horn)[0],
        "assoc_3_simplices": len(nerve_3.simplices[3]),
        "degeneracy_identities": nerve_3.validate_degeneracy_identities(),
    }


if __name__ == "__main__":
    print(main())