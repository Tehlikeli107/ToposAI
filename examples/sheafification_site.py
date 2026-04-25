from examples.formal_yoneda_density import walking_arrow_category
from topos_ai.formal_category import GrothendieckTopology, Presheaf, PresheafTopos


def main():
    category = walking_arrow_category()
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"id1", "up"}), frozenset({"up"})},
        },
    )
    presheaf = Presheaf(
        category,
        sets={"0": {"a"}, "1": {"u", "v"}},
        restrictions={"id0": {"a": "a"}, "id1": {"u": "u", "v": "v"}, "up": {"u": "a", "v": "a"}},
    )
    sheaf, _ = topos.sheafification(presheaf, topology)
    identity = topos.identity_transformation(sheaf)
    _, _, factorization = topos.sheafification_factorization(identity, topology)

    return {
        "is_sheaf_before_sheafification": topos.is_sheaf(presheaf, topology),
        "is_sheaf_after_sheafification": topos.is_sheaf(sheaf, topology),
        "factorization_valid": factorization.validate_naturality(),
    }


if __name__ == "__main__":
    print(main())