from examples.formal_yoneda_density import walking_arrow_category
from topos_ai.formal_category import Presheaf, PresheafTopos, Subpresheaf


def main():
    category = walking_arrow_category()
    presheaf = Presheaf(
        category,
        sets={"0": {"a", "b"}, "1": {"u"}},
        restrictions={"id0": {"a": "a", "b": "b"}, "id1": {"u": "u"}, "up": {"u": "a"}},
    )
    topos = PresheafTopos(category)
    subobject = Subpresheaf(presheaf, {"0": {"a"}, "1": {"u"}})
    identity = topos.identity_transformation(presheaf)

    return {
        "forcing_at_1": subobject.parent.category.identities["1"] in topos.forcing_sieve(subobject, "1", "u"),
        "forces_closed_truth": topos.forces(subobject, "1", "u"),
        "equality_truth_is_maximal": topos.equality_truth(presheaf, "1", "u", "u") == topos.maximal_sieve("1"),
        "quantifier_adjunctions_valid": topos.validate_quantifier_adjunctions(identity),
    }


if __name__ == "__main__":
    print(main())
