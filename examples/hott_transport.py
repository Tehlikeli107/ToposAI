from topos_ai.hott import FinitePathGroupoid, PathFamily


def main():
    paths = FinitePathGroupoid(
        objects=("A", "B"),
        paths={"idA": ("A", "A"), "idB": ("B", "B"), "p": ("A", "B"), "p_inv": ("B", "A")},
        identities={"A": "idA", "B": "idB"},
        inverses={"idA": "idA", "idB": "idB", "p": "p_inv", "p_inv": "p"},
        composition={
            ("idA", "idA"): "idA",
            ("idB", "idB"): "idB",
            ("p", "idA"): "p",
            ("idB", "p"): "p",
            ("p_inv", "idB"): "p_inv",
            ("idA", "p_inv"): "p_inv",
            ("p_inv", "p"): "idA",
            ("p", "p_inv"): "idB",
        },
    )
    family = PathFamily(
        base=paths,
        fibers={"A": {0, 1}, "B": {"zero", "one"}},
        transports={
            "idA": {0: 0, 1: 1},
            "idB": {"zero": "zero", "one": "one"},
            "p": {0: "zero", 1: "one"},
            "p_inv": {"zero": 0, "one": 1},
        },
    )

    return {
        "identity_type_size": len(paths.identity_type("A", "B")),
        "transported_one": family.transport("p", 1),
        "transport_functorial": family.validate_functorial_transport(),
    }


if __name__ == "__main__":
    print(main())
