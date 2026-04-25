from topos_ai.formal_category import (
    FiniteCategory,
    FiniteFunctor,
    GrothendieckTopology,
    Presheaf,
    PresheafTopos,
    Subpresheaf,
    category_of_elements,
    natural_transformations,
    representable_presheaf,
    yoneda_density_colimit,
    yoneda_element_to_transformation,
    yoneda_lemma_bijection,
    yoneda_transformation_to_element,
)


def walking_arrow_category():
    """Return the finite category 0 -> 1."""
    return FiniteCategory(
        objects=("0", "1"),
        morphisms={
            "id0": ("0", "0"),
            "id1": ("1", "1"),
            "up": ("0", "1"),
        },
        identities={"0": "id0", "1": "id1"},
        composition={
            ("id0", "id0"): "id0",
            ("id1", "id1"): "id1",
            ("up", "id0"): "up",
            ("id1", "up"): "up",
        },
    )


def walking_arrow_presheaf(category):
    return Presheaf(
        category=category,
        sets={
            "0": {"a", "b"},
            "1": {"u"},
        },
        restrictions={
            "id0": {"a": "a", "b": "b"},
            "id1": {"u": "u"},
            "up": {"u": "a"},
        },
    )


def terminal_category():
    return FiniteCategory(
        objects=("*",),
        morphisms={"id*": ("*", "*")},
        identities={"*": "id*"},
        composition={("id*", "id*"): "id*"},
    )


def test_finite_category_validates_identities_and_associativity():
    category = walking_arrow_category()

    assert category.validate_laws() is True
    assert category.compose("up", "id0") == "up"
    assert category.compose("id1", "up") == "up"
    assert category.hom("0", "1") == ("up",)


def test_representable_presheaf_and_yoneda_round_trip():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)

    y1 = representable_presheaf(category, "1")
    assert y1.sets["0"] == frozenset({"up"})
    assert y1.sets["1"] == frozenset({"id1"})
    assert y1.restrict("up", "id1") == "up"

    transformation = yoneda_element_to_transformation(category, "1", presheaf, "u")
    assert transformation.validate_naturality() is True
    assert transformation.apply("1", "id1") == "u"
    assert transformation.apply("0", "up") == "a"
    assert yoneda_transformation_to_element("1", transformation) == "u"

    y0_to_b = yoneda_element_to_transformation(category, "0", presheaf, "b")
    assert y0_to_b.validate_naturality() is True
    assert y0_to_b.apply("0", "id0") == "b"
    assert yoneda_transformation_to_element("0", y0_to_b) == "b"


def test_natural_transformation_enumeration_matches_yoneda_bijection():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    y0 = representable_presheaf(category, "0")

    transformations = natural_transformations(y0, presheaf)
    assert len(transformations) == 2
    assert {yoneda_transformation_to_element("0", alpha) for alpha in transformations} == {"a", "b"}

    bijection = yoneda_lemma_bijection(category, "0", presheaf)
    assert set(bijection) == {"a", "b"}
    assert bijection["a"].apply("0", "id0") == "a"
    assert bijection["b"].apply("0", "id0") == "b"


def test_category_of_elements_and_yoneda_density_reconstruct_presheaf():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    elements, projection = category_of_elements(presheaf)
    density, to_presheaf, from_presheaf = yoneda_density_colimit(presheaf)

    assert set(elements.objects) == {("0", "a"), ("0", "b"), ("1", "u")}
    assert elements.hom(("0", "a"), ("1", "u")) == (("element", "up", "u"),)
    assert elements.hom(("0", "b"), ("1", "u")) == ()
    assert projection.map_object(("1", "u")) == "1"
    assert projection.map_morphism(("element", "up", "u")) == "up"

    assert topos.compose_transformations(to_presheaf, from_presheaf).components == (
        topos.identity_transformation(presheaf).components
    )
    assert topos.compose_transformations(from_presheaf, to_presheaf).components == (
        topos.identity_transformation(density).components
    )
    assert to_presheaf.apply("1", from_presheaf.apply("1", "u")) == "u"
    assert density.restrict("up", from_presheaf.apply("1", "u")) == from_presheaf.apply("0", "a")


def test_presheaf_topos_sieves_and_subobject_classifier():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    assert topos.sieves_on("0") == frozenset(
        {
            frozenset(),
            frozenset({"id0"}),
        }
    )
    assert topos.sieves_on("1") == frozenset(
        {
            frozenset(),
            frozenset({"up"}),
            frozenset({"id1", "up"}),
        }
    )
    assert topos.pullback_sieve("up", frozenset({"up"})) == frozenset({"id0"})

    subobject = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a"},
            "1": set(),
        },
    )

    classifier = topos.characteristic_map(subobject)
    assert classifier.validate_naturality() is True
    assert classifier.apply("0", "a") == frozenset({"id0"})
    assert classifier.apply("0", "b") == frozenset()
    assert classifier.apply("1", "u") == frozenset({"up"})

    recovered = topos.pullback_truth(classifier)
    assert recovered.subsets["0"] == frozenset({"a"})
    assert recovered.subsets["1"] == frozenset()


def test_presheaf_topos_products_and_projections_are_pointwise():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    product_presheaf, first_projection, second_projection = topos.product_presheaf(presheaf, presheaf)

    assert product_presheaf.sets["0"] == frozenset(
        {
            ("a", "a"),
            ("a", "b"),
            ("b", "a"),
            ("b", "b"),
        }
    )
    assert product_presheaf.sets["1"] == frozenset({("u", "u")})
    assert product_presheaf.restrict("up", ("u", "u")) == ("a", "a")
    assert first_projection.validate_naturality() is True
    assert second_projection.validate_naturality() is True
    assert first_projection.apply("0", ("b", "a")) == "b"
    assert second_projection.apply("0", ("b", "a")) == "a"


def test_presheaf_topos_equalizer_is_stable_under_restriction():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    identity = topos.identity_transformation(presheaf)
    collapse_to_a = topos.natural_transformation(
        source=presheaf,
        target=presheaf,
        components={
            "0": {"a": "a", "b": "a"},
            "1": {"u": "u"},
        },
    )

    equalizer = topos.equalizer(identity, collapse_to_a)
    assert equalizer.subsets["0"] == frozenset({"a"})
    assert equalizer.subsets["1"] == frozenset({"u"})


def test_presheaf_topos_exponential_uses_yoneda_power_object_formula():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    terminal = topos.terminal_presheaf()

    exponential = topos.exponential_presheaf(terminal, presheaf)

    assert len(exponential.sets["0"]) == len(presheaf.sets["0"])
    assert len(exponential.sets["1"]) == len(presheaf.sets["1"])

    terminal_0 = next(iter(terminal.sets["0"]))
    terminal_1 = next(iter(terminal.sets["1"]))
    value_at_1 = next(iter(exponential.sets["1"]))

    assert value_at_1.apply("1", ("id1", terminal_1)) == "u"

    restricted_to_0 = exponential.restrict("up", value_at_1)
    assert restricted_to_0.apply("0", ("id0", terminal_0)) == "a"
    assert {alpha.apply("0", ("id0", terminal_0)) for alpha in exponential.sets["0"]} == {"a", "b"}


def test_presheaf_topos_initial_and_coproduct_are_pointwise():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    terminal = topos.terminal_presheaf()

    initial = topos.initial_presheaf()
    assert initial.sets["0"] == frozenset()
    assert initial.sets["1"] == frozenset()

    coproduct, left_injection, right_injection = topos.coproduct_presheaf(presheaf, terminal)

    terminal_0 = next(iter(terminal.sets["0"]))
    terminal_1 = next(iter(terminal.sets["1"]))
    assert coproduct.sets["0"] == frozenset(
        {
            ("left", "a"),
            ("left", "b"),
            ("right", terminal_0),
        }
    )
    assert coproduct.sets["1"] == frozenset({("left", "u"), ("right", terminal_1)})
    assert coproduct.restrict("up", ("left", "u")) == ("left", "a")
    assert coproduct.restrict("up", ("right", terminal_1)) == ("right", terminal_0)
    assert left_injection.apply("0", "b") == ("left", "b")
    assert right_injection.apply("1", terminal_1) == ("right", terminal_1)


def test_presheaf_topos_pullback_is_fiber_product_of_natural_transformations():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    identity = topos.identity_transformation(presheaf)
    collapse_to_a = topos.natural_transformation(
        source=presheaf,
        target=presheaf,
        components={
            "0": {"a": "a", "b": "a"},
            "1": {"u": "u"},
        },
    )

    pullback, first_projection, second_projection = topos.pullback(identity, collapse_to_a)

    assert pullback.sets["0"] == frozenset({("a", "a"), ("a", "b")})
    assert pullback.sets["1"] == frozenset({("u", "u")})
    assert pullback.restrict("up", ("u", "u")) == ("a", "a")
    assert first_projection.apply("0", ("a", "b")) == "a"
    assert second_projection.apply("0", ("a", "b")) == "b"


def test_subobject_heyting_operations_are_stable_under_pullback():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    only_a = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a"},
            "1": set(),
        },
    )
    only_b = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"b"},
            "1": set(),
        },
    )

    meet = topos.subobject_meet(only_a, only_b)
    join = topos.subobject_join(only_a, only_b)
    implication = topos.subobject_implication(only_a, only_b)
    negation = topos.subobject_negation(only_a)

    assert meet.subsets["0"] == frozenset()
    assert meet.subsets["1"] == frozenset()
    assert join.subsets["0"] == frozenset({"a", "b"})
    assert join.subsets["1"] == frozenset()
    assert implication.subsets["0"] == frozenset({"b"})
    assert implication.subsets["1"] == frozenset()
    assert negation.subsets["0"] == frozenset({"b"})
    assert negation.subsets["1"] == frozenset()

    assert topos.subobject_implication(only_a, only_a).subsets["0"] == presheaf.sets["0"]
    assert topos.subobject_implication(only_a, only_a).subsets["1"] == presheaf.sets["1"]


def test_power_object_membership_and_subobject_name_round_trip():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    only_a = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a"},
            "1": set(),
        },
    )

    name = topos.name_subobject(only_a)
    power, product_object, membership = topos.membership_relation(presheaf, power=name.target)

    terminal_0 = next(iter(name.source.sets["0"]))
    terminal_1 = next(iter(name.source.sets["1"]))
    named_at_0 = name.apply("0", terminal_0)
    named_at_1 = name.apply("1", terminal_1)

    assert power is name.target
    assert (("a", named_at_0)) in membership.subsets["0"]
    assert (("b", named_at_0)) not in membership.subsets["0"]
    assert (("u", named_at_1)) not in membership.subsets["1"]
    assert product_object.restrict("up", ("u", named_at_1)) == ("a", named_at_0)

    recovered = topos.extension_of_name(presheaf, name)
    assert recovered.subsets["0"] == only_a.subsets["0"]
    assert recovered.subsets["1"] == only_a.subsets["1"]


def test_image_kernel_pair_and_coequalizer_are_computed_pointwise():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    collapse_to_a = topos.natural_transformation(
        source=presheaf,
        target=presheaf,
        components={
            "0": {"a": "a", "b": "a"},
            "1": {"u": "u"},
        },
    )

    image = topos.image(collapse_to_a)
    assert image.subsets["0"] == frozenset({"a"})
    assert image.subsets["1"] == frozenset({"u"})

    kernel_pair, first_projection, second_projection = topos.kernel_pair(collapse_to_a)
    assert kernel_pair.sets["0"] == frozenset(
        {
            ("a", "a"),
            ("a", "b"),
            ("b", "a"),
            ("b", "b"),
        }
    )
    assert first_projection.apply("0", ("b", "a")) == "b"
    assert second_projection.apply("0", ("b", "a")) == "a"

    quotient, projection = topos.coequalizer(first_projection, second_projection)
    class_0 = projection.apply("0", "a")
    class_1 = projection.apply("1", "u")
    assert quotient.sets["0"] == frozenset({frozenset({"a", "b"})})
    assert quotient.sets["1"] == frozenset({frozenset({"u"})})
    assert quotient.restrict("up", class_1) == class_0


def test_image_factorization_truth_and_inverse_image_are_categorical():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)

    collapse_to_a = topos.natural_transformation(
        source=presheaf,
        target=presheaf,
        components={
            "0": {"a": "a", "b": "a"},
            "1": {"u": "u"},
        },
    )
    only_a = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a"},
            "1": set(),
        },
    )

    truth = topos.truth_map()
    terminal_1 = next(iter(truth.source.sets["1"]))
    assert truth.apply("1", terminal_1) == topos.maximal_sieve("1")

    inverse = topos.inverse_image(collapse_to_a, only_a)
    assert inverse.subsets["0"] == frozenset({"a", "b"})
    assert inverse.subsets["1"] == frozenset()

    image_object, epi, mono = topos.image_factorization(collapse_to_a)
    recomposed = topos.compose_transformations(mono, epi)

    assert image_object.sets["0"] == frozenset({"a"})
    assert image_object.sets["1"] == frozenset({"u"})
    assert epi.apply("0", "b") == "a"
    assert mono.apply("1", "u") == "u"
    assert topos.is_epimorphism(epi) is True
    assert topos.is_monomorphism(mono) is True
    assert recomposed.components == collapse_to_a.components
    assert topos.validate_regular_image_factorization(collapse_to_a) is True
    assert topos.validate_effective_epimorphism(epi) is True


def test_exponential_evaluation_and_transpose_are_cartesian_closed():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    terminal = topos.terminal_presheaf()

    power, evaluation_product, evaluation = topos.evaluation_map(terminal, presheaf)
    terminal_0 = next(iter(terminal.sets["0"]))
    terminal_1 = next(iter(terminal.sets["1"]))
    alpha_1 = next(iter(power.sets["1"]))

    assert evaluation_product.restrict("up", (alpha_1, terminal_1)) == (
        power.restrict("up", alpha_1),
        terminal_0,
    )
    assert evaluation.apply("1", (alpha_1, terminal_1)) == "u"
    assert evaluation.apply("0", (power.restrict("up", alpha_1), terminal_0)) == "a"

    product_ht, _pi_h, _pi_t = topos.product_presheaf(terminal, terminal)
    constant_a = topos.natural_transformation(
        source=product_ht,
        target=presheaf,
        components={
            "0": {(terminal_0, terminal_0): "a"},
            "1": {(terminal_1, terminal_1): "u"},
        },
    )

    transposed_power, transpose = topos.transpose(constant_a, terminal, terminal, presheaf)
    transposed_alpha_1 = transpose.apply("1", terminal_1)
    _same_power, _same_product, same_eval = topos.evaluation_map(terminal, presheaf, power=transposed_power)

    assert transposed_alpha_1.apply("1", ("id1", terminal_1)) == "u"
    assert same_eval.apply("1", (transposed_alpha_1, terminal_1)) == "u"


def test_finite_elementary_topos_universal_property_validators():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    terminal = topos.terminal_presheaf()
    identity = topos.identity_transformation(presheaf)
    collapse_to_a = topos.natural_transformation(
        source=presheaf,
        target=presheaf,
        components={
            "0": {"a": "a", "b": "a"},
            "1": {"u": "u"},
        },
    )

    assert topos.validate_product_universal_property(presheaf, terminal, presheaf) is True
    assert topos.validate_pullback_universal_property(identity, collapse_to_a, presheaf) is True
    assert topos.validate_equalizer_universal_property(identity, collapse_to_a, terminal) is True
    assert topos.validate_coproduct_universal_property(presheaf, terminal, presheaf) is True

    kernel_pair, first_projection, second_projection = topos.kernel_pair(collapse_to_a)
    assert topos.validate_coequalizer_universal_property(first_projection, second_projection, terminal) is True
    assert topos.validate_exponential_adjunction(terminal, terminal, presheaf) is True
    assert topos.validate_subobject_classifier_universal_property(presheaf) is True


def test_grothendieck_topology_and_sheaf_condition_use_covering_sieves():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    terminal = topos.terminal_presheaf()

    topology = GrothendieckTopology(
        category=category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"up"}), frozenset({"id1", "up"})},
        },
    )

    cover = frozenset({"up"})
    families = topos.matching_families(presheaf, "1", cover)
    by_value = {family["up"]: family for family in families}

    assert topology.validate() is True
    assert set(by_value) == {"a", "b"}
    assert topos.amalgamations(presheaf, "1", by_value["a"]) == ("u",)
    assert topos.amalgamations(presheaf, "1", by_value["b"]) == ()
    assert topos.is_sheaf(terminal, topology) is True
    assert topos.is_sheaf(presheaf, topology) is False


def test_plus_construction_and_sheafification_add_missing_local_gluings():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category=category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"up"}), frozenset({"id1", "up"})},
        },
    )

    plus, unit = topos.plus_construction(presheaf, topology)

    assert len(plus.sets["0"]) == 2
    assert len(plus.sets["1"]) == 2
    assert topos.is_sheaf(plus, topology) is True
    assert plus.restrict("up", unit.apply("1", "u")) == unit.apply("0", "a")
    assert {plus.restrict("up", element) for element in plus.sets["1"]} == plus.sets["0"]

    sheaf, sheaf_unit = topos.sheafification(presheaf, topology)

    assert len(sheaf.sets["0"]) == 2
    assert len(sheaf.sets["1"]) == 2
    assert topos.is_sheaf(sheaf, topology) is True
    assert sheaf.restrict("up", sheaf_unit.apply("1", "u")) == sheaf_unit.apply("0", "a")


def test_sheafification_reflector_factors_maps_to_sheaves():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category=category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"up"}), frozenset({"id1", "up"})},
        },
    )
    plus, unit = topos.plus_construction(presheaf, topology)

    sheaf, sheaf_unit, factor = topos.sheafification_factorization(unit, topology)
    recomposed = topos.compose_transformations(factor, sheaf_unit)

    assert factor.target is plus
    assert topos.is_sheaf(sheaf, topology) is True
    assert topos.is_monomorphism(factor) is True
    assert topos.is_epimorphism(factor) is True
    assert recomposed.components == unit.components


def test_lawvere_tierney_operator_closure_and_closed_sieve_classifier():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category=category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"up"}), frozenset({"id1", "up"})},
        },
    )
    only_a = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a"},
            "1": set(),
        },
    )
    dense_base = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a", "b"},
            "1": set(),
        },
    )

    j = topos.lawvere_tierney_operator(topology)
    omega_j = topos.omega_j(topology)
    closure = topos.subobject_closure(only_a, topology)

    assert j.validate_naturality() is True
    assert j.apply("1", frozenset({"up"})) == frozenset({"id1", "up"})
    assert j.apply("0", frozenset()) == frozenset()
    assert omega_j.sets["1"] == frozenset({frozenset(), frozenset({"id1", "up"})})
    assert omega_j.restrict("up", frozenset({"id1", "up"})) == frozenset({"id0"})

    assert closure.subsets["0"] == frozenset({"a"})
    assert closure.subsets["1"] == frozenset({"u"})
    assert topos.subobject_closure(closure, topology).subsets == closure.subsets
    assert topos.is_j_closed_subobject(only_a, topology) is False
    assert topos.is_j_closed_subobject(closure, topology) is True
    assert topos.is_dense_subobject(dense_base, topology) is True


def test_j_sheaf_subobject_classifier_recovers_closed_subobjects():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category=category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"up"}), frozenset({"id1", "up"})},
        },
    )
    only_a = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a"},
            "1": set(),
        },
    )
    closure = topos.subobject_closure(only_a, topology)

    raw_classifier = topos.characteristic_map(only_a)
    modal_classifier = topos.characteristic_map_j(only_a, topology)
    closed_classifier = topos.characteristic_map_j(closure, topology)
    recovered = topos.pullback_truth_j(closed_classifier, topology)
    truth_j = topos.truth_map_j(topology)

    assert topos.validate_lawvere_tierney_axioms(topology) is True
    assert raw_classifier.apply("1", "u") == frozenset({"up"})
    assert modal_classifier.apply("1", "u") == frozenset({"id1", "up"})
    assert modal_classifier.components == closed_classifier.components
    assert closed_classifier.target.sets == topos.omega_j(topology).sets
    assert recovered.subsets == closure.subsets
    assert truth_j.apply("1", next(iter(truth_j.source.sets["1"]))) == topos.maximal_sieve("1")


def test_lawvere_tierney_reconstructs_topology_and_omega_j_heyting_logic():
    category = walking_arrow_category()
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category=category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"up"}), frozenset({"id1", "up"})},
        },
    )

    j = topos.lawvere_tierney_operator(topology)
    reconstructed = topos.topology_from_lawvere_tierney_operator(j)
    bottom_1 = frozenset()
    cover_1 = frozenset({"up"})
    top_1 = topos.maximal_sieve("1")

    assert reconstructed.covering_sieves == topology.covering_sieves
    assert topos.sieve_j_join(topology, "1", bottom_1, cover_1) == top_1
    assert topos.sieve_j_meet(topology, "1", top_1, cover_1) == top_1
    assert topos.sieve_j_implication(topology, "1", top_1, bottom_1) == bottom_1
    assert topos.sieve_j_implication(topology, "1", bottom_1, top_1) == top_1
    assert topos.sieve_j_negation(topology, "1", top_1) == bottom_1
    assert topos.validate_omega_j_heyting_laws(topology) is True


def test_j_closed_subobjects_form_internal_heyting_algebra():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    topology = GrothendieckTopology(
        category=category,
        covering_sieves={
            "0": {frozenset({"id0"})},
            "1": {frozenset({"up"}), frozenset({"id1", "up"})},
        },
    )
    only_a = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"a"},
            "1": set(),
        },
    )
    only_b = Subpresheaf(
        parent=presheaf,
        subsets={
            "0": {"b"},
            "1": set(),
        },
    )
    closed_a = topos.subobject_closure(only_a, topology)
    bottom = topos.subobject_bottom(presheaf)
    top = topos.subobject_top(presheaf)

    assert topos.subobject_j_meet(closed_a, only_b, topology).subsets == bottom.subsets
    assert topos.subobject_j_join(closed_a, only_b, topology).subsets == top.subsets
    assert topos.subobject_j_implication(closed_a, only_b, topology).subsets == only_b.subsets
    assert topos.subobject_j_negation(closed_a, topology).subsets == only_b.subsets
    assert {tuple(sorted(item.subsets["0"])) for item in topos.j_closed_subobjects(presheaf, topology)} == {
        tuple(),
        ("a",),
        ("b",),
        ("a", "b"),
    }
    assert topos.validate_j_subobject_heyting_laws(presheaf, topology) is True


def test_kripke_joyal_forcing_and_internal_quantifiers_are_adjoints():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    terminal = topos.terminal_presheaf()
    product_tf, pi_terminal, _pi_f = topos.product_presheaf(terminal, presheaf)
    terminal_0 = next(iter(terminal.sets["0"]))
    terminal_1 = next(iter(terminal.sets["1"]))
    pair_0_a = (terminal_0, "a")
    pair_0_b = (terminal_0, "b")
    pair_1_u = (terminal_1, "u")
    predicate_a = Subpresheaf(
        parent=product_tf,
        subsets={
            "0": {pair_0_a},
            "1": set(),
        },
    )
    predicate_top = topos.subobject_top(product_tf)

    exists_a = topos.exists_along(pi_terminal, predicate_a)
    forall_a = topos.forall_along(pi_terminal, predicate_a)
    forall_top = topos.forall_along(pi_terminal, predicate_top)
    local_truth = topos.truth_value(exists_a)

    assert topos.forcing_sieve(predicate_a, "1", pair_1_u) == frozenset({"up"})
    assert topos.forces(predicate_a, "0", pair_0_a) is True
    assert topos.forces(predicate_a, "0", pair_0_b) is False
    assert topos.forces(predicate_a, "1", pair_1_u) is False
    assert exists_a.subsets["0"] == frozenset({terminal_0})
    assert exists_a.subsets["1"] == frozenset()
    assert forall_a.subsets["0"] == frozenset()
    assert forall_a.subsets["1"] == frozenset()
    assert forall_top.subsets == topos.subobject_top(terminal).subsets
    assert local_truth.apply("1", terminal_1) == frozenset({"up"})
    assert topos.validate_quantifier_adjunctions(pi_terminal) is True


def test_internal_equality_frobenius_and_beck_chevalley_for_quantifiers():
    category = walking_arrow_category()
    presheaf = walking_arrow_presheaf(category)
    topos = PresheafTopos(category)
    equality = topos.equality_subobject(presheaf)
    diagonal = topos.diagonal_transformation(presheaf)
    diagonal_image = topos.image(diagonal)
    identity = topos.identity_transformation(presheaf)
    collapse_to_a = topos.natural_transformation(
        source=presheaf,
        target=presheaf,
        components={
            "0": {"a": "a", "b": "a"},
            "1": {"u": "u"},
        },
    )

    assert equality.subsets["0"] == frozenset({("a", "a"), ("b", "b")})
    assert equality.subsets["1"] == frozenset({("u", "u")})
    assert diagonal.apply("0", "b") == ("b", "b")
    assert diagonal_image.subsets == equality.subsets
    assert topos.equality_truth(presheaf, "1", "u", "u") == topos.maximal_sieve("1")
    assert topos.equality_truth(presheaf, "0", "a", "b") == frozenset()
    assert topos.validate_frobenius_reciprocity(collapse_to_a) is True
    assert topos.validate_beck_chevalley(collapse_to_a, identity) is True


def test_reindexing_along_finite_functor_is_inverse_image_of_presheaf_topoi():
    walking = walking_arrow_category()
    terminal = terminal_category()
    functor = FiniteFunctor(
        source=walking,
        target=terminal,
        object_map={"0": "*", "1": "*"},
        morphism_map={"id0": "id*", "id1": "id*", "up": "id*"},
    )
    terminal_topos = PresheafTopos(terminal)
    walking_topos = PresheafTopos(walking)
    colors = Presheaf(
        category=terminal,
        sets={"*": {"red", "blue"}},
        restrictions={"id*": {"red": "red", "blue": "blue"}},
    )

    reindexed = walking_topos.reindex_presheaf(functor, colors)
    product_colors, _left, _right = terminal_topos.product_presheaf(colors, colors)
    reindexed_product = walking_topos.reindex_presheaf(functor, product_colors)
    product_reindexed, _pi_left, _pi_right = walking_topos.product_presheaf(reindexed, reindexed)
    collapse = terminal_topos.natural_transformation(
        source=colors,
        target=colors,
        components={"*": {"red": "red", "blue": "red"}},
    )
    reindexed_collapse = walking_topos.reindex_transformation(functor, collapse)

    assert functor.validate() is True
    assert reindexed.sets["0"] == frozenset({"red", "blue"})
    assert reindexed.sets["1"] == frozenset({"red", "blue"})
    assert reindexed.restrict("up", "blue") == "blue"
    assert reindexed_product.sets == product_reindexed.sets
    assert reindexed_product.restrictions == product_reindexed.restrictions
    assert reindexed_collapse.apply("0", "blue") == "red"
    assert reindexed_collapse.apply("1", "blue") == "red"


def test_finite_kan_extensions_are_adjoints_to_reindexing():
    walking = walking_arrow_category()
    terminal = terminal_category()
    functor = FiniteFunctor(
        source=walking,
        target=terminal,
        object_map={"0": "*", "1": "*"},
        morphism_map={"id0": "id*", "id1": "id*", "up": "id*"},
    )
    presheaf = walking_arrow_presheaf(walking)
    colors = Presheaf(
        category=terminal,
        sets={"*": {"red", "blue"}},
        restrictions={"id*": {"red": "red", "blue": "blue"}},
    )
    walking_topos = PresheafTopos(walking)
    terminal_topos = PresheafTopos(terminal)

    sigma = terminal_topos.left_kan_extension_presheaf(functor, presheaf)
    pi = terminal_topos.right_kan_extension_presheaf(functor, presheaf)
    reindexed_colors = walking_topos.reindex_presheaf(functor, colors)

    assert len(sigma.sets["*"]) == 2
    assert len(pi.sets["*"]) == 1
    assert len(natural_transformations(sigma, colors)) == len(natural_transformations(presheaf, reindexed_colors))
    assert len(natural_transformations(reindexed_colors, presheaf)) == len(natural_transformations(colors, pi))


def test_finite_kan_extensions_expose_adjunction_witnesses():
    walking = walking_arrow_category()
    terminal = terminal_category()
    functor = FiniteFunctor(
        source=walking,
        target=terminal,
        object_map={"0": "*", "1": "*"},
        morphism_map={"id0": "id*", "id1": "id*", "up": "id*"},
    )
    presheaf = walking_arrow_presheaf(walking)
    colors = Presheaf(
        category=terminal,
        sets={"*": {"red", "blue"}},
        restrictions={"id*": {"red": "red", "blue": "blue"}},
    )
    walking_topos = PresheafTopos(walking)
    terminal_topos = PresheafTopos(terminal)

    sigma = terminal_topos.left_kan_extension_presheaf(functor, presheaf)
    reindexed_colors = walking_topos.reindex_presheaf(functor, colors)
    ordered_sigma_values = sorted(sigma.sets["*"], key=repr)
    alpha = terminal_topos.natural_transformation(
        source=sigma,
        target=colors,
        components={"*": {ordered_sigma_values[0]: "red", ordered_sigma_values[1]: "blue"}},
    )
    beta = walking_topos.natural_transformation(
        source=presheaf,
        target=reindexed_colors,
        components={"0": {"a": "red", "b": "blue"}, "1": {"u": "red"}},
    )

    transposed_alpha = terminal_topos.left_kan_transpose(functor, presheaf, colors, alpha)
    untransposed_beta = terminal_topos.left_kan_untranspose(functor, presheaf, colors, beta)

    assert terminal_topos.left_kan_untranspose(functor, presheaf, colors, transposed_alpha).components == alpha.components
    assert terminal_topos.left_kan_transpose(functor, presheaf, colors, untransposed_beta).components == beta.components
    assert terminal_topos.left_kan_unit(functor, presheaf).source is presheaf
    assert terminal_topos.left_kan_counit(functor, colors).target is colors
    assert terminal_topos.validate_left_kan_adjunction(functor, presheaf, colors) is True

    pi = terminal_topos.right_kan_extension_presheaf(functor, presheaf)
    right_alpha = walking_topos.natural_transformation(
        source=reindexed_colors,
        target=presheaf,
        components={"0": {"red": "a", "blue": "a"}, "1": {"red": "u", "blue": "u"}},
    )
    only_pi_value = next(iter(pi.sets["*"]))
    right_beta = terminal_topos.natural_transformation(
        source=colors,
        target=pi,
        components={"*": {"red": only_pi_value, "blue": only_pi_value}},
    )

    right_transposed_alpha = terminal_topos.right_kan_transpose(functor, colors, presheaf, right_alpha)
    right_untransposed_beta = terminal_topos.right_kan_untranspose(functor, colors, presheaf, right_beta)

    assert terminal_topos.right_kan_untranspose(functor, colors, presheaf, right_transposed_alpha).components == (
        right_alpha.components
    )
    assert terminal_topos.right_kan_transpose(functor, colors, presheaf, right_untransposed_beta).components == (
        right_beta.components
    )
    assert terminal_topos.right_kan_unit(functor, colors).source is colors
    assert terminal_topos.right_kan_counit(functor, presheaf).target is presheaf
    assert terminal_topos.validate_right_kan_adjunction(functor, colors, presheaf) is True
