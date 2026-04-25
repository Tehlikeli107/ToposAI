import torch

import topos_ai
from topos_ai.adjoint import FreeForgetfulAdjoint, LinearAdjoint
from topos_ai.distributed import setup_distributed_topos
from topos_ai.elementary_topos import ElementaryTopos
from topos_ai.formal_category import FiniteCategory
from topos_ai.generation import ToposConstrainedDecoder
from topos_ai.hott import FinitePathGroupoid, HomotopyEquivalence, PathFamily
from topos_ai.infinity_categories import (
    FiniteHorn,
    FiniteSimplicialSet,
    HodgeLaplacianEngine,
    InfinityCategoryLayer,
    SimplicialComplexBuilder,
    nerve_2_skeleton,
    nerve_3_skeleton,
)
from topos_ai.kan import KanAdjunction, NeuralKanExtension
from topos_ai.lawvere_tierney import LawvereTierneyTopology
from topos_ai.logic import HeytingNeuralLayer, SubobjectClassifier
from topos_ai.mamba import ToposMambaBlock
from topos_ai.monad import GiryMonad
from topos_ai.motives import UniversalMotiveEngine
from topos_ai.optics import Lens, VanLaarhovenLens
from topos_ai.polynomial_functors import DynamicalSystem, PolynomialFunctor, WiringDiagram
from topos_ai.quantum_logic import QuantumLogicGate
from topos_ai.rl_killer import TopologicalPlanner
from topos_ai.tame_geometry import OMinimalProjector, TameNeuralLayer
from topos_ai.verification import Lean4VerificationBridge


def test_linear_adjoint_instantiates_and_round_trips():
    adjoint = LinearAdjoint(dim_in=3, dim_out=2)
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)

    assert adjoint.W.shape == (2, 3)
    assert adjoint.unit(x).shape == x.shape
    assert adjoint.counit(y).shape == y.shape

    unit_loss, counit_loss = adjoint.triangle_loss(x, y)
    assert torch.isfinite(unit_loss)
    assert torch.isfinite(counit_loss)


def test_elementary_topos_and_logic_proxies_are_bounded():
    topo = ElementaryTopos(dim=3)
    A = torch.tensor([0.2, 0.8, 1.0])
    B = torch.tensor([0.5, 0.4, 1.0])

    torch.testing.assert_close(topo.product(A, B), torch.minimum(A, B))
    torch.testing.assert_close(topo.coproduct(A, B), torch.maximum(A, B))
    torch.testing.assert_close(topo.exponential(torch.tensor([0.51]), torch.tensor([0.5])), torch.tensor([0.5]))
    assert topo.check_morphism(torch.tensor([0.1, 0.2]), torch.tensor([0.2, 0.2])) is True
    assert topo.check_morphism(torch.tensor([0.3, 0.2]), torch.tensor([0.2, 0.2])) is False

    omega = SubobjectClassifier()
    implication = omega.implies(A, B)
    negation = omega.logical_not(A)
    expected_implication = torch.where(A <= B, torch.ones_like(B), B)
    expected_negation = torch.where(A <= 0.0, torch.ones_like(A), torch.zeros_like(A))
    torch.testing.assert_close(implication, expected_implication)
    torch.testing.assert_close(negation, expected_negation)
    assert torch.isfinite(implication).all()
    assert torch.isfinite(negation).all()
    assert torch.all((negation >= 0.0) & (negation <= 1.0))

    layer = HeytingNeuralLayer(in_features=3, out_features=2)
    out = layer(torch.randn(4, 3))
    assert out.shape == (4, 2)
    assert torch.all((out >= 0.0) & (out <= 1.0))


def test_homotopy_equivalence_aligns_rotated_point_cloud():
    aligner = HomotopyEquivalence()
    space_A = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    translation = torch.tensor([2.0, -1.0])
    space_B = space_A @ rotation.T + translation

    R, t = aligner.find_homotopy_path(space_A, space_B)
    transported = aligner.transport_along_path(space_A, R, t)

    torch.testing.assert_close(transported, space_B, atol=1e-5, rtol=1e-5)

    try:
        aligner.find_homotopy_path(torch.randn(2, 3), torch.randn(2, 4))
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected mismatched spaces to raise ValueError.")


def test_finite_path_groupoid_models_identity_types_and_transport():
    paths = FinitePathGroupoid(
        objects=("A", "B"),
        paths={
            "idA": ("A", "A"),
            "idB": ("B", "B"),
            "p": ("A", "B"),
            "p_inv": ("B", "A"),
        },
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

    assert paths.identity_type("A", "B") == frozenset({"p"})
    assert paths.refl("A") == "idA"
    assert paths.inverse("p") == "p_inv"
    assert paths.compose("p_inv", "p") == "idA"

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

    assert family.transport("p", 1) == "one"
    assert family.transport(paths.compose("p_inv", "p"), 0) == 0
    assert family.validate_functorial_transport() is True
    assert family.validate_transport_equivalences() is True
    forward, backward = family.transport_equivalence("p")
    assert forward[0] == "zero"
    assert backward["one"] == 1


def test_distributed_setup_returns_model_when_fsdp_unavailable_or_uninitialized():
    model = torch.nn.Linear(2, 2)
    wrapped = setup_distributed_topos(model, rank=0, world_size=1)
    assert isinstance(wrapped, torch.nn.Module)


def test_free_forgetful_triangle_loss_uses_native_spaces():
    adjoint = FreeForgetfulAdjoint(vocab_size=5, embed_dim=3)
    idx = torch.tensor([1, 2])
    emb = torch.randn(2, 3)

    unit_loss, counit_loss = adjoint.triangle_loss(idx, emb)
    assert torch.isfinite(unit_loss)
    assert torch.isfinite(counit_loss)


def test_giry_join_marginalizes_distribution_of_distributions():
    monad = GiryMonad(dim=3)
    weighted_inner = torch.tensor(
        [[[0.10, 0.20, 0.00], [0.00, 0.30, 0.40]]],
        dtype=torch.float32,
    )

    out = monad.join(weighted_inner)
    expected = weighted_inner.sum(dim=-2)
    expected = expected / expected.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(1))


def test_quantum_logic_meet_join_return_projections():
    q_logic = QuantumLogicGate()
    ket0 = torch.tensor([[1.0], [0.0]])
    ket_plus = torch.tensor([[1.0], [1.0]]) / (2**0.5)
    P = ket0 @ ket0.T
    Q = ket_plus @ ket_plus.T

    meet = q_logic.quantum_and(P, Q)
    join = q_logic.quantum_or(P, Q)

    for projection in (meet, join):
        torch.testing.assert_close(projection, projection.T, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(projection @ projection, projection, atol=1e-6, rtol=1e-6)

    torch.testing.assert_close(meet, torch.zeros_like(P), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(join, torch.eye(2), atol=1e-6, rtol=1e-6)

    sequential = q_logic.sequential_and(P, Q)
    sequential_reversed = q_logic.sequential_and(Q, P)
    assert not torch.allclose(sequential, sequential_reversed)


def test_kan_extensions_return_expected_shapes():
    source = torch.randn(6, 3)
    target = torch.randn(2, 4)

    left, right, restriction = KanAdjunction(dim_c=3, dim_e=4, dim_d=5)(source, target)
    assert left.shape == (2, 5)
    assert right.shape == (2, 5)
    assert restriction.shape == (2, 3)

    learned = NeuralKanExtension(num_source=6, dim_c=3, dim_e=4, dim_d=5)
    assert learned(target).shape == (2, 5)


def test_van_laarhoven_lens_composition_forwards():
    first = VanLaarhovenLens(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(8, 4),
    )
    second = VanLaarhovenLens(
        torch.nn.Linear(4, 3),
        torch.nn.Linear(7, 2),
    )

    composed = second.compose(first)
    assert composed(torch.randn(5, 4)).shape == (5, 2)

    lens = Lens.linear(dim_s=4, dim_a=2)
    assert torch.isfinite(lens.lens_laws_loss(torch.randn(5, 4)))


def test_polynomial_functor_and_wiring_diagram_shapes():
    poly = PolynomialFunctor(positions=3, directions_per_pos=[1, 2, 4])
    assert poly.apply(torch.randn(2, 4)).shape == (2, 3)

    systems = [
        DynamicalSystem(state_dim=4, input_dim=3, output_dim=3),
        DynamicalSystem(state_dim=5, input_dim=3, output_dim=2),
    ]
    wiring = WiringDiagram(systems)
    outputs, new_states = wiring(
        states=[torch.randn(2, 4), torch.randn(2, 5)],
        external_input=torch.randn(2, 3),
    )

    assert outputs[-1].shape == (2, 2)
    assert new_states[0].shape == (2, 4)
    assert new_states[1].shape == (2, 5)


def test_topos_mamba_block_shapes_and_state_reuse():
    block = ToposMambaBlock(d_model=4, d_state=3)
    x = torch.rand(2, 5, 4)

    y, state = block(x)
    assert y.shape == x.shape
    assert state.shape == (2, 4, 3)

    y_next, state_next = block(torch.rand(2, 2, 4), state=state)
    assert y_next.shape == (2, 2, 4)
    assert state_next.shape == state.shape


def test_topos_mamba_block_accepts_empty_sequence():
    block = ToposMambaBlock(d_model=4, d_state=3)
    x = torch.rand(2, 0, 4)

    y, state = block(x)

    assert y.shape == x.shape
    assert state.shape == (2, 4, 3)


def test_hodge_proxy_layer_returns_expected_shapes():
    points = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    builder = SimplicialComplexBuilder(epsilon=2.0)
    edges, triangles, edge_to_idx = builder.build_complex(points)

    assert len(edges) == 3
    assert triangles == [(0, 1, 2)]

    engine = HodgeLaplacianEngine(num_nodes=3, edges=edges, triangles=triangles, edge_to_idx=edge_to_idx)
    L0, L1 = engine.get_laplacians()
    assert L0.shape == (3, 3)
    assert L1.shape == (3, 3)

    layer = InfinityCategoryLayer(node_dim=2, edge_dim=2, out_dim=4)
    H0_new, H1_new = layer(torch.randn(3, 2), torch.randn(3, 2), L0, L1)
    assert H0_new.shape == (3, 4)
    assert H1_new.shape == (3, 4)


def test_finite_simplicial_set_checks_inner_horn_fillers():
    category = FiniteCategory(
        objects=("0", "1", "2"),
        morphisms={
            "id0": ("0", "0"),
            "id1": ("1", "1"),
            "id2": ("2", "2"),
            "f": ("0", "1"),
            "g": ("1", "2"),
            "h": ("0", "2"),
        },
        identities={"0": "id0", "1": "id1", "2": "id2"},
        composition={
            ("id0", "id0"): "id0",
            ("id1", "id1"): "id1",
            ("id2", "id2"): "id2",
            ("f", "id0"): "f",
            ("id1", "f"): "f",
            ("g", "id1"): "g",
            ("id2", "g"): "g",
            ("h", "id0"): "h",
            ("id2", "h"): "h",
            ("g", "f"): "h",
        },
    )

    nerve = nerve_2_skeleton(category)
    horn = FiniteHorn(
        dimension=2,
        missing_face=1,
        faces={
            0: ("mor", "g"),
            2: ("mor", "f"),
        },
    )

    assert nerve.validate_face_identities() is True
    assert nerve.horn_fillers(horn) == (("comp", "g", "f"),)
    assert nerve.is_inner_kan(max_dimension=2) is True
    assert nerve.has_unique_inner_horn_fillers(max_dimension=2) is True

    broken = FiniteSimplicialSet(
        simplices={
            0: (("obj", "0"), ("obj", "1"), ("obj", "2")),
            1: (("mor", "f"), ("mor", "g")),
            2: (),
        },
        faces={
            (1, ("mor", "f"), 0): ("obj", "1"),
            (1, ("mor", "f"), 1): ("obj", "0"),
            (1, ("mor", "g"), 0): ("obj", "2"),
            (1, ("mor", "g"), 1): ("obj", "1"),
        },
    )

    assert broken.horn_fillers(horn) == ()
    assert broken.is_inner_kan(max_dimension=2) is False


def test_nerve_3_skeleton_encodes_associativity_coherence():
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
            if middle != after_src:
                continue
            composition[(after, before)] = identities[src] if src == dst else f"m{src}{dst}"

    category = FiniteCategory(
        objects=objects,
        morphisms=morphisms,
        identities=identities,
        composition=composition,
    )

    nerve = nerve_3_skeleton(category)
    simplex = ("assoc", "m23", "m12", "m01")
    horn = nerve.horn_from_simplex(3, simplex, missing_face=1)

    assert nerve.face(3, simplex, 1) == ("comp", "m23", "m02")
    assert nerve.face(3, simplex, 2) == ("comp", "m13", "m01")
    assert nerve.horn_fillers(horn) == (simplex,)
    assert nerve.validate_degeneracy_identities() is True
    assert nerve.is_inner_kan(max_dimension=3) is True
    assert nerve.has_unique_inner_horn_fillers(max_dimension=3) is True


def test_tame_geometry_layers_are_finite_and_bounded():
    projector = OMinimalProjector(degree=3)
    x = torch.linspace(-100.0, 100.0, steps=21)
    y = projector(x)

    assert torch.isfinite(y).all()
    assert torch.max(torch.abs(y)) <= 2.0

    layer = TameNeuralLayer(in_features=3, out_features=2)
    assert torch.isfinite(layer(torch.randn(4, 3))).all()


def test_reachability_decoder_masks_and_falls_back_safely():
    reachability = torch.tensor(
        [
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 0.0],
            [0.7, 0.1, 0.9],
        ]
    )
    logits = torch.tensor([1.0, 2.0, 3.0])
    decoder = ToposConstrainedDecoder(reachability, threshold=0.5)

    masked = decoder.apply_topological_mask(0, logits)
    assert torch.isneginf(masked[0])
    assert masked[1] == logits[1]
    assert torch.isneginf(masked[2])

    fallback = decoder.apply_topological_mask(1, logits)
    torch.testing.assert_close(fallback, logits)

    assert decoder.generate_safe_token(0, logits, temperature=0.0) == 1


def test_reachability_decoder_rejects_invalid_top_k():
    decoder = ToposConstrainedDecoder(torch.eye(3), threshold=0.5)

    for top_k in (0, -1):
        try:
            decoder.generate_safe_token(0, torch.ones(3), top_k=top_k)
        except ValueError as exc:
            assert "top_k" in str(exc)
        else:
            raise AssertionError("Expected top_k validation to reject non-positive values.")


def test_planner_reports_residual_for_unreachable_goals():
    planner = TopologicalPlanner(state_dim=2, action_dim=1)
    planner.transition_matrix = torch.zeros(2, 2)
    planner.control_matrix = torch.tensor([[1.0], [0.0]])
    start = torch.zeros(2)

    reachable_action, reachable_residual = planner.topos_contravariant_pullback(
        start,
        torch.tensor([2.0, 0.0]),
    )
    torch.testing.assert_close(reachable_action, torch.tensor([2.0]))
    torch.testing.assert_close(reachable_residual, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    _, unreachable_residual = planner.topos_contravariant_pullback(
        start,
        torch.tensor([0.0, 1.0]),
    )
    assert unreachable_residual > 0.9


def test_verification_bridge_generates_ascii_lean_chain():
    bridge = Lean4VerificationBridge(["1-A", "B!"])
    script = bridge._generate_lean_code([0, 1], confidence=0.75)

    assert "topos_proof_v_1_a_to_b" in script
    assert "v_1_a -> b" in script
    assert "Confidence Score: 0.7500" in script
    assert bridge._generate_lean_code([0], confidence=0.0).startswith("Error:")


def test_verification_bridge_generates_defeater_chain():
    bridge = Lean4VerificationBridge(["healthy", "immune", "mutation", "cancer"])
    script = bridge._generate_lean_code([0, 1, 2, 3], confidence=0.5, has_defeater=True)

    assert "topos_proof_healthy_to_Not_cancer" in script
    assert "(h3 : mutation -> Not cancer)" in script
    assert ": healthy -> Not cancer := by" in script


def test_motive_engine_outputs_finite_mmd_and_latents():
    engine = UniversalMotiveEngine(dim_A=3, dim_B=4, motive_dim=2)
    loss, M_A, M_B = engine.topological_mmd_loss(torch.randn(5, 3), torch.randn(5, 4))

    assert torch.isfinite(loss)
    assert M_A.shape == (5, 2)
    assert M_B.shape == (5, 2)


def test_lawvere_tierney_axiom_checker_reports_finite_residuals():
    topology = LawvereTierneyTopology()
    P = torch.tensor([0.0, 0.2, 1.0])
    Q = torch.tensor([1.0, 0.4, 0.0])
    C = torch.tensor([0.5, 0.5, 0.5])

    residuals = topology.check_axioms(P, Q, C, topology.double_negation_topology)
    assert len(residuals) == 3
    assert all(torch.isfinite(torch.tensor(value)) for value in residuals)


def test_package_exports_core_math_modules():
    for name in (
        "quantum_logic",
        "formal_category",
        "yoneda",
        "infinity_categories",
        "lawvere_tierney",
        "tame_geometry",
        "mamba",
        "motives",
        "rl_killer",
    ):
        assert name in topos_ai.__all__


def test_free_category_generator_basic_path():
    from topos_ai.lazy.free_category import FreeCategoryGenerator

    gen = FreeCategoryGenerator()
    gen.add_morphism("f", "A", "B")
    gen.add_morphism("g", "B", "C")
    gen.add_morphism("h", "C", "D")

    # Identity path
    assert gen.find_morphism_path_lazy("A", "A") == "id_A"

    # Direct single-step
    assert gen.find_morphism_path_lazy("A", "B") == "f"

    # Multi-step composition (right-to-left notation)
    path = gen.find_morphism_path_lazy("A", "C")
    assert path == "g o f"

    # Three-hop
    path = gen.find_morphism_path_lazy("A", "D")
    assert path == "h o g o f"

    # Disconnected returns None
    gen2 = FreeCategoryGenerator()
    gen2.add_morphism("x", "X", "Y")
    assert gen2.find_morphism_path_lazy("X", "Z") is None


def test_free_category_generator_exception_sieve():
    from topos_ai.lazy.free_category import FreeCategoryGenerator

    gen = FreeCategoryGenerator()
    gen.add_morphism("flies", "Bird", "Sky")
    gen.add_morphism("swims", "Bird", "Water")

    # Exception sieve blocks a specific route
    result = gen.find_morphism_path_lazy("Bird", "Sky", exceptions=[("Bird", "Sky")])
    assert result is None

    # Without exception, route exists
    assert gen.find_morphism_path_lazy("Bird", "Sky") == "flies"


def test_free_category_generator_duplicate_edge_ignored():
    from topos_ai.lazy.free_category import FreeCategoryGenerator

    gen = FreeCategoryGenerator()
    gen.add_morphism("f1", "A", "B")
    gen.add_morphism("f2", "A", "B")  # duplicate target — should be ignored

    # Only one edge from A to B
    assert len(gen.generators.get("A", [])) == 1
