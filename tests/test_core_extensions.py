import subprocess

import numpy as np
import pytest
import torch

from topos_ai.cohomology import CechCohomology
from topos_ai.distributed import setup_expert_parallelism
from topos_ai.logic import StrictGodelImplication, SubobjectClassifier
from topos_ai.math import soft_godel_composition
from topos_ai.monad import ContinuationMonad, GiryMonad, KleisliLayer, WriterMonad
from topos_ai.open_games import ComposedOpenGame, OpenGame
from topos_ai.optics import Lens, Prism, Traversal, VanLaarhovenLens
from topos_ai.reasoning import AutonomousTheoremProver, DefeasibleReasoning
from topos_ai.rl_killer import TopologicalPlanner
from topos_ai.sheaf_dataloader import SheafDataloader
from topos_ai.tokenization import TopologicalTokenizer
from topos_ai.topology import PersistentHomology
from topos_ai.verification import Lean4VerificationBridge
from topos_ai.yoneda import YonedaReconstructor, YonedaUniverse


def test_cech_cohomology_detects_consensus_and_cycle_obstruction():
    cover = CechCohomology(num_nodes=3, edges=[(0, 1), (1, 2)])

    disagreement, beta0 = cover.compute_H0_consensus(torch.tensor([[2.0], [2.0], [2.0]]))
    assert disagreement == 0.0
    assert beta0 == 1

    disagreement, _ = cover.compute_H0_consensus(torch.tensor([[0.0], [1.0], [1.0]]))
    assert disagreement > 0.0

    cycle = CechCohomology(num_nodes=3, edges=[(0, 1), (1, 2), (0, 2)])
    obstruction, beta1, residual = cycle.compute_H1_obstruction(torch.tensor([[1.0], [1.0], [1.0]]))
    assert beta1 == 1
    assert obstruction > 0.0
    assert residual.shape == (3, 1)


def test_persistent_homology_counts_components_cycles_and_filled_triangles():
    homology = PersistentHomology(num_nodes=4)
    disconnected = torch.tensor(
        [
            [0.0, 0.1, 9.0, 9.0],
            [0.1, 0.0, 9.0, 9.0],
            [9.0, 9.0, 0.0, 0.1],
            [9.0, 9.0, 0.1, 0.0],
        ]
    )
    assert homology.calculate_betti(disconnected, threshold=0.5) == (2, 0)

    square = torch.tensor(
        [
            [0.0, 0.1, 9.0, 0.1],
            [0.1, 0.0, 0.1, 9.0],
            [9.0, 0.1, 0.0, 0.1],
            [0.1, 9.0, 0.1, 0.0],
        ]
    )
    assert homology.calculate_betti(square, threshold=0.5) == (1, 1)

    triangle = PersistentHomology(num_nodes=3)
    complete = torch.tensor(
        [
            [0.0, 0.1, 0.1],
            [0.1, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ]
    )
    assert triangle.calculate_betti(complete, threshold=0.5) == (1, 0)


def test_defeasible_reasoning_blocks_defeated_conclusion():
    reasoner = DefeasibleReasoning(num_nodes=3)
    reasoner.add_rule(0, 1, weight=1.0)
    reasoner.add_rule(1, 2, weight=1.0)

    without_defeater = reasoner.deliberate(iterations=2, start_node=0)
    assert without_defeater[0, 2] > 0.9

    reasoner.add_rule(1, 2, weight=1.0, is_defeater=True)
    with_defeater = reasoner.deliberate(iterations=2, start_node=0)
    assert with_defeater[0, 2] == 0.0


def test_autonomous_theorem_prover_reports_new_composed_edge():
    R = torch.zeros(3, 3)
    R[0, 1] = 1.0
    R[1, 2] = 1.0

    closed, new_theorems = AutonomousTheoremProver(R).discover_theorems(iterations=2, threshold=0.75)

    assert closed[0, 2] > 0.75
    assert any((i, j) == (0, 2) for i, j, _step, _score in new_theorems)


def test_topological_tokenizer_trains_round_trips_and_persists(tmp_path, capsys):
    tokenizer = TopologicalTokenizer(vocab_size=6)
    tokenizer.train("abababab ")
    capsys.readouterr()

    encoded = tokenizer.encode("abab ")
    decoded = tokenizer.decode(encoded)
    assert decoded == "abab "
    assert tokenizer.merges

    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    loaded = TopologicalTokenizer()
    loaded.load(str(path))
    assert loaded.encode("abab ") == encoded
    assert loaded.decode(encoded) == decoded


def test_open_game_play_coplay_and_sequential_composition():
    def play_functor(x, params):
        return x * params

    def coplay_functor(_x, _y, reward, params):
        return reward * params, torch.tensor([0.25])

    game = OpenGame("scale", play_functor, coplay_functor, params=torch.tensor([0.5]))
    torch.testing.assert_close(game.play(torch.tensor([4.0])), torch.tensor([2.0]))
    torch.testing.assert_close(game.coplay(torch.tensor([2.0]), lr=1.0), torch.tensor([1.0]))
    torch.testing.assert_close(game.params, torch.tensor([0.75]))

    game_a = OpenGame("a", play_functor, coplay_functor, params=torch.tensor([0.5]))
    game_b = OpenGame("b", play_functor, coplay_functor, params=torch.tensor([0.25]))
    composed = ComposedOpenGame(game_a, game_b)

    torch.testing.assert_close(composed.play(torch.tensor([8.0])), torch.tensor([1.0]))
    torch.testing.assert_close(composed.coplay(torch.tensor([4.0]), lr=0.0), torch.tensor([0.5]))


def test_sheaf_dataloader_streams_memmap_batches_on_cpu(tmp_path):
    file_path = tmp_path / "features.dat"
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    mmap = np.memmap(file_path, dtype="float32", mode="w+", shape=data.shape)
    mmap[:] = data
    mmap.flush()

    loader = SheafDataloader(
        file_path=str(file_path),
        num_samples=3,
        feature_dim=4,
        num_probes=2,
        batch_size=2,
    )

    direct = loader._get_morphism(torch.tensor(data[:2]))
    assert direct.shape == (2, 2)
    assert torch.all((direct >= 0.0) & (direct <= 1.0))

    batches = list(loader.stream_batches(device="cpu"))
    assert [batch.shape for batch in batches] == [torch.Size([2, 2]), torch.Size([1, 2])]
    assert all(batch.device.type == "cpu" for batch in batches)


def test_yoneda_probe_distances_and_reconstructor_gradients():
    universe = YonedaUniverse(num_probes=2, dim=2)
    with torch.no_grad():
        universe.probes.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))

    X = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(universe.get_morphisms(X), expected)

    reconstructor = YonedaReconstructor(num_probes=2, dim=2)
    loss, estimated = reconstructor(expected[:1], universe)
    loss.backward()

    assert torch.isfinite(loss)
    assert estimated.shape == (1, 2)
    assert reconstructor.estimated_X.grad is not None


def test_monad_variants_and_kleisli_layer_are_finite():
    giry = GiryMonad(dim=3)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(giry.T(x).sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(giry.unit(x), torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(giry.markov_compose(torch.eye(2), torch.eye(2)), torch.eye(2))
    assert torch.isfinite(giry.monad_laws_loss(x))

    cont = ContinuationMonad()
    cps = cont.cps_transform(lambda value: value + 1.0, torch.tensor([2.0]))
    torch.testing.assert_close(cps(lambda value: value * 3.0), torch.tensor([9.0]))

    writer = WriterMonad(log_dim=2)
    value, log = writer.unit(torch.ones(1, 3))
    assert value.shape == (1, 3)
    assert log.shape == (1, 2)
    told = writer.tell(value, torch.ones(1, 2))
    bound_value, bound_log = writer.bind(told, lambda inner: (inner + 1.0, torch.full((1, 2), 2.0)))
    torch.testing.assert_close(bound_value, torch.full((1, 3), 2.0))
    torch.testing.assert_close(bound_log, torch.full((1, 2), 3.0))
    joined_value, joined_log = writer.join(((value, torch.ones(1, 2)), torch.full((1, 2), 4.0)))
    torch.testing.assert_close(joined_value, value)
    torch.testing.assert_close(joined_log, torch.full((1, 2), 5.0))

    layer = KleisliLayer(in_features=3, out_features=2)
    mu, sigma = layer(torch.zeros(4, 3))
    assert mu.shape == sigma.shape == (4, 2)
    assert torch.isfinite(layer.sample(torch.zeros(4, 3))).all()
    assert torch.isfinite(layer.kl_divergence(torch.zeros(4, 3)))


def test_prism_and_traversal_shape_contracts():
    prism = Prism.gating(dim_s=4, dim_a=2)
    rebuilt, confidence = prism.review(torch.randn(3, 4))
    assert rebuilt.shape == (3, 4)
    assert confidence.shape == (3, 1)
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))

    traversal = Traversal(dim_s=4, dim_a=2, num_positions=3)
    s = torch.randn(5, 4)
    parts = traversal.get_all(s)
    assert parts.shape == (5, 3, 2)
    assert traversal.put_all(s, parts).shape == (5, 4)
    assert traversal.traverse(s, lambda part: part + 1.0).shape == (5, 4)
    assert traversal(s).shape == (5, 3, 2)


def test_structural_lens_modify_and_van_laarhoven_transform_branch():
    lens = Lens.linear(dim_s=4, dim_a=2)
    s = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    torch.testing.assert_close(lens.modify(s, lambda part: part + 10.0), torch.tensor([[11.0, 12.0, 3.0, 4.0]]))
    torch.testing.assert_close(lens.lens_laws_loss(s), torch.tensor(0.0))

    optic = VanLaarhovenLens(
        torch.nn.Linear(4, 2),
        torch.nn.Linear(6, 3),
    )
    assert optic(torch.randn(2, 4), f=lambda focused: focused * 0.0).shape == (2, 3)


def test_strict_godel_implication_and_composition_backward_paths():
    A = torch.tensor([0.2, 0.8], requires_grad=True)
    B = torch.tensor([0.5, 0.4], requires_grad=True)

    implication = StrictGodelImplication.apply(A, B)
    implication.sum().backward()
    assert A.grad is not None
    assert B.grad is not None

    classifier = SubobjectClassifier()
    torch.testing.assert_close(classifier.logical_or(A.detach(), B.detach()), torch.maximum(A.detach(), B.detach()))

    R1 = torch.eye(3, requires_grad=True)
    R2 = torch.eye(3, requires_grad=True)
    composed = soft_godel_composition(R1, R2, tau=5.0)
    composed.sum().backward()

    assert R1.grad is not None
    assert R2.grad is not None
    assert torch.isfinite(R1.grad).all()
    assert torch.isfinite(R2.grad).all()


def test_planner_random_search_baseline_returns_best_sample():
    torch.manual_seed(0)
    planner = TopologicalPlanner(state_dim=2, action_dim=1)
    planner.transition_matrix = torch.eye(2)
    planner.control_matrix = torch.tensor([[1.0], [0.0]])

    action, residual = planner.reinforcement_learning_simulate(
        torch.zeros(2),
        torch.tensor([0.5, 0.0]),
        num_episodes=10,
    )

    assert action.shape == (1,)
    assert torch.isfinite(residual)


def test_expert_parallelism_records_local_universe_indices():
    class MultiUniverseToposAttention(torch.nn.Module):
        def __init__(self, num_universes):
            super().__init__()
            self.num_universes = num_universes

    model = torch.nn.Sequential(MultiUniverseToposAttention(5))
    setup_expert_parallelism(model, rank=1, world_size=2)
    assert model[0].local_universe_indices == [2, 3, 4]

    too_many_ranks = torch.nn.Sequential(MultiUniverseToposAttention(1))
    setup_expert_parallelism(too_many_ranks, rank=1, world_size=2)
    assert not hasattr(too_many_ranks[0], "local_universe_indices")


def test_verification_bridge_handles_missing_lean(tmp_path, monkeypatch, capsys):
    def missing_lean(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("topos_ai.verification.subprocess.run", missing_lean)
    bridge = Lean4VerificationBridge(["A", "B"])
    filename = tmp_path / "proof.lean"

    status, script, output = bridge.prove_theorem([0, 1], confidence=0.25, filename=str(filename))

    capsys.readouterr()
    assert status is None
    assert output == "Compiler Not Found"
    assert filename.read_text(encoding="utf-8") == script


def test_verification_bridge_reports_lean_failure(tmp_path, monkeypatch, capsys):
    class FailedProcess:
        returncode = 1
        stdout = "stdout"
        stderr = "stderr"

    monkeypatch.setattr("topos_ai.verification.subprocess.run", lambda *_args, **_kwargs: FailedProcess())
    bridge = Lean4VerificationBridge(["A", "B"])

    status, _script, output = bridge.prove_theorem([0, 1], confidence=0.25, filename=str(tmp_path / "bad.lean"))

    capsys.readouterr()
    assert status is False
    assert output == "stdout\nstderr"


def test_verification_bridge_reports_lean_timeout(tmp_path, monkeypatch, capsys):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="lean", timeout=10)

    monkeypatch.setattr("topos_ai.verification.subprocess.run", timeout)
    bridge = Lean4VerificationBridge(["A", "B"])

    status, _script, output = bridge.prove_theorem([0, 1], confidence=0.25, filename=str(tmp_path / "timeout.lean"))

    capsys.readouterr()
    assert status is None
    assert output == "Timeout"
