import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from topos_ai.quantum_logic import QuantumLogicGate


def create_projection_matrix(eigenvector):
    """Create a rank-1 orthogonal projection P = |v><v|."""
    v = torch.tensor(eigenvector, dtype=torch.float32).unsqueeze(1)
    v = v / torch.norm(v)
    return v @ v.T


def run_quantum_topoi_experiment():
    print("=========================================================================")
    print(" QUANTUM PROJECTIONS AND NON-COMMUTATIVE SEQUENTIAL MEASUREMENT")
    print(" Projection-lattice meet/join is kept separate from ordered measurement.")
    print(" The demo below shows that measuring P then Q can differ from Q then P.")
    print("=========================================================================\n")

    q_logic = QuantumLogicGate()

    print("--- 1. COMMUTING / ORTHOGONAL CASE ---")
    P_A = create_projection_matrix([1.0, 0.0])
    P_B = create_projection_matrix([0.0, 1.0])

    is_comm_classic, _ = q_logic.check_commutation(P_A, P_B)
    print(f"  > P_A and P_B commute? {is_comm_classic}")

    A_then_B = q_logic.sequential_and(P_A, P_B)
    B_then_A = q_logic.sequential_and(P_B, P_A)
    diff_classic = torch.norm(A_then_B - B_then_A).item()
    print(f"  > ||A then B - B then A||: {diff_classic:.6f}")

    meet = q_logic.quantum_and(P_A, P_B)
    join = q_logic.quantum_or(P_A, P_B)
    print(f"  > meet idempotent error: {torch.norm(meet @ meet - meet).item():.6f}")
    print(f"  > join idempotent error: {torch.norm(join @ join - join).item():.6f}")

    print("\n--- 2. NON-COMMUTING SEQUENTIAL MEASUREMENT ---")
    P_Z = create_projection_matrix([1.0, 0.0])
    P_X = create_projection_matrix([0.7071, 0.7071])

    is_comm_quant, commutator = q_logic.check_commutation(P_Z, P_X)
    print(f"  > P_Z and P_X commute? {is_comm_quant}")
    print(f"  > commutator [Z, X]:\n{commutator.numpy()}")

    Z_then_X = q_logic.sequential_and(P_Z, P_X)
    X_then_Z = q_logic.sequential_and(P_X, P_Z)
    diff_quant = torch.norm(Z_then_X - X_then_Z).item()
    print(f"  > ||Z then X - X then Z||: {diff_quant:.6f}")

    lattice_meet = q_logic.quantum_and(P_Z, P_X)
    lattice_join = q_logic.quantum_or(P_Z, P_X)
    print(f"  > lattice meet projection error: {torch.norm(lattice_meet @ lattice_meet - lattice_meet).item():.6f}")
    print(f"  > lattice join projection error: {torch.norm(lattice_join @ lattice_join - lattice_join).item():.6f}")
    print("\nSonuc: non-commutativity sirali olcumde gorulur; lattice meet/join projeksiyon olarak kalir.")


if __name__ == "__main__":
    run_quantum_topoi_experiment()
