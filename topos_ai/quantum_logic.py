import torch
import torch.nn as nn


class QuantumLogicGate(nn.Module):
    """
    Finite-dimensional quantum logic over orthogonal projections.

    `quantum_and` and `quantum_or` implement the projection-lattice
    meet and join. The non-commutative measurement product is exposed
    separately as `sequential_and`.
    """

    def __init__(self, tol: float = 1e-5):
        super().__init__()
        self.tol = tol

    def _validate_projection_pair(
        self, P: torch.Tensor, Q: torch.Tensor
    ) -> None:
        if P.shape != Q.shape or P.shape[-1] != P.shape[-2]:
            raise ValueError("P and Q must be square matrices with matching shapes.")

    def _eye_like(self, P: torch.Tensor) -> torch.Tensor:
        n = P.shape[-1]
        eye = torch.eye(n, dtype=P.dtype, device=P.device)
        return eye.expand(P.shape[:-2] + (n, n))

    def _symmetrize(self, P: torch.Tensor) -> torch.Tensor:
        return 0.5 * (P + P.transpose(-2, -1).conj())

    def _spectral_projector(
        self,
        operator: torch.Tensor,
        keep_mask,
    ) -> torch.Tensor:
        hermitian = self._symmetrize(operator)
        eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
        mask = keep_mask(eigenvalues).to(dtype=eigenvectors.dtype)
        return (eigenvectors * mask.unsqueeze(-2)) @ eigenvectors.transpose(-2, -1).conj()

    def check_commutation(self, P, Q, tol=1e-5):
        """Return whether two observables commute and their commutator."""
        self._validate_projection_pair(P, Q)
        commutator = torch.matmul(P, Q) - torch.matmul(Q, P)
        is_commuting = torch.norm(commutator).item() < tol
        return is_commuting, commutator

    def quantum_and(self, P, Q):
        """
        Projection-lattice meet: the projection onto range(P) intersect range(Q).
        """
        self._validate_projection_pair(P, Q)
        return self._spectral_projector(
            P + Q,
            lambda eig: eig > (2.0 - self.tol),
        )

    def sequential_and(self, P, Q):
        """
        Non-commutative sequential measurement: first P, then Q.

        This is useful for Heisenberg-ordering demos, but it is not the
        lattice meet because the result need not be a projection.
        """
        self._validate_projection_pair(P, Q)
        return torch.matmul(Q, P)

    def quantum_not(self, P):
        """Orthogonal complement of a projection."""
        return self._eye_like(P) - P

    def quantum_or(self, P, Q):
        """
        Projection-lattice join: projection onto span(range(P), range(Q)).
        """
        self._validate_projection_pair(P, Q)
        return self._spectral_projector(
            P + Q,
            lambda eig: eig > self.tol,
        )
