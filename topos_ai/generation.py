import torch
import torch.nn.functional as F


class ToposConstrainedDecoder:
    """
    Reachability-constrained decoding helper.

    The decoder masks next-token logits using a provided reachability matrix.
    This can enforce a user-supplied graph of allowed token transitions, but it
    is not a general mathematical guarantee against hallucination.
    """

    def __init__(self, reachability_matrix, threshold=0.1):
        """
        Args:
            reachability_matrix: Square [vocab_size, vocab_size] tensor.
            threshold: Minimum reachability score required to keep a token.
        """
        assert reachability_matrix.dim() == 2, "Reachability matrix must be 2D."
        assert reachability_matrix.size(0) == reachability_matrix.size(1), (
            "Reachability matrix must be square."
        )

        self.reachability_matrix = reachability_matrix
        self.threshold = threshold

    def apply_topological_mask(self, current_token_idx, next_token_logits):
        """
        Mask logits whose reachability score falls below the threshold.

        If no token is reachable, the original logits are returned as a safe
        fallback to avoid all -inf probabilities.
        """
        logical_connections = self.reachability_matrix[current_token_idx]
        valid_logical_mask = logical_connections >= self.threshold

        if not valid_logical_mask.any():
            return next_token_logits.clone()

        masked_logits = next_token_logits.clone()
        masked_logits[~valid_logical_mask] = float("-inf")
        return masked_logits

    def generate_safe_token(self, current_token_idx, next_token_logits, temperature=1.0, top_k=None):
        """
        Sample a token after reachability masking.

        With very small temperatures the method uses greedy argmax decoding.
        """
        safe_logits = self.apply_topological_mask(current_token_idx, next_token_logits)

        if temperature < 1e-4:
            return torch.argmax(safe_logits).item()

        safe_logits = safe_logits / temperature

        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be a positive integer when provided.")
            k = min(int(top_k), safe_logits.numel())
            indices_to_remove = safe_logits < torch.topk(safe_logits, k)[0][..., -1, None]
            safe_logits[indices_to_remove] = float("-inf")

        probs = F.softmax(safe_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
