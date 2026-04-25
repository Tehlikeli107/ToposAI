import torch

from topos_ai.generation import ToposConstrainedDecoder


def test_topological_constrained_decoding_masks_disallowed_tokens():
    """
    Reachability constraints should override raw logits for disallowed tokens.

    This verifies masking behavior only; it is not a general hallucination
    guarantee for language models.
    """
    vocab_size = 5
    reachability = torch.zeros(vocab_size, vocab_size)
    reachability[0, 1] = 1.0
    reachability[1, 2] = 1.0
    reachability[0, 2] = 1.0

    current_idx = 0
    raw_logits = torch.tensor([-5.0, 10.0, 5.0, 2.0, -10.0])
    assert torch.argmax(raw_logits).item() == 1

    strict_reachability = torch.zeros_like(reachability)
    strict_reachability[0, 2] = 1.0
    strict_decoder = ToposConstrainedDecoder(strict_reachability, threshold=0.5)

    safe_prediction = strict_decoder.generate_safe_token(current_idx, raw_logits, temperature=0.0)
    assert safe_prediction == 2

    masked_logits = strict_decoder.apply_topological_mask(current_idx, raw_logits)
    assert masked_logits[1] == float("-inf")
    assert masked_logits[2] == 5.0
