import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import TopologicalLinear, TopologicalNorm


class ToposMambaBlock(nn.Module):
    """
    Categorical/state-space inspired recurrent block.

    The block scans a sequence with a bounded state tensor. Its Python loop is
    linear in sequence length, but it is not an optimized Mamba kernel and it
    does not provide literal unbounded context. Long-range behavior is limited
    by the learned state size, numerical saturation, and training.
    """

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A = nn.Parameter(torch.rand(d_model, d_state))
        self.B = TopologicalLinear(d_model, d_state, bias=False)
        self.C = TopologicalLinear(d_state, d_model, bias=False)

        self.norm = TopologicalNorm(d_model)

    def forward(self, x, state=None):
        """
        Args:
            x: Tensor with shape [batch, seq_len, d_model].
            state: Optional previous state [batch, d_model, d_state].

        Returns:
            A pair of (outputs, next_state).
        """
        batch, seq_len, _ = x.shape

        if state is None:
            h = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        if seq_len == 0:
            return x, h

        outputs = []
        A_gate = torch.sigmoid(self.A).unsqueeze(0)

        for t in range(seq_len):
            x_t = x[:, t, :]
            input_morphism = self.B(x_t)
            input_morphism = input_morphism.unsqueeze(1).expand(-1, self.d_model, -1)
            input_morphism = input_morphism * x_t.unsqueeze(-1)

            h = torch.clamp((h * A_gate) + input_morphism, min=0.0, max=1.0)

            C_weights = torch.sigmoid(self.C.weight_raw)
            y_t = torch.sum(h * C_weights.unsqueeze(0), dim=-1)
            outputs.append(self.norm(y_t).unsqueeze(1))

        y_final = torch.cat(outputs, dim=1)
        return torch.clamp(y_final + x, min=0.0, max=1.0), h


class ToposMambaLM(nn.Module):
    """
    Small language-model wrapper around ToposMambaBlock.

    This is a research prototype for streaming-state language modeling. It is
    attention-free and scans tokens linearly in this implementation, but claims
    about speed, quality, or effective context length must be benchmarked.
    """

    def __init__(self, vocab_size, d_model=128, d_state=32, num_layers=4):
        super().__init__()
        from .nn import YonedaEmbedding

        self.yoneda_emb = YonedaEmbedding(vocab_size)
        self.yoneda_proj = TopologicalLinear(vocab_size, d_model, bias=False)

        self.blocks = nn.ModuleList(
            [ToposMambaBlock(d_model=d_model, d_state=d_state) for _ in range(num_layers)]
        )
        self.norm = TopologicalNorm(d_model)

    def forward(self, idx, states=None):
        yoneda_repr = self.yoneda_emb(idx)
        x = self.yoneda_proj(yoneda_repr)

        new_states = []
        for i, block in enumerate(self.blocks):
            state_i = states[i] if states is not None else None
            x, new_h = block(x, state=state_i)
            new_states.append(new_h)

        x_norm = self.norm(x)
        vocab_embeddings = torch.sigmoid(self.yoneda_proj.weight_raw)

        x_normalized = F.normalize(x_norm, p=2, dim=-1)
        vocab_normalized = F.normalize(vocab_embeddings, p=2, dim=0)
        cosine_sim = torch.matmul(x_normalized, vocab_normalized)

        reachability_logits = (cosine_sim + 1.0) / 2.0
        reachability_logits = torch.clamp(reachability_logits, min=1e-6, max=1.0 - 1e-6)
        return reachability_logits, new_states
