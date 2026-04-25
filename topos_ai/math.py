import torch


class StrictGodelComposition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R1, R2, tau=10.0):
        ctx.save_for_backward(R1, R2)
        ctx.tau = tau
        return godel_composition(R1, R2)

    @staticmethod
    def backward(ctx, grad_output):
        R1, R2 = ctx.saved_tensors
        tau = ctx.tau

        with torch.enable_grad():
            R1_soft = R1.detach().requires_grad_(True)
            R2_soft = R2.detach().requires_grad_(True)

            R1_exp = R1_soft.unsqueeze(-1)
            R2_exp = R2_soft.unsqueeze(-3)
            n = R1_soft.shape[-1]
            batch_shape = R1_soft.shape[:-2]

            expanded_R1 = R1_exp.expand(*batch_shape, n, n, n).unsqueeze(-1)
            expanded_R2 = R2_exp.expand(*batch_shape, n, n, n).unsqueeze(-1)

            concat_for_min = torch.cat([-tau * expanded_R1, -tau * expanded_R2], dim=-1)
            soft_t_norm = -(1.0 / tau) * torch.logsumexp(concat_for_min, dim=-1)
            soft_composition = (1.0 / tau) * torch.logsumexp(tau * soft_t_norm, dim=-2)

            soft_composition.backward(grad_output)

        return R1_soft.grad, R2_soft.grad, None


def godel_composition(R1, R2):
    """
    Exact relation composition for the Goedel-Heyting algebra on [0, 1].

    This is max_k min(R1[i, k], R2[k, j]), the standard composition for
    fuzzy/preorder-style relations enriched over the min t-norm.
    """
    R1_exp = R1.unsqueeze(-1)
    R2_exp = R2.unsqueeze(-3)
    t_norm = torch.minimum(R1_exp, R2_exp)
    composition, _ = torch.max(t_norm, dim=-2)
    return composition


def soft_godel_composition(R1, R2, tau=10.0):
    """
    Exact Goedel composition in the forward pass with a smooth backward path.

    The forward value preserves the categorical max-min relation algebra. The
    custom backward uses a log-sum-exp relaxation so neural experiments still
    receive useful gradients.
    """
    return StrictGodelComposition.apply(R1, R2, tau)


def lukasiewicz_composition(R1, R2):
    """Optional Lukasiewicz relation composition for fuzzy-logic comparisons."""
    R1_exp = R1.unsqueeze(-1)
    R2_exp = R2.unsqueeze(-3)
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0)
    composition, _ = torch.max(t_norm, dim=-2)
    return composition


def transitive_closure(R, max_steps=5, composition="godel"):
    """Compute reachability closure; default is Goedel-Heyting max-min composition."""
    if composition == "godel":
        compose = godel_composition
    elif composition == "lukasiewicz":
        compose = lukasiewicz_composition
    else:
        raise ValueError("composition must be 'godel' or 'lukasiewicz'.")

    R_closure = R.clone()
    for _ in range(max_steps):
        new_R = compose(R_closure, R)
        R_closure = torch.max(R_closure, new_R)
    return R_closure


def sheaf_gluing(truth_A, truth_B, threshold=0.2):
    """
    Check a numeric sheaf-style compatibility condition.

    Sections glue when their overlap disagreement is below the chosen tolerance.
    The glued section uses the local join/max operation.
    """
    disagreement = torch.abs(truth_A - truth_B)
    max_conflict = torch.max(disagreement).item()

    if max_conflict > threshold:
        return False, None
    return True, torch.max(truth_A, truth_B)
