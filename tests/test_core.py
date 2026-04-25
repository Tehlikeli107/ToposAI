import pytest
import torch

from topos_ai.kernels import flash_topos_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA.")
def test_flash_topos_attention_matches_pytorch_baseline():
    """FlashTopos should match the PyTorch reference within tolerance."""
    batch, seq_q, dim, seq_k = 2, 64, 32, 64

    Q = torch.rand(batch, seq_q, dim, device="cuda", requires_grad=True)
    K = torch.rand(batch, seq_k, dim, device="cuda", requires_grad=True)

    Q_ref = Q.clone().detach().requires_grad_(True)
    K_ref = K.clone().detach().requires_grad_(True)

    out_triton = flash_topos_attention(Q, K)
    out_triton.sum().backward()

    Q_exp = Q_ref.unsqueeze(2)
    K_exp = K_ref.unsqueeze(1)
    impl = torch.where(Q_exp <= K_exp, torch.ones_like(K_exp), K_exp)
    out_torch = impl.mean(dim=-1)
    out_torch.sum().backward()

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)
    q_ref_grad = torch.zeros_like(Q_ref) if Q_ref.grad is None else Q_ref.grad
    torch.testing.assert_close(Q.grad, q_ref_grad, rtol=0.05, atol=0.05)
    torch.testing.assert_close(K.grad, K_ref.grad, rtol=0.05, atol=0.05)


def test_sheaf_gluing_consistency():
    """Compatible local sections glue; conflicting ones are rejected."""
    from topos_ai.math import sheaf_gluing

    A = torch.tensor([[0.9, 0.5], [0.1, 0.9]])
    B = torch.tensor([[0.9, 0.6], [0.2, 0.9]])

    can_glue, _ = sheaf_gluing(A, B, threshold=0.3)
    assert can_glue is True

    C = torch.tensor([[0.9, 0.5], [0.1, 0.9]])
    D = torch.tensor([[0.1, 0.5], [0.1, 0.9]])

    can_glue, _ = sheaf_gluing(C, D, threshold=0.1)
    assert can_glue is False


def test_flash_topos_attention_uses_godel_heyting_implication_on_cpu():
    """The CPU path should use the same Goedel-Heyting implication as topos logic."""
    Q = torch.tensor([[[0.2, 0.8]]])
    K = torch.tensor([[[0.5, 0.4], [0.1, 0.9]]])

    Q_exp = Q.unsqueeze(2)
    K_exp = K.unsqueeze(1)
    expected = torch.where(Q_exp <= K_exp, torch.ones_like(K_exp), K_exp).mean(dim=-1)

    torch.testing.assert_close(flash_topos_attention(Q, K), expected)


def test_core_math_logic_cpu():
    """CPU-safe checks for composition and transitive closure."""
    from topos_ai.math import godel_composition, lukasiewicz_composition, transitive_closure

    R = torch.zeros((3, 3))
    R[0, 1] = 1.0
    R[1, 2] = 1.0

    R_comp = lukasiewicz_composition(R, R)
    assert R_comp[0, 2].item() == 1.0
    assert R_comp[0, 1].item() == 0.0

    R_inf = transitive_closure(R, max_steps=2)
    assert R_inf[0, 1].item() == 1.0
    assert R_inf[1, 2].item() == 1.0
    assert R_inf[0, 2].item() == 1.0

    fuzzy_R = torch.zeros((3, 3))
    fuzzy_R[0, 1] = 0.8
    fuzzy_R[1, 2] = 0.7
    torch.testing.assert_close(godel_composition(fuzzy_R, fuzzy_R)[0, 2], torch.tensor(0.7))
    torch.testing.assert_close(lukasiewicz_composition(fuzzy_R, fuzzy_R)[0, 2], torch.tensor(0.5))
    fuzzy_closure = transitive_closure(fuzzy_R, max_steps=2)
    torch.testing.assert_close(fuzzy_closure[0, 2], torch.tensor(0.7))

def test_topos_adam_optimizer_vanishing_gradient():
    """
    ToposAdam optimizatörünün [0, 1] sınırlarında Fisher Information
    ile gradyanları doğru bir şekilde (ölmeden) yönlendirdiğini test eder.
    """
    import torch.nn as nn

    from topos_ai.optim import ToposAdam

    model = nn.Linear(10, 1, bias=False)

    # Ağırlıkları -10 gibi çok küçük bir değere çekelim (sigmoid sonrası p ~ 0.000045)
    # Klasik SGD veya Adam burada "Vanishing Gradient" yaşar çünkü p*(1-p) çok küçüktür.
    nn.init.constant_(model.weight, -10.0)

    opt = ToposAdam(model.parameters(), lr=0.1)

    x = torch.ones(1, 10)
    y_true = torch.tensor([[1.0]]) # Hedef 1.0 (Yani weight'in büyümesi lazım)

    # İleri besleme (Sigmoid üzerinden)
    y_pred = torch.sigmoid(model(x))
    loss = (y_pred - y_true)**2
    loss.backward()

    weight_before = model.weight.clone()

    # Optimizasyon adımı
    opt.step()

    # Adımdan sonra ağırlık BÜYÜMÜŞ olmalı.
    # Fisher Information Metric (Natural Gradient) sayesinde ölü gradyan sorunu aşılmış olur.
    assert model.weight[0, 0].item() > weight_before[0, 0].item(), "ToposAdam ölü noktalardan (p~0) çıkabilmelidir."
