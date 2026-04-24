import torch
import pytest
from topos_ai.kernels import flash_topos_attention

# --- Kategori Teorisi Triton Kernel Testleri ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kerneli CUDA gerektirir.")
def test_flash_topos_attention_matches_pytorch_baseline():
    """
    FlashTopos (Triton) kernelinin, standart PyTorch tensör işlemiyle
    matematiksel olarak %100 aynı (yakın) sonucu verip vermediğini test eder.
    """
    # Matris boyutları
    B, M, K_dim, N = 2, 64, 32, 64
    
    # Kuantize (0-1 arası) mantıksal girdiler
    Q = torch.rand(B, M, K_dim, device='cuda', requires_grad=True)
    K = torch.rand(B, N, K_dim, device='cuda', requires_grad=True)
    
    Q_ref = Q.clone().detach().requires_grad_(True)
    K_ref = K.clone().detach().requires_grad_(True)
    
    # 1. Triton Kernel Çıktısı
    out_triton = flash_topos_attention(Q, K)
    loss_triton = out_triton.sum()
    loss_triton.backward()
    
    # 2. PyTorch Baseline (Lukasiewicz Mantığı) Çıktısı
    Q_exp = Q_ref.unsqueeze(2) # [B, M, 1, K_dim]
    K_exp = K_ref.unsqueeze(1) # [B, 1, N, K_dim]
    
    # min(1, 1 - Q + K) (Lukasiewicz Implication)
    impl = torch.clamp(1.0 - Q_exp + K_exp, min=0.0, max=1.0)
    
    # Boyut (K_dim) üzerinden ortalama al (Conjunction)
    out_torch = impl.mean(dim=-1) # [B, M, N]
    loss_torch = out_torch.sum()
    loss_torch.backward()
    
    # 3. Kıyaslama (Tolerance < 1e-3)
    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)
    
    # 4. Backward Pass Kıyaslaması (OOM-free Gradient check)
    # Sınır değerlerdeki (0.0 ve 1.0) PyTorch Clamp vs Triton Mask alt-türev (subgradient)
    # farklılıkları nedeniyle toleransı 0.05 yapıyoruz (1/32 boyutundan kaynaklı).
    torch.testing.assert_close(Q.grad, Q_ref.grad, rtol=0.05, atol=0.05)
    torch.testing.assert_close(K.grad, K_ref.grad, rtol=0.05, atol=0.05)

def test_sheaf_gluing_consistency():
    """Sheaf (Demet) birleştirme kuralının mantıksal sınırlarını test eder."""
    from topos_ai.math import sheaf_gluing
    
    # Çelişmeyen Matrisler
    A = torch.tensor([[0.9, 0.5], [0.1, 0.9]])
    B = torch.tensor([[0.9, 0.6], [0.2, 0.9]])
    
    can_glue, global_truth = sheaf_gluing(A, B, threshold=0.3)
    assert can_glue is True, "Çelişmeyen evrenler yapıştırılabilmelidir."
    
    # Çelişen Matrisler (Biri 0.9, diğeri 0.1 diyor ve eminseler)
    C = torch.tensor([[0.9, 0.5], [0.1, 0.9]])
    D = torch.tensor([[0.1, 0.5], [0.1, 0.9]])
    
    can_glue, global_truth = sheaf_gluing(C, D, threshold=0.1)
    assert can_glue is False, "Çelişen evrenlerin (Sheaf Violation) yapıştırılması reddedilmelidir."

def test_core_math_logic_cpu():
    """
    GPU/Triton olmayan CI ortamlarında (Örn: Github Actions) ToposAI'ın çekirdek
    mantığının (Geçişlilik ve Syllogism) kırılıp kırılmadığını test eder.
    """
    from topos_ai.math import lukasiewicz_composition, transitive_closure
    
    # 3x3 bir Evren: 0->1 ve 1->2 okları (Morfizmaları) mevcut.
    R = torch.zeros((3, 3))
    R[0, 1] = 1.0
    R[1, 2] = 1.0
    
    # Doğrudan Lukasiewicz Composition Testi (R * R = R^2)
    # A->B ve B->C ise, R^2'de A->C = 1.0 olmalıdır.
    R_comp = lukasiewicz_composition(R, R)
    assert R_comp[0, 2].item() == 1.0, "Lukasiewicz T-Norm geçişliliği (A->C) sağlamalıdır."
    assert R_comp[0, 1].item() == 0.0, "Geçişlilik matrisi R^2, sadece 2. derece okları göstermelidir."
    
    # Transitive Closure (Sonsuz Geçişlilik) Testi
    # R_inf, hem doğrudan okları (R) hem de zincirleme okları (R^2, R^3) içermelidir.
    R_inf = transitive_closure(R, max_steps=2)
    assert R_inf[0, 1].item() == 1.0, "Transitive Closure orijinal R'yi (A->B) korumalıdır."
    assert R_inf[1, 2].item() == 1.0, "Transitive Closure orijinal R'yi (B->C) korumalıdır."
    assert R_inf[0, 2].item() == 1.0, "Transitive Closure türetilmiş oku (A->C) içermelidir."
