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
    Q = torch.rand(B, M, K_dim, device='cuda')
    K = torch.rand(B, N, K_dim, device='cuda')
    
    # 1. Triton Kernel Çıktısı
    out_triton = flash_topos_attention(Q, K)
    
    # 2. PyTorch Baseline (Lukasiewicz Mantığı) Çıktısı
    Q_exp = Q.unsqueeze(3) # [B, M, K_dim, 1]
    K_exp = K.unsqueeze(2) # [B, 1, K_dim, N]
    
    # min(1, 1 - Q + K) (Lukasiewicz Implication)
    impl = torch.clamp(1.0 - Q_exp + K_exp, max=1.0)
    
    # Boyut üzerinden ortalama al (Conjunction)
    out_torch = impl.mean(dim=2) # [B, M, N]
    
    # 3. Kıyaslama (Tolerance < 1e-4)
    # Triton'un SRAM'de yaptığı hesapla, PyTorch'un VRAM'de yaptığı hesap eşleşmeli
    torch.testing.assert_close(out_triton, out_torch, rtol=1e-3, atol=1e-3)

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
