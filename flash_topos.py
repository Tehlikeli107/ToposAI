import torch
import triton
import triton.language as tl
import time

# =====================================================================
# FLASHTOPOS: UZUN BAĞLAMLI (LONG-CONTEXT) MANTIKSAL DİKKAT KERNELİ
# O(N^2 * D) VRAM gereksinimini O(N^2) seviyesine indirir.
# Devasa (8K - 16K) kelime/token dizilerini çökmeden işlemek için 
# GPU SRAM'ini (L1 Cache) bloklar halinde kullanır.
# =====================================================================

@triton.jit
def flash_topos_truth_kernel(
    q_ptr, k_ptr, out_ptr,
    B, M, N, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Hangi batch ve blokta olduğumuzu bul
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # İşlenecek satır (Q) ve sütun (K) aralıkları
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D) # Feature boyutu (Örn: 64)
    
    # Pointer adreslerini hesapla
    q_ptrs = q_ptr + batch_id * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = k_ptr + batch_id * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    
    # Sınır dışına çıkmamak için maskeler
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_d = offs_d < D
    
    # Q ve K bloklarını SRAM'e (L1 Cache) YÜKLE
    # q: [BLOCK_M, BLOCK_D]
    # k: [BLOCK_N, BLOCK_D]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    
    # --- TOPOS MANTIĞI (LUKASIEWICZ İMPLİKASYONU) ---
    # PyTorch'un OOM verdiği devasa tensör genişletmesi burada SADECE SRAM'de yapılır!
    # q[:, None, :] -> [BLOCK_M, 1, BLOCK_D]
    # k[None, :, :] -> [1, BLOCK_N, BLOCK_D]
    
    # 1.0 - Q + K
    impl = 1.0 - q[:, None, :] + k[None, :, :]
    
    # clamp max 1.0 (Topos Truth kuralı)
    impl = tl.minimum(impl, 1.0)
    
    # Sınır dışı boyutları sıfırla ki ortalamayı bozmasın
    impl = tl.where(mask_d[None, None, :], impl, 0.0)
    
    # Conjunction (Tüm featurelar üzerinden mantıksal doğruluk ortalaması)
    # [BLOCK_M, BLOCK_N, BLOCK_D] tensörünü D ekseninde topla
    truth = tl.sum(impl, axis=2) / D # -> [BLOCK_M, BLOCK_N]
    
    # Çıktı matrisine (Attention Matrisi) YAZ
    out_ptrs = out_ptr + batch_id * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, truth, mask=mask_m[:, None] & mask_n[None, :])


def flash_topos_attention(q, k):
    # Girdiler [Batch, SeqLen, Dim]
    B, M, D = q.shape
    _, N, _ = k.shape
    
    # Çıktı matrisi: Her kelimenin diğer kelimeye olan Mantıksal Doğruluğu [Batch, SeqQ, SeqK]
    out = torch.empty((B, M, N), device=q.device, dtype=torch.float32)
    
    # SRAM kapasitesine göre blok boyutları
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(D) # Dim boyutunu 2'nin üssüne yuvarla (Örn: 64)
    
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    flash_topos_truth_kernel[grid](
        q, k, out,
        B, M, N, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )
    
    return out

def pytorch_topos_attention(q, k):
    # Standart PyTorch Kodu (OOM Garantili!)
    q_exp = q.unsqueeze(2) # [B, M, 1, D]
    k_exp = k.unsqueeze(1) # [B, 1, N, D]
    impl = torch.clamp(1.0 - q_exp + k_exp, max=1.0)
    truth = impl.mean(dim=-1) # [B, M, N]
    return truth

def run_extreme_long_context_test():
    if not torch.cuda.is_available():
        return
        
    device = torch.device('cuda')
    
    # DEvasa Bir Bağlam (Örn: 8.192 Kelimelik Uzun Bir Belge/Kitap)
    B = 1
    SeqLen = 8192
    Dim = 64
    
    print(f"--- FLASHTOPOS: UZUN BAĞLAM (LONG-CONTEXT) STRES TESTİ ---")
    print(f"Token Sayısı: {SeqLen} (Yaklaşık 12-15 sayfalık bir döküman)")
    print(f"PyTorch için gereken Geçici VRAM: ~{B * SeqLen * SeqLen * Dim * 4 / (1024**3):.2f} GB\n")
    
    # Tokenları (Değerleri 0-1 arasında sigmoid'den geçmiş varsayıyoruz)
    Q = torch.rand((B, SeqLen, Dim), device=device, dtype=torch.float32)
    K = torch.rand((B, SeqLen, Dim), device=device, dtype=torch.float32)
    
    # 1. FLASHTOPOS KERNEL
    print("[1] FlashTopos Kernel Başlıyor (GPU SRAM İşlemi)...")
    try:
        start = time.perf_counter()
        out_triton = flash_topos_attention(Q, K)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) * 1000
        print(f"    BAŞARILI! Süre: {triton_time:.2f} ms")
        print(f"    Üretilen Attention Matrisi: {out_triton.shape}")
    except Exception as e:
        print(f"    HATA: {e}")

    # 2. STANDART PYTORCH
    print("\n[2] Standart PyTorch Topos Katmanı Başlıyor (VRAM Genişlemesi)...")
    try:
        start = time.perf_counter()
        out_torch = pytorch_topos_attention(Q, K)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) * 1000
        print(f"    BAŞARILI! Süre: {torch_time:.2f} ms")
    except torch.cuda.OutOfMemoryError:
        print("    SONUÇ: ÇÖKTÜ! (Out of Memory) - VRAM bu devasa işlemi taşıyamadı.")
        print("    Klasik PyTorch kodu, Kategori Teorisi mantığını Uzun Metinlerde KULLANAMAZ.")
        
if __name__ == "__main__":
    run_extreme_long_context_test()
