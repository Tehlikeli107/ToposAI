import torch
import triton
import triton.language as tl
import time

# =====================================================================
# FLASHTOPOS v2.0: 2D REDUCTION OPTİMİZASYONLU KERNEL
# 3D Tensor yaratmadan (Compiler Bug'larını aşarak) SRAM içerisinde 
# kusursuz O(N^2) Lukasiewicz mantık hesabı yapar.
# =====================================================================

@triton.jit
def flash_topos_truth_kernel_2d(
    q_ptr, k_ptr, out_ptr,
    M, N, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    batch_id = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Akümülatör matrisi: Doğrudan 2D (BLOCK_M x BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Batch adreslerini hesapla
    q_batch_base = q_ptr + batch_id * stride_qb
    k_batch_base = k_ptr + batch_id * stride_kb

    # Optimizasyon: D ekseni (Feature) boyunca tek tek veya bloklar halinde ilerle.
    # 3D matris yayınlamak (broadcasting) yerine 2D Vektör Dış Çarpımı (Outer Product) mantığı kullan.
    for d in range(D):
        q_ptrs = q_batch_base + offs_m * stride_qm + d * stride_qd
        k_ptrs = k_batch_base + offs_n * stride_kn + d * stride_kd

        # Q'dan 1 boyutlu sütun, K'dan 1 boyutlu satır alıyoruz
        q_val = tl.load(q_ptrs, mask=mask_m, other=0.0) # [BLOCK_M]
        k_val = tl.load(k_ptrs, mask=mask_n, other=0.0) # [BLOCK_N]

        # Sadece 2D üzerinde işlem yapıyoruz (SRAM dostu, anında derlenir)
        impl = 1.0 - q_val[:, None] + k_val[None, :]
        impl = tl.minimum(impl, 1.0)
        
        acc += impl

    # Feature boyutuna bölerek ortalama (Conjunction) al
    acc = acc / D

    # Çıktı VRAM'ine kaydet
    out_ptrs = out_ptr + batch_id * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def flash_topos_attention(q, k):
    B, M, D = q.shape
    _, N, _ = k.shape
    
    out = torch.empty((B, M, N), device=q.device, dtype=torch.float32)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    flash_topos_truth_kernel_2d[grid](
        q, k, out,
        M, N, D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out

def pytorch_topos_attention(q, k):
    q_exp = q.unsqueeze(2) 
    k_exp = k.unsqueeze(1) 
    impl = torch.clamp(1.0 - q_exp + k_exp, max=1.0)
    truth = impl.mean(dim=-1) 
    return truth

def run_extreme_long_context_test():
    if not torch.cuda.is_available():
        return
        
    device = torch.device('cuda')
    
    B, SeqLen, Dim = 1, 8192, 64
    
    print("--- FLASHTOPOS v2.0: UZUN BAĞLAM (LONG-CONTEXT) STRES TESTİ ---")
    print(f"Token Sayısı: {SeqLen} (Yaklaşık 15 sayfalık döküman)")
    print(f"PyTorch için arkaplanda gereken geçici RAM: ~16.1 GB\n")
    
    Q = torch.rand((B, SeqLen, Dim), device=device, dtype=torch.float32)
    K = torch.rand((B, SeqLen, Dim), device=device, dtype=torch.float32)
    
    print("[1] FlashTopos 2D Kernel (Triton SRAM İşlemi) Başlıyor...")
    start = time.perf_counter()
    # Isınma (Derleme)
    out_triton = flash_topos_attention(Q, K)
    torch.cuda.synchronize()
    
    # Asıl Hız Testi
    start = time.perf_counter()
    for _ in range(5):
        out_triton = flash_topos_attention(Q, K)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 5 * 1000
    print(f"    BAŞARILI! Ortalama Süre: {triton_time:.2f} ms")
    print(f"    Üretilen Topos Dikkat Matrisi: {out_triton.shape}\n")

    print("[2] Standart PyTorch Topos Katmanı Başlıyor...")
    try:
        out_torch = pytorch_topos_attention(Q, K)
        torch.cuda.synchronize()
        print("    PYTORCH BAŞARILI. (OOM almadı, 16 GB VRAM'e sığdı.)")
        
        diff = torch.max(torch.abs(out_triton - out_torch)).item()
        print(f"    Matematiksel Doğruluk Farkı (Max Diff): {diff:.6f}")
        
    except torch.cuda.OutOfMemoryError:
        print("    SONUÇ: PYTORCH ÇÖKTÜ! (Out of Memory - VRAM Yetmedi)")
        print("    Normal sistemler 8192 kelimelik mantıksal zincirde hafıza yetersizliğinden çuvalladı.")
        print(f"    FlashTopos ise bunu 0 ekstra hafıza ile {triton_time:.2f} ms'de geçti!")

if __name__ == "__main__":
    run_extreme_long_context_test()
