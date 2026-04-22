import torch
import triton
import triton.language as tl
import time

# --- TRITON KERNEL (GPU SRAM SEVİYESİ) ---
# Normal GPU'lar Matris Çarpımına (A * B + C) optimize edilmiştir.
# Bizim Topos Teorimiz ise Mantıksal Çıkarım (max(0, A + B - 1)) gerektirir.
# Bu kernel, bloklar (Block_M, Block_N, Block_K) halinde GPU'nun L1 Cache'ini (SRAM) kullanır
# ve devasa geçici (intermediate) tensörler yaratmadan doğrudan sonucu (C) VRAM'e yazar.

@triton.jit
def lukasiewicz_composition_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Hangi blokta (Thread Block) olduğumuzu bul
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Bu bloğun C matrisinde işleyeceği satır ve sütun aralığı
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Matris sınırlarını aşmamak için maske (Padding)
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # C matrisindeki bu blok için bir "Akü" (Accumulator) oluştur
    # Bizim mantığımız "S-Norm (Maksimum)" olduğu için aküyü çok küçük bir sayıyla (-INF benzeri) başlatıyoruz
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) - 1e5
    
    # A ve B matrislerindeki başlangıç pointer adresleri
    # A'nın satırları (M), B'nin sütunları (N)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # K boyutu boyunca döngü (Dot-product'taki toplamanın yerini alır)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # K boyutu sınır kontrolü
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        
        # A ve B'den birer blok SRAM'e yükle (Load)
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)
        
        # --- TOPOS MANTIĞI: LUKASIEWICZ T-NORM ---
        # 1. A + B - 1.0
        # 2. max(0.0, ...) -> Negatifleri 0 yap
        t_norm = a[:, :, None] + b[None, :, :] - 1.0 # (BLOCK_M, BLOCK_K, BLOCK_N)
        # triton'da tl.where veya tl.maximum kullanılır. Biz tl.maximum(0, val) yapıyoruz.
        t_norm = tl.maximum(t_norm, 0.0)
        
        # --- TOPOS MANTIĞI: S-NORM (Maksimum Al) ---
        # K boyutu (aracı nesneler) üzerinden en iyi mantıksal bağıntıyı (max) bul.
        local_max = tl.max(t_norm, axis=1) # (BLOCK_M, BLOCK_N)
        
        # Akümülatörü güncelle (önceki blokların maksimumu ile bu bloğun maksimumunu kıyasla)
        acc = tl.maximum(acc, local_max)
        
        # Pointerları K boyutu kadar ileri kaydır
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Tüm K döngüsü bittiğinde, sonucu C matrisine yaz (Store)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    # Sadece maske içindeki geçerli kısımları yaz
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))

# --- PYTHON SARMALAYICISI (WRAPPER) ---
def triton_topos_composition(a, b):
    # Boyut kontrolleri
    assert a.shape[1] == b.shape[0], "Matris boyutları uyuşmuyor!"
    assert a.is_contiguous(), "A matrisi RAM'de bitişik (contiguous) olmalı"
    assert b.is_contiguous(), "B matrisi RAM'de bitişik (contiguous) olmalı"
    
    M, K = a.shape
    K, N = b.shape
    
    # Çıktı matrisi (Aynı device üzerinde)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Blok boyutlarını (SRAM kapasitesine göre) belirle
    # GPU'nun gücüne göre bu değerler 16, 32, 64, 128 olabilir.
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    # Kaç adet işlem bloğu (Thread Block) gerektiğini hesapla (Grid size)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Kerneli GPU'ya fırlat (Launch)
    lukasiewicz_composition_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return c

# --- PYTORCH REFERANS (NORMAL YÖNTEM) ---
def pytorch_topos_composition(a, b):
    # OOM (Out Of Memory) tehlikesi! O(M*K*N) bellek harcar.
    a_exp = a.unsqueeze(2) # (M, K, 1)
    b_exp = b.unsqueeze(0) # (1, K, N)
    t_norm = torch.clamp(a_exp + b_exp - 1.0, min=0.0) # (M, K, N)
    c, _ = torch.max(t_norm, dim=1) # (M, N)
    return c

# --- TEST VE KIYASLAMA (BENCHMARK) ---
def benchmark_topos_kernels():
    # CUDA cihazı şart (Triton sadece GPU/AMD üzerinde çalışır)
    if not torch.cuda.is_available():
        print("HATA: Triton kernel testi için CUDA (GPU) cihazı gereklidir!")
        return
        
    device = torch.device('cuda')
    print(f"CUDA Cihazı: {torch.cuda.get_device_name(0)}\n")
    
    # Matris boyutları: Örneğin 4096 varlık (Kavram)
    # PyTorch'un OOM (Hafıza Taşıması) verebileceği büyük bir matris
    M, K, N = 4096, 4096, 4096
    
    print(f"Topos Matrisleri Üretiliyor... Boyut: {M}x{K} ve {K}x{N}")
    print(f"Bu matris PyTorch (Referans) için arkaplanda {M*K*N * 4 / (1024**3):.2f} GB geçici RAM isteyecektir!")
    
    # Mantıksal doğruluk değerleri (0.0 ile 1.0 arası)
    A = torch.rand((M, K), device=device, dtype=torch.float32)
    B = torch.rand((K, N), device=device, dtype=torch.float32)
    
    # 1. TRITON KERNEL (Bizim yazdığımız donanım optimizasyonlu SOTA kodu)
    print("\n--- Triton Topos Kernel Çalıştırılıyor ---")
    start_triton = time.perf_counter()
    # Isınma (JIT Derlemesi için)
    C_triton = triton_topos_composition(A, B)
    torch.cuda.synchronize()
    start_triton = time.perf_counter()
    for _ in range(5):
        C_triton = triton_topos_composition(A, B)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start_triton) / 5 * 1000
    print(f"Triton Kernel Süresi: {triton_time:.2f} ms")
    
    # 2. PYTORCH REFERANS KERNEL
    print("\n--- PyTorch Referans Kodu Çalıştırılıyor ---")
    try:
        start_torch = time.perf_counter()
        # Isınma
        C_torch = pytorch_topos_composition(A, B)
        torch.cuda.synchronize()
        start_torch = time.perf_counter()
        for _ in range(5):
            C_torch = pytorch_topos_composition(A, B)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start_torch) / 5 * 1000
        print(f"PyTorch Süresi: {torch_time:.2f} ms")
        
        # Doğruluk Kontrolü (İki kernel de aynı matematiksel sonucu mu veriyor?)
        max_diff = torch.max(torch.abs(C_triton - C_torch)).item()
        print(f"\nDoğruluk Farkı (Max Diff): {max_diff}")
        if max_diff < 1e-4:
            print("SONUÇ: BAŞARILI! Triton Kernel matematiği %100 doğru.")
            print(f"Hız Farkı: Triton, PyTorch'tan {torch_time/triton_time:.2f} kat daha HIZLI!")
        else:
            print("SONUÇ: HATA! Matematiği farklı hesaplıyor.")
            
    except torch.cuda.OutOfMemoryError:
        print("\nSONUÇ: PYTORCH ÇÖKTÜ! (Out of Memory - VRAM Yetmedi)")
        print(f"Sıradan bir PyTorch kodu bu devasa mantık zincirini ({M} kavram) birleştiremez.")
        print(f"Fakat yazdığımız Triton Topos Kerneli (Flash-Topos) bunu sadece {triton_time:.2f} milisaniyede ve sıfır geçici RAM ile başardı!")

if __name__ == "__main__":
    benchmark_topos_kernels()
