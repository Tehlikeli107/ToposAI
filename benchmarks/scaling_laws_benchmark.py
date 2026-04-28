import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import gc
from topos_ai.kernels import flash_topos_attention

# =====================================================================
# HARDWARE SCALING LAWS BENCHMARK (VRAM ÖLÇÜMÜ)
# İddia: Standart PyTorch (O(N^2)) uzun bağlamlarda VRAM'i patlatır.
# FlashTopos (Triton) uses blockwise SRAM computation to reduce intermediate
# memory. The materialized output matrix still scales with sequence length.
# =====================================================================

def get_vram_mb():
    """GPU'nun o an kullandığı maksimum belleği MB cinsinden döndürür."""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)

def run_scaling_benchmark(seq_lengths=None):
    if not torch.cuda.is_available():
        print("CUDA bulunamadı. Bu donanım testi GPU gerektirir.")
        return

    print("--- 1. ARAŞTIRMA DEMOSU: DONANIM ÖLÇEKLENME YASALARI (SCALING LAWS) ---")
    print("PyTorch vs FlashTopos (Triton) VRAM Tüketim Analizi\n")

    # Test edilecek Bağlam Uzunlukları (Context Length - N)
    # 1K, 2K, 4K, 8K, 16K, 32K (Çok uzun bir kitap)
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]
    dim = 64
    batch_size = 1

    print(f"{'Bağlam (N)':<12} | {'PyTorch VRAM (MB)':<20} | {'FlashTopos VRAM (MB)':<20} | {'Durum'}")
    print("-" * 75)

    for N in seq_lengths:
        # GPU'yu temizle
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        # ---------------------------------------------------------
        # 1. PYTORCH STANDART HESAPLAMA (OOM Bekleniyor)
        # ---------------------------------------------------------
        torch_vram = "OOM (Çöktü)"
        torch_status = "Bşrsz"
        try:
            Q = torch.rand((batch_size, N, dim), device='cuda', dtype=torch.float32)
            K = torch.rand((batch_size, N, dim), device='cuda', dtype=torch.float32)
            
            torch.cuda.reset_peak_memory_stats()
            base_vram = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # PyTorch'un arkaplanda yaratacağı O(N^2 * D) devasa matris:
            Q_exp = Q.unsqueeze(2) 
            K_exp = K.unsqueeze(1) 
            impl = torch.clamp(1.0 - Q_exp + K_exp, max=1.0)
            _ = impl.mean(dim=-1)
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch_vram = f"{peak_vram - base_vram:.1f}"
            torch_status = "Geçti"
            
            # Matrisi bellekten sil
            del Q_exp, K_exp, impl, Q, K
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pass # Beklenen çöküş
            else:
                torch_vram = "HATA"

        # GPU'yu tekrar temizle
        torch.cuda.empty_cache()
        gc.collect()

        # ---------------------------------------------------------
        # 2. FLASHTOPOS (TRITON SRAM KERNEL) HESAPLAMASI
        # ---------------------------------------------------------
        topos_vram = "HATA"
        topos_status = "Bşrsz"
        try:
            Q = torch.rand((batch_size, N, dim), device='cuda', dtype=torch.float32)
            K = torch.rand((batch_size, N, dim), device='cuda', dtype=torch.float32)
            
            torch.cuda.reset_peak_memory_stats()
            base_vram = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # FlashTopos, işlemi SRAM'de 64x64 bloklarla yapar
            _ = flash_topos_attention(Q, K)
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            topos_vram = f"{max(0.0, peak_vram - base_vram):.1f}"
            topos_status = "Geçti"
            
            del Q, K
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                topos_vram = "OOM (Çöktü)"
            else:
                topos_vram = "HATA"
        except Exception as e:
            topos_vram = "HATA"

        print(f"{N:<12} | {torch_vram:<20} | {topos_vram:<20} | Topos: {topos_status}")

if __name__ == "__main__":
    run_scaling_benchmark()
