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

def _format_error_detail(exc):
    """Hatanın sınıfını ve kısa mesajını tek satırda döndürür."""
    return f"{exc.__class__.__name__}: {repr(exc)}"


def run_scaling_benchmark(seq_lengths=None, include_error_detail=False, verbose=False):
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

    header = f"{'Bağlam (N)':<12} | {'PyTorch VRAM (MB)':<20} | {'FlashTopos VRAM (MB)':<20} | {'Durum'}"
    if include_error_detail:
        header += " | error_detail"
    print(header)
    print("-" * 75)

    summary = {
        "passed": 0,
        "oom": 0,
        "unexpected_error": 0,
    }

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
        torch_error_detail = ""
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
                summary["oom"] += 1
                torch_error_detail = _format_error_detail(e)
            else:
                torch_vram = "HATA"
                summary["unexpected_error"] += 1
                torch_error_detail = _format_error_detail(e)
        except Exception as e:
            torch_vram = "HATA"
            summary["unexpected_error"] += 1
            torch_error_detail = _format_error_detail(e)
        else:
            summary["passed"] += 1

        # GPU'yu tekrar temizle
        torch.cuda.empty_cache()
        gc.collect()

        # ---------------------------------------------------------
        # 2. FLASHTOPOS (TRITON SRAM KERNEL) HESAPLAMASI
        # ---------------------------------------------------------
        topos_vram = "HATA"
        topos_status = "Bşrsz"
        topos_error_detail = ""
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
            summary["passed"] += 1
            
            del Q, K
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                topos_vram = "OOM (Çöktü)"
                summary["oom"] += 1
                topos_error_detail = _format_error_detail(e)
            else:
                topos_vram = "HATA"
                summary["unexpected_error"] += 1
                topos_error_detail = _format_error_detail(e)
        except Exception as e:
            topos_vram = "HATA"
            summary["unexpected_error"] += 1
            topos_error_detail = _format_error_detail(e)

        row = f"{N:<12} | {torch_vram:<20} | {topos_vram:<20} | Topos: {topos_status}"
        error_detail = "; ".join(
            detail for detail in [
                f"PyTorch={torch_error_detail}" if torch_error_detail else "",
                f"Topos={topos_error_detail}" if topos_error_detail else "",
            ] if detail
        )

        if include_error_detail:
            row += f" | {error_detail}"

        print(row)

        if verbose and error_detail:
            print(f"  ↳ error_detail: {error_detail}")

    total_cases = len(seq_lengths) * 2
    print("\nÖzet İstatistikler")
    print("-" * 30)
    print(f"Toplam case: {total_cases}")
    print(f"Geçti: {summary['passed']}")
    print(f"OOM: {summary['oom']}")
    print(f"Beklenmeyen hata: {summary['unexpected_error']}")

if __name__ == "__main__":
    run_scaling_benchmark()
