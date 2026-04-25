import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from topos_ai.nn import MultiUniverseToposAttention, precompute_freqs_cis


PERFECT_RECALL_THRESHOLD = 0.9999
STRONG_RECALL_THRESHOLD = 0.99


def run_needle_in_haystack(seq_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print(f"[UYARI] Cihaz {device}. Uzun baglam benchmarki icin CUDA tavsiye edilir.")
        seq_len = min(seq_len, 2048)

    print(f"\n--- [TEST]: {seq_len:,} token Needle-in-a-Haystack ---")

    dim = 64
    num_universes = 4
    haystack_x = torch.randn(1, seq_len, dim, device=device)

    needle_position = int(seq_len * 0.73)
    secret_key = torch.ones(dim, device=device) * 5.0
    haystack_x[0, needle_position, :] = secret_key

    query_pos = seq_len - 1
    haystack_x[0, query_pos, :] = secret_key

    topos_attn = MultiUniverseToposAttention(
        d_model=dim,
        num_universes=num_universes,
    ).to(device)
    freqs_cis = precompute_freqs_cis(dim // num_universes, seq_len * 2)[:seq_len].to(device)

    haystack_x = torch.sigmoid(haystack_x)
    secret_key = torch.sigmoid(secret_key)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        output, _ = topos_attn(haystack_x, freqs_cis=freqs_cis)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    query_result = output[0, query_pos, :]
    similarity = torch.nn.functional.cosine_similarity(
        query_result.unsqueeze(0),
        secret_key.unsqueeze(0),
    ).item()

    time_taken = (t1 - t0) * 1000

    print(f"  > Context size : {seq_len:,} tokens")
    print(f"  > Needle pos   : {needle_position:,}")
    print(f"  > Latency      : {time_taken:.2f} ms")
    print(f"  > Recall       : %{similarity * 100:.4f}")

    if similarity >= PERFECT_RECALL_THRESHOLD:
        print("  SONUC: PERFECT RECALL.")
    elif similarity >= STRONG_RECALL_THRESHOLD:
        print("  SONUC: GUCLU AMA İDEALİZE DEGIL.")
    else:
        print("  SONUC: BASARISIZ / LOST-IN-THE-MIDDLE RISKI.")

    return similarity, time_taken


def run_benchmark(context_sizes=None):
    if context_sizes is None:
        context_sizes = [4096, 8192, 16384, 32768]

    print("=========================================================================")
    print(" NEEDLE-IN-A-HAYSTACK BENCHMARK")
    print(" Bu benchmark, tek bir gizli vektoru uzun baglam icinden geri cagirma")
    print(" davranisini olcer. Sonuc donanim sinirlariyla birlikte okunmalidir.")
    print("=========================================================================\n")

    results = []
    for size in context_sizes:
        try:
            acc, t = run_needle_in_haystack(size)
            results.append((size, acc, t))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print(f"  SONUC: OOM. {size:,} token denemesi atlandi; daha buyuk boyutlar denenmeyecek.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc

                gc.collect()
                results.append((size, "OOM", "OOM"))
                break
            raise

    print("\n--- NIHAI BILANCO (TOPOSAI LONG-CONTEXT PERFORMANCE) ---")
    print(f"{'Context':<15} | {'Recall':<20} | {'Latency':<15}")
    print("-" * 55)
    for size, acc, t in results:
        if acc == "OOM":
            print(f"{size:<15,} | {'OOM':<20} | {'-':<15}")
        else:
            print(f"{size:<15,} | %{acc * 100:<19.4f} | {t:.2f} ms")

    numeric_results = [(size, acc, t) for size, acc, t in results if acc != "OOM"]
    best = max(numeric_results, key=lambda item: item[1], default=None)

    print("\n[BILIMSEL DEGERLENDIRME]")
    if best is None:
        print("Bu kosuda basarili bir needle olcumu uretilemedi; tum denemeler OOM oldu.")
    else:
        size, acc, _ = best
        print(f"En iyi olculen recall: %{acc * 100:.4f} @ {size:,} token.")
        if acc >= PERFECT_RECALL_THRESHOLD:
            print("Calistigi baglam araliginda perfect-recall esigini gecti.")
        elif acc >= STRONG_RECALL_THRESHOLD:
            print("Guclu recall var, fakat perfect-recall iddiasi icin yeterli degil.")
        else:
            print("Perfect-recall iddiasi desteklenmedi; sonuc basarisiz/karisik okunmali.")
    print("OOM satirlari algoritmik basari degil, donanim siniri olarak raporlanir.")


if __name__ == "__main__":
    run_benchmark()
