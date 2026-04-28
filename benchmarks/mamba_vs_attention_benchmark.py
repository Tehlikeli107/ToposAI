import os
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from topos_ai.mamba import ToposMambaLM
from topos_ai.models import ToposTransformer


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_speed_benchmark(seq_lengths=None):
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]

    print("=========================================================================")
    print(" TOPOS-MAMBA VS TOPOS-TRANSFORMER SPEED BENCHMARK")
    print(" This compares the current Python implementations, not an optimized")
    print(" fused/parallel-scan kernel. Treat theoretical O(N) claims separately.")
    print("=========================================================================\n")

    vocab_size = 100
    d_model = 64
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE]: {device.type.upper()}\n")

    transformer = ToposTransformer(vocab_size, d_model=d_model, num_layers=2).to(device)
    mamba = ToposMambaLM(vocab_size, d_model=d_model, num_layers=2).to(device)

    print(f"{'SeqLen':<12} | {'Transformer':<18} | {'ToposMamba':<18} | {'Observed'}")
    print("-" * 76)

    observed_speedups = []
    for seq in seq_lengths:
        idx = torch.randint(0, vocab_size, (batch_size, seq), device=device)

        try:
            _sync(device)
            t0 = time.time()
            with torch.no_grad():
                _ = transformer(idx)
            _sync(device)
            time_transformer = (time.time() - t0) * 1000
            trans_status = f"{time_transformer:.2f} ms"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                trans_status = "OOM"
                time_transformer = float("inf")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

        try:
            _sync(device)
            t0 = time.time()
            with torch.no_grad():
                _, _ = mamba(idx)
            _sync(device)
            time_mamba = (time.time() - t0) * 1000
            mamba_status = f"{time_mamba:.2f} ms"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mamba_status = "OOM"
                time_mamba = float("inf")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

        if time_transformer != float("inf") and time_mamba != float("inf") and time_mamba > 0:
            speedup = time_transformer / time_mamba
            observed_speedups.append(speedup)
            diff_text = f"{speedup:.2f}x faster" if speedup > 1.0 else f"{(1 / speedup):.2f}x slower"
        elif time_transformer == float("inf") and time_mamba != float("inf"):
            diff_text = "Mamba survived"
        else:
            diff_text = "-"

        print(f"{seq:<12,} | {trans_status:<18} | {mamba_status:<18} | {diff_text}")

    print("\n[OBSERVED RESULT]")
    if observed_speedups and max(observed_speedups) > 1.0:
        print("ToposMamba wins on at least one measured size in this run.")
    else:
        print("This Python ToposMamba implementation did not beat the Transformer in this run.")
    print("A real speedup claim needs an optimized scan/kernel implementation and repeated timing.")


if __name__ == "__main__":
    run_speed_benchmark()
