import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from topos_ai.yoneda import YonedaReconstructor, YonedaUniverse


def run_yoneda_lemma_experiment():
    print("=========================================================================")
    print(" YONEDA-INSPIRED PROBE-DISTANCE RECONSTRUCTION")
    print(" This is a relation-vector reconstruction demo, not a proof of Yoneda.")
    print("=========================================================================\n")

    torch.manual_seed(42)

    try:
        from sklearn.datasets import load_digits
    except ImportError:
        print("scikit-learn bulunamadi; deney atlandi.")
        return

    digits = load_digits()
    img_index = 8
    true_X_np = digits.data[img_index : img_index + 1] / 16.0
    true_X = torch.tensor(true_X_np, dtype=torch.float32)
    dim = 64

    num_probes = 200
    print(f"[DATA] sklearn digits hedefi: {digits.target[img_index]}")
    print(f"[SETUP] {num_probes} random probe, {dim} boyut.")

    universe = YonedaUniverse(num_probes, dim)
    with torch.no_grad():
        true_morphisms = universe.get_morphisms(true_X)

    reconstructor = YonedaReconstructor(num_probes, dim)
    optimizer = torch.optim.Adam(reconstructor.parameters(), lr=0.1)

    epochs = 4000
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, _ = reconstructor(true_morphisms, universe)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == 1:
            print(f"  [Epoch {epoch:<4}] probe-distance loss: {loss.item():.6f}")

    elapsed = time.time() - t0
    final_reconstructed_X = reconstructor.estimated_X.detach()
    absolute_error = torch.mean(torch.abs(true_X - final_reconstructed_X)).item()

    print("\n--- RECONSTRUCTION RESULT ---")
    print(f"  > mean absolute pixel error: {absolute_error:.6f}")
    print(f"  > elapsed: {elapsed:.2f}s")

    pixels = final_reconstructed_X.view(8, 8).numpy()
    print("-" * 20)
    for row in range(8):
        line = ""
        for col in range(8):
            val = pixels[row, col]
            if val > 0.7:
                char = "##"
            elif val > 0.4:
                char = "**"
            elif val > 0.1:
                char = ".."
            else:
                char = "  "
            line += char
        print(line)
    print("-" * 20)

    print("\n[INTERPRETATION]")
    if absolute_error < 1e-2:
        print("Probe-distance reconstruction succeeded on this small image.")
    else:
        print("Probe-distance reconstruction did not reach the target threshold.")
    print("This supports the toy reconstruction method; it does not prove the categorical Yoneda lemma.")


if __name__ == "__main__":
    run_yoneda_lemma_experiment()
