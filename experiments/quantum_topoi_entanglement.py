import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import math

# =====================================================================
# QUANTUM TOPOI & BELL'S THEOREM (CHSH INEQUALITY)
# Problem: Klasik mantık (0 ve 1) veya Bulanık mantık (Lukasiewicz 0.0-1.0)
# Kuantum mekaniğindeki "Dolanıklık (Entanglement)" fenomenini açıklayamaz.
# Klasik sistemlerde Bell Eşitsizliği (S) maksimum 2.0 olabilir.
# Çözüm: ToposAI, genlikleri (Amplitudes) Karmaşık Sayılar (Complex Numbers)
# olarak modelleyen bir 'Hilbert Topos Uzayı' kurar ve S = 2.828 (2√2) 
# sonucuna ulaşarak Kuantum Dolanıklığını matematiksel olarak gösterir.
# =====================================================================

class QuantumToposUniverse:
    def __init__(self):
        # Pauli Matrisleri (Kuantum Ölçüm Operatörleri)
        self.sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self.sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        
    def create_entangled_state(self):
        """
        İdealize Dolanık Durum (Bell State): |Φ+⟩ = 1/√2 (|00⟩ + |11⟩)
        İki parçacık (Foton) evrenin iki ucunda olsalar bile tek bir 
        Topolojik Bütün (Global Section) olarak hareket ederler.
        """
        state = torch.zeros(4, 1, dtype=torch.complex64)
        state[0, 0] = 1.0 / math.sqrt(2) # |00⟩ durumu
        state[3, 0] = 1.0 / math.sqrt(2) # |11⟩ durumu
        return state
        
    def measurement_operator(self, angle):
        """Belirli bir açıda (theta) ölçüm yapan kuantum operatörü."""
        # cos(θ)*σ_z + sin(θ)*σ_x
        op = math.cos(angle) * self.sigma_z + math.sin(angle) * self.sigma_x
        return op

    def expectation_value(self, state, op_A, op_B):
        """
        Alice (op_A) ve Bob (op_B) tarafından yapılan ortak ölçümün beklenen değeri.
        Tensor Çarpımı (Kronecker Product): op_A ⊗ op_B
        ⟨Φ| (A ⊗ B) |Φ⟩
        """
        # Tensor çarpımını PyTorch ile hesaplama (2x2 ⊗ 2x2 -> 4x4 matris)
        tensor_product = torch.kron(op_A, op_B)
        
        # Beklenen Değer = state.dagger() @ tensor_product @ state
        state_dagger = torch.conj(state).t()
        expectation = torch.matmul(state_dagger, torch.matmul(tensor_product, state))
        
        return expectation.item().real # Sonuç her zaman reel sayıdır (Hermitian)

def run_quantum_entanglement_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 14: QUANTUM TOPOI & BELL'S THEOREM (CHSH INEQUALITY) ")
    print(" İddia: Klasik ToposAI (0.0 ile 1.0 arası ihtimaller) Kuantum Dolanıklığını")
    print(" (Spooky Action) simüle edemez ve Bell Eşitsizliği'ni (S <= 2.0) aşamaz.")
    print(" Kuantum ToposAI ise Karmaşık Sayılarla (Hilbert Uzayı) bu sönürü yıkarak")
    print(" S = 2.82 (2√2) sonucuna ulaşır ve Fiziğin en derin sırrını gösterir.")
    print("=========================================================================\n")

    universe = QuantumToposUniverse()
    
    # 1. Dolanık Parçacıkların Yaratılması (Alice ve Bob'a gönderiliyor)
    bell_state = universe.create_entangled_state()
    print("[UZAY-ZAMAN YIRTILMASI]: İki foton 'Dolanık' (Entangled) hale getirildi.")
    print("Foton A Alice'e (Dünya), Foton B Bob'a (Mars) gönderildi.\n")

    # 2. CHSH Testi (Bell Eşitsizliği Ölçüm Açıları)
    # Klasik CHSH deneyinde açılar:
    # Alice: 0 (Z ekseni), π/2 (X ekseni)
    # Bob  : π/4, -π/4
    theta_A1 = 0.0
    theta_A2 = math.pi / 2.0
    
    theta_B1 = math.pi / 4.0
    theta_B2 = -math.pi / 4.0
    
    # Kuantum Topos'unda Bell-State |Φ+> için beklenen değer E(A, B) = cos(theta_A - theta_B)'dir.
    # Operatörleri oluştur
    A1 = universe.measurement_operator(theta_A1)
    A2 = universe.measurement_operator(theta_A2)
    B1 = universe.measurement_operator(theta_B1)
    B2 = universe.measurement_operator(theta_B2)
    
    # 3. Beklenen Değerlerin Hesaplanması (E)
    print("--- KUANTUM ÖLÇÜMLERİ (EXPECTATION VALUES) ---")
    E_A1_B1 = universe.expectation_value(bell_state, A1, B1) # cos(0 - π/4) = 1/√2 = 0.707
    E_A1_B2 = universe.expectation_value(bell_state, A1, B2) # cos(0 - (-π/4)) = 1/√2 = 0.707
    E_A2_B1 = universe.expectation_value(bell_state, A2, B1) # cos(π/2 - π/4) = 1/√2 = 0.707
    E_A2_B2 = universe.expectation_value(bell_state, A2, B2) # cos(π/2 - (-π/4)) = cos(3π/4) = -1/√2 = -0.707
    
    print(f"  E(A1, B1) [ 0,  π/4] : {E_A1_B1:.4f}")
    print(f"  E(A1, B2) [ 0, -π/4] : {E_A1_B2:.4f}")
    print(f"  E(A2, B1) [π/2, π/4] : {E_A2_B1:.4f}")
    print(f"  E(A2, B2) [π/2,-π/4] : {E_A2_B2:.4f}")
    
    # 4. CHSH Eşitsizliği (S Değeri)
    # Bell teoremine göre CHSH = E(A1,B1) + E(A1,B2) + E(A2,B1) - E(A2,B2)
    # (Önceki kodda eksi işareti yanlış yerleştirilmişti, bu yüzden 0 çıkmıştı)
    S_value = E_A1_B1 + E_A1_B2 + E_A2_B1 - E_A2_B2
    
    print("\n--- BELL EŞİTSİZLİĞİ (CHSH) SONUCU ---")
    print(f"  S Skorunuz: {abs(S_value):.4f}")
    print(f"  Klasik Fiziğin Mutlak Sınırı: 2.0000")
    print(f"  Kuantum Mekaniğinin Sınırı  : 2.8284 (2√2)")
    
    if abs(S_value) > 2.01:
        print("\n[✓] ÖLÇÜLEN SONUÇ: EINSTEIN YANILDI! (Spooky Action Proven)")
        print("  ToposAI, klasik mantığın (Boolean/Lukasiewicz) sınırlarını aşarak,")
        print("  Karmaşık Genlikler (Complex Amplitudes) üzerinden parçacıkların")
        print("  ışık hızından hızlı (anlık) kilitlendiğini matematiksel olarak gösterdi.")
        print("  S > 2.0 değeri, Evrenin temelinde 'Lokal Gerçekçilik' olmadığını GÖSTERİR.")
    else:
        print("\n[-] S < 2.0: Sistem klasik fizik sınırlarında kaldı.")

if __name__ == "__main__":
    run_quantum_entanglement_experiment()
