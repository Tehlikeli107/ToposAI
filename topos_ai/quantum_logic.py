import torch
import torch.nn as nn

# =====================================================================
# QUANTUM TOPOI & NON-COMMUTATIVE LOGIC (ORTHOMODULAR LATTICES)
# Amacı: Klasik YZ mantığı (Boolean) veya Sezgisel Mantık (Heyting)
# her zaman "Değişmelidir (Commutative)": A AND B == B AND A.
# Ancak Kuantum dünyasında ölçüm sırası sonucu değiştirir (Heisenberg).
# Bu modül, PyTorch üzerinde Kuantum Mantığını (Quantum Logic) inşa eder.
# Değerler (Truth Values) artık skaler sayılar [0,1] değil;
# Hilbert Uzayındaki 'İzdüşüm Operatörleridir (Projection Matrices)'.
# Kuantum mantığında: P AND Q != Q AND P (Eğer P ve Q birbiriyle değişmiyorsa/commute).
# =====================================================================

class QuantumLogicGate(nn.Module):
    """
    Kuantum Toposlarında (Quantum Topoi) Mantık Kapısı.
    A ve B klasik sayılar değil, Karmaşık veya Reel Hermitian (Simetrik) Matrislerdir.
    Her mantıksal önerme, Hilbert uzayındaki bir 'Alt Uzaya (Subspace)' karşılık gelir.
    """
    def __init__(self):
        super().__init__()

    def check_commutation(self, P, Q, tol=1e-5):
        """
        [HEISENBERG BELİRSİZLİK KONTROLÜ]
        İki gözlemlenebilir (Observable) aynı anda ölçülebilir mi?
        [P, Q] = P*Q - Q*P == 0 ise Değişmelidir (Commutative).
        """
        PQ = torch.matmul(P, Q)
        QP = torch.matmul(Q, P)
        commutator = PQ - QP
        is_commuting = torch.norm(commutator).item() < tol
        return is_commuting, commutator

    def quantum_and(self, P, Q):
        """
        [QUANTUM SEQUENTIAL MEASUREMENT (AND)]
        Kuantum mantığında 'P ve ardından Q' (P THEN Q), 
        dalga fonksiyonunun sırayla çökmesidir: Q * P
        (Klasik mantıkta bu P*Q == Q*P'dir, ama Kuantum'da asimetriktir).
        """
        # P ölçümü yapılır, uzay P'ye çöker. Sonra Q ölçümü yapılır.
        return torch.matmul(Q, P)

    def quantum_not(self, P):
        """
        [QUANTUM NEGATION (NOT)]
        Kuantum mantığında bir şeyin değili (Orthogonal Complement),
        Birim Matris eksi İzdüşüm Matrisidir: I - P
        Bu, uzayın geri kalanına izdüşüm yapmaktır.
        """
        I = torch.eye(P.size(0), dtype=P.dtype, device=P.device)
        return I - P

    def quantum_or(self, P, Q):
        """
        [QUANTUM DISJUNCTION (OR)]
        De Morgan Kuralları kuantum mantığında da geçerlidir (İzdüşümler için).
        P OR Q = ~(~P AND ~Q)
        """
        not_P = self.quantum_not(P)
        not_Q = self.quantum_not(Q)
        not_P_and_not_Q = self.quantum_and(not_P, not_Q)
        return self.quantum_not(not_P_and_not_Q)
