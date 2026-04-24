import torch


def soft_godel_composition(R1, R2, tau=10.0):
    """
    [DIFFERENTIABLE TOPOS LOGIC]
    Gödel mantığındaki keskin min/max fonksiyonları, Backpropagation (Geri Yayılım)
    sırasında türevleri sıfırlayarak (Dead Gradients) modelin eğitimini durdurur.
    Bunu engellemek için, Sıcaklık (Temperature - tau) parametresi ile çalışan
    'Softmin' ve 'Softmax' (LogSumExp tabanlı) operatörler kullanılır.
    """
    R1_exp = R1.unsqueeze(-1) # [..., N, N, 1]
    R2_exp = R2.unsqueeze(-3) # [..., 1, N, N]

    N = R1.shape[-1]
    batch_shape = R1.shape[:-2]

    # Negatif eksende logsumexp (Softmin yaklaşımı)
    expanded_R1 = R1_exp.expand(*batch_shape, N, N, N).unsqueeze(-1)
    expanded_R2 = R2_exp.expand(*batch_shape, N, N, N).unsqueeze(-1)

    concat_for_min = torch.cat([-tau * expanded_R1, -tau * expanded_R2], dim=-1)
    soft_t_norm = - (1.0 / tau) * torch.logsumexp(concat_for_min, dim=-1) # [..., N, N, N]

    # Softmax (S-Norm / Mantıksal VEYA)
    soft_composition = (1.0 / tau) * torch.logsumexp(tau * soft_t_norm, dim=-2) # [..., N, N]
    return soft_composition

def lukasiewicz_composition(R1, R2):
    """Lukasiewicz T-Norm ve S-Norm ile Mantıksal Geçişlilik (Composition of Morphisms)."""
    R1_exp = R1.unsqueeze(-1) # [..., N, M, 1]
    R2_exp = R2.unsqueeze(-3) # [..., 1, M, P]
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) # [..., N, M, P]
    composition, _ = torch.max(t_norm, dim=-2) # [..., N, P]
    return composition
def transitive_closure(R, max_steps=5):
    """Bir mantık matrisinin tüm olası geleceğini (Reachability) hesaplar."""
    R_closure = R.clone()
    for _ in range(max_steps):
        new_R = lukasiewicz_composition(R_closure, R)
        R_closure = torch.max(R_closure, new_R)
    return R_closure

def sheaf_gluing(truth_A, truth_B, threshold=0.2):
    """
    [STRICT SHEAF CONDITION]
    Çelişen evrenleri izole eder, uyuşanları birleştirir (Sheaf Condition).
    Gerçek Sheaf koşulu, lokal kesitlerin örtüşme bölgelerinde KESİN olarak
    (veya çok düşük bir toleransla) eşit olmasını gerektirir.
    """
    disagreement = torch.abs(truth_A - truth_B)
    max_conflict = torch.max(disagreement).item()

    if max_conflict > threshold:
        return False, None
    return True, torch.max(truth_A, truth_B)
