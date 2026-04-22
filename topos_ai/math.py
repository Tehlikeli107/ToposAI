import torch

def soft_godel_composition(R1, R2, tau=10.0):
    """
    [DIFFERENTIABLE TOPOS LOGIC]
    Gödel mantığındaki keskin min/max fonksiyonları, Backpropagation (Geri Yayılım)
    sırasında türevleri sıfırlayarak (Dead Gradients) modelin eğitimini durdurur.
    Bunu engellemek için, Sıcaklık (Temperature - tau) parametresi ile çalışan
    'Softmin' ve 'Softmax' (LogSumExp tabanlı) operatörler kullanılır.
    """
    R1_exp = R1.unsqueeze(2) # [N, N, 1]
    R2_exp = R2.unsqueeze(0) # [1, N, N]
    
    N = R1.size(0)
    
    # Negatif eksende logsumexp (Softmin yaklaşımı)
    # Broadcastleri açıkça (explicit) belirleyip yeni bir boyut (unsqueeze) açıyoruz
    expanded_R1 = R1_exp.expand(N, N, N).unsqueeze(-1)
    expanded_R2 = R2_exp.expand(N, N, N).unsqueeze(-1)
    
    concat_for_min = torch.cat([-tau * expanded_R1, -tau * expanded_R2], dim=-1)
    soft_t_norm = - (1.0 / tau) * torch.logsumexp(concat_for_min, dim=-1) # [N, N, N]
    
    # Softmax (S-Norm / Mantıksal VEYA)
    soft_composition = (1.0 / tau) * torch.logsumexp(tau * soft_t_norm, dim=1) # [N, N]
    return soft_composition

def lukasiewicz_composition(R1, R2):
    """Lukasiewicz T-Norm ve S-Norm ile Mantıksal Geçişlilik (Composition of Morphisms)."""
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) 
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

def transitive_closure(R, max_steps=5):
    """Bir mantık matrisinin tüm olası geleceğini (Reachability) hesaplar."""
    R_closure = R.clone()
    for _ in range(max_steps):
        new_R = lukasiewicz_composition(R_closure, R)
        R_closure = torch.max(R_closure, new_R)
    return R_closure

def sheaf_gluing(truth_A, truth_B, threshold=0.2):
    """Çelişen evrenleri izole eder, uyuşanları birleştirir (Sheaf Condition)."""
    certainty_A = torch.abs(truth_A - 0.5) * 2.0
    certainty_B = torch.abs(truth_B - 0.5) * 2.0
    overlap = certainty_A * certainty_B
    disagreement = torch.abs(truth_A - truth_B)
    conflict_score = torch.sum(overlap * disagreement).item()
    
    if conflict_score > threshold:
        return False, None
    return True, torch.max(truth_A, truth_B)
