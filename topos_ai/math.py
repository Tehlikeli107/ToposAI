import torch

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
