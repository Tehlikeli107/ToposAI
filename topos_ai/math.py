import torch


class StrictGodelComposition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R1, R2, tau=10.0):
        ctx.save_for_backward(R1, R2)
        ctx.tau = tau
        
        # [KESİN/STRICT İLERİ YÖN - KATEGORİ TEORİSİ]
        R1_exp = R1.unsqueeze(-1)
        R2_exp = R2.unsqueeze(-3)
        t_norm = torch.minimum(R1_exp, R2_exp)
        composition, _ = torch.max(t_norm, dim=-2)
        return composition

    @staticmethod
    def backward(ctx, grad_output):
        # [YUMUŞAK/SOFT GERİ YÖN - DERİN ÖĞRENME BACKPROPAGATION]
        R1, R2 = ctx.saved_tensors
        tau = ctx.tau
        
        with torch.enable_grad():
            R1_soft = R1.detach().requires_grad_(True)
            R2_soft = R2.detach().requires_grad_(True)
            
            R1_exp = R1_soft.unsqueeze(-1)
            R2_exp = R2_soft.unsqueeze(-3)
            N = R1_soft.shape[-1]
            batch_shape = R1_soft.shape[:-2]
            
            expanded_R1 = R1_exp.expand(*batch_shape, N, N, N).unsqueeze(-1)
            expanded_R2 = R2_exp.expand(*batch_shape, N, N, N).unsqueeze(-1)
            
            concat_for_min = torch.cat([-tau * expanded_R1, -tau * expanded_R2], dim=-1)
            soft_t_norm = - (1.0 / tau) * torch.logsumexp(concat_for_min, dim=-1) 
            soft_composition = (1.0 / tau) * torch.logsumexp(tau * soft_t_norm, dim=-2)
            
            soft_composition.backward(grad_output)
            
        return R1_soft.grad, R2_soft.grad, None

def soft_godel_composition(R1, R2, tau=10.0):
    """
    [STRICT TOPOS LOGIC (Hybrid Autograd)]
    Gödel mantığındaki keskin min/max fonksiyonları (Associativity ihlalini önler).
    Backpropagation (Geri Yayılım) sırasında türevleri sıfırlayarak (Dead Gradients) 
    modelin eğitimini durdurmaması için Custom Autograd ile yumuşatılmış türev döndürür.
    """
    return StrictGodelComposition.apply(R1, R2, tau)

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
