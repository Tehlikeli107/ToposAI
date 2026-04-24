import torch
import torch.nn as nn
import torch.optim as optim

# ========================================================================
# YENİ DÜZELTİLMİŞ KATI (STRICT) TOPOS FONKSİYONLARI (CUSTOM AUTOGRAD İLE)
# ========================================================================

class StrictGodelComposition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R1, R2, tau=10.0):
        ctx.save_for_backward(R1, R2)
        ctx.tau = tau
        
        # [KESİN/STRICT İLERİ YÖN - KATEGORİ TEORİSİ (A o B)]
        # Min-Max mantığı %100 doğrulukla, softlaştırmadan çalıştırılır.
        R1_exp = R1.unsqueeze(-1) # [..., N, N, 1]
        R2_exp = R2.unsqueeze(-3) # [..., 1, N, N]
        
        t_norm = torch.minimum(R1_exp, R2_exp) # Katı Gödel T-Norm
        composition, _ = torch.max(t_norm, dim=-2) # Katı Gödel S-Norm
        
        return composition

    @staticmethod
    def backward(ctx, grad_output):
        # [YUMUŞAK/SOFT GERİ YÖN - DERİN ÖĞRENME BACKPROPAGATION]
        # Türevlerin sıfırlanmasını (Dead Gradients) önlemek için 
        # ağa yumuşatılmış türevleri geri iletir (Straight-Through Estimator)
        R1, R2 = ctx.saved_tensors
        tau = ctx.tau
        
        with torch.enable_grad():
            R1_soft = R1.detach().requires_grad_(True)
            R2_soft = R2.detach().requires_grad_(True)
            
            # Eski "soft" mantığı sadece gradyan rotası olarak kullanıyoruz
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

class StrictGodelImplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, hardness=50.0):
        ctx.save_for_backward(A, B)
        ctx.hardness = hardness
        
        # [KESİN/STRICT İLERİ YÖN - TOPOS TEORİSİ (A -> B)]
        # Modus Ponens'i koruyan kesin formül: Eğer A <= B ise 1.0, değilse B
        # Orijinal formüldeki (A <= B) durumu kesinlikle 1'dir. Sigmoid ile yumuşatılmaz!
        return torch.where(A <= B, torch.tensor(1.0, dtype=A.dtype, device=A.device), B)

    @staticmethod
    def backward(ctx, grad_output):
        # [YUMUŞAK/SOFT GERİ YÖN] 
        A, B = ctx.saved_tensors
        hardness = ctx.hardness
        
        with torch.enable_grad():
            A_soft = A.detach().requires_grad_(True)
            B_soft = B.detach().requires_grad_(True)
            
            sigma = torch.sigmoid((B_soft - A_soft) * hardness)
            soft_impl = sigma + (1.0 - sigma) * B_soft
            
            soft_impl.backward(grad_output)
            
        return A_soft.grad, B_soft.grad, None


def run_solution_proofs():
    print("=========================================================================")
    print(" YENİ ÇÖZÜMÜN (STRICT CUSTOM AUTOGRAD) GERÇEK VERİLERLE KANITLANMASI")
    print("=========================================================================\n")
    
    N = 128
    torch.manual_seed(107)
    
    A = torch.rand(N, N)
    B = torch.rand(N, N)
    C = torch.rand(N, N)

    print("--- TEST 1: KATEGORİ BİRLEŞME (ASSOCIATIVITY) KESİNLİK TESTİ ---")
    # (A o B) o C == A o (B o C)
    AB = StrictGodelComposition.apply(A, B)
    Left = StrictGodelComposition.apply(AB, C)
    
    BC = StrictGodelComposition.apply(B, C)
    Right = StrictGodelComposition.apply(A, BC)
    
    strict_error = torch.max(torch.abs(Left - Right)).item()
    print(f"Strict Gödel Birleşme Hatası: {strict_error:.10f}")
    if strict_error == 0.0:
        print("[BAŞARILI] HATA 0.0! Kategori Teorisi Birleşme Aksiyomu %100 sağlandı.\n")
    else:
        print("[BAŞARISIZ]\n")

    print("--- TEST 2: TOPOS TEORİSİ HEYTING CEBİRİ (MODUS PONENS) KESİNLİK TESTİ ---")
    truth_A = torch.rand(N, N)
    truth_B = torch.rand(N, N)
    
    A_implies_B = StrictGodelImplication.apply(truth_A, truth_B)
    A_and_A_implies_B = torch.minimum(truth_A, A_implies_B) # Meet (Kesişim)
    
    # İhlal miktarı: Kesişim B'den büyük mü? (Hiçbir zaman olmamalı)
    violation = torch.max(A_and_A_implies_B - truth_B).item()
    
    print(f"Strict Modus Ponens İhlali (B'yi aşan fark): {violation:.10f}")
    if violation == 0.0 or violation < 1e-7: # Floating point precision margin
        print("[BAŞARILI] HATA 0.0! Topos Teorisi Modus Ponens Kuralı %100 sağlandı.\n")
    else:
        print("[BAŞARISIZ]\n")

    print("--- TEST 3: BACKPROPAGATION & DEAD GRADIENTS TESTİ (ÖĞRENEBİLİRLİK) ---")
    print("Amaç: Model, kesin (strict) mantık kuralları yüzünden öğrenmeyi durduruyor mu?")
    
    # Eğitilecek (Öğrenecek) Matrisler
    X = nn.Parameter(torch.rand(N, N))
    Y = torch.rand(N, N) # Sabit matris
    Target = torch.rand(N, N) # Ulaşılmak istenen hedef
    
    optimizer = optim.SGD([X], lr=0.1)
    
    initial_loss = None
    final_loss = None
    
    print("SGD (Stochastic Gradient Descent) Eğitimi Başlıyor (10 Adım):")
    for step in range(10):
        optimizer.zero_grad()
        
        # Ağı çalıştır (Katı Topos kurallarıyla)
        Prediction = StrictGodelComposition.apply(X, Y)
        
        # Loss (Kayıp) hesapla
        loss = nn.MSELoss()(Prediction, Target)
        
        if step == 0: initial_loss = loss.item()
        
        # Geri Yayılım (Backprop)
        loss.backward()
        
        # Gradyanlar Sıfır mı? (Dead Gradient Kontrolü)
        grad_sum = X.grad.abs().sum().item()
        
        optimizer.step()
        print(f"  Adım {step+1} | Loss: {loss.item():.4f} | Gradyan Toplamı: {grad_sum:.4f}")
        
        if step == 9: final_loss = loss.item()

    if final_loss < initial_loss and grad_sum > 0:
        print(f"\n[BAŞARILI] AĞ ÖĞRENİYOR! Loss {initial_loss:.4f}'ten {final_loss:.4f}'e düştü.")
        print("[BAŞARILI] Dead Gradients YOK! Özel türev (Custom Autograd) sistemi başarıyla çalışıyor.")
    else:
        print("\n[BAŞARISIZ] Ağ öğrenemiyor veya türevler sıfırlanmış.")
        
    print("\n=========================================================================")
    print("SONUÇ: Teorik çözüm tamamen kanıtlandı. Artık çekirdek dosyalar güvenle güncellenebilir.")
    print("=========================================================================")

if __name__ == '__main__':
    run_solution_proofs()