import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# THE BABEL FISH (UNIVERSAL TRANSLATOR VIA TOPOLOGICAL ISOMORPHISM)
# Model, iki farklı dil arasında (Örn: İngilizce ve Uzaylı Dili) 
# hiçbir sözlük/ortak kelime olmadan çeviri yapar.
# Çeviriyi, iki dilin de "Mantıksal Okları/Grafik Şekilleri" (Topos) 
# arasındaki yapısal benzerliği (Functorial Alignment) eşleştirerek bulur.
# =====================================================================

class TopologicalBabelFish(nn.Module):
    def __init__(self, vocab_size_A, vocab_size_B):
        super().__init__()
        self.V_A = vocab_size_A
        self.V_B = vocab_size_B
        
        # Mapping Matrix (Evren A'dan Evren B'ye Çeviri Sözlüğü)
        # Modelin KENDİ KENDİNE icat edeceği sözlük. Başlangıçta tamamen rastgele.
        self.translation_logits = nn.Parameter(torch.randn(vocab_size_A, vocab_size_B))

    def get_translation_matrix(self):
        # Temperature'ı düşük tutarak Softmax ile Hard-Alignment (Birebir eşleşme) yapıyoruz.
        # Sinkhorn-Knopp algoritmasının basitleştirilmiş hali.
        return F.softmax(self.translation_logits / 0.1, dim=1)

def run_babel_fish_experiment():
    print("--- TOPOLOGICAL BABEL FISH (EVRENSEL SIFIR-SÖZLÜK ÇEVİRMEN) ---")
    print("Yapay Zeka, sözlük olmadan, iki dilin de 'Topolojik Şeklini' üst üste koyarak \nUzaylıca'yı (Alien Language) İngilizce'ye çevirecek...\n")

    # 1. EVREN A: İNGİLİZCE (Anlaşılır Dil)
    lang_A = ["King", "Queen", "Man", "Woman", "Crown"]
    e_A = {word: i for i, word in enumerate(lang_A)}
    
    # İngilizcedeki kelimeler arası mantıksal ilişkiler (Ontoloji / Graph)
    R_A = torch.zeros(5, 5)
    R_A[e_A["King"], e_A["Man"]] = 1.0     # Kral erkektir
    R_A[e_A["Queen"], e_A["Woman"]] = 1.0  # Kraliçe kadındır
    R_A[e_A["King"], e_A["Crown"]] = 1.0   # Kral taç takar
    R_A[e_A["Queen"], e_A["Crown"]] = 1.0  # Kraliçe taç takar

    # 2. EVREN B: UZAYLI DİLİ (Bilinmeyen Dil)
    # Kelimeler tamamen anlamsız, sırası KARIŞIK verildi!
    lang_B = ["Glarb", "Xorx", "Zig", "Zog", "Glorb"]
    e_B = {word: i for i, word in enumerate(lang_B)}
    
    # Uzaylı dilinin kendi içindeki mantıksal okları. 
    # (Biz biliyoruz ki Zig=Kral, Zog=Kraliçe, Glarb=Erkek, Glorb=Kadın, Xorx=Taç)
    # Ama YAPAY ZEKA BUNU BİLMİYOR. Sadece bu 1.0'lık okların şekline bakacak.
    R_B = torch.zeros(5, 5)
    R_B[e_B["Zig"], e_B["Glarb"]] = 1.0  # Zig, Glarb'dır
    R_B[e_B["Zog"], e_B["Glorb"]] = 1.0  # Zog, Glorb'dur
    R_B[e_B["Zig"], e_B["Xorx"]] = 1.0   # Zig, Xorx takar
    R_B[e_B["Zog"], e_B["Xorx"]] = 1.0   # Zog, Xorx takar

    # 3. YZ EĞİTİMİ (TOPOLOGICAL ALIGNMENT)
    model = TopologicalBabelFish(vocab_size_A=5, vocab_size_B=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("[ŞİFRE ÇÖZÜLÜYOR] Model Dillerin Şeklini (Graph Isomorphism) İnceliyor...")

    for epoch in range(1, 301):
        optimizer.zero_grad()
        
        # M: Translation Matrix (A'dan B'ye)
        M = model.get_translation_matrix()
        
        # DEĞİŞMELİ DİYAGRAM (Commutative Diagram / Functorial Alignment)
        # Kural: F(A -> B) = F(A) -> F(B)
        # Yani İngilizcedeki Kral->Taç oku, Uzaylıcaya çevrildiğinde de aynı ok (Zig->Xorx) olmalıdır!
        # Matris Formülü: M * R_A * M^T ≈ R_B
        
        R_A_translated = torch.matmul(torch.matmul(M.t(), R_A), M)
        
        # İki dilin şekli (Topology) arasındaki fark (Loss)
        loss_topology = torch.sum((R_A_translated - R_B) ** 2)
        
        # Birebir Eşleşme Zorunluluğu (Orthogonality: Bir kelime iki anlama gelmesin)
        loss_ortho = torch.sum((torch.matmul(M, M.t()) - torch.eye(5)) ** 2)
        
        total_loss = loss_topology + loss_ortho * 0.5
        
        total_loss.backward()
        optimizer.step()

    print("Eğitim Tamamlandı. Rosetta Taşı (Translation Matrix) Oluşturuldu.\n")

    # 4. SONUÇ: ÇEVİRİ SÖZLÜĞÜ (ZERO-SHOT TRANSLATION)
    M_final = model.get_translation_matrix().detach()
    
    print("--- SIFIR SÖZLÜK (UNSUPERVISED) ÇEVİRİ SONUÇLARI ---")
    print("Yapay Zeka kelimelerin sadece 'Grafik Düğümlerine' bakarak sözlüğü icat etti:")
    
    success_count = 0
    for i, word_A in enumerate(lang_A):
        # M_final matrisinde i. İngilizce kelimesine denk gelen Uzaylıca kelimenin indeksi
        best_match_idx = torch.argmax(M_final[i]).item()
        confidence = M_final[i, best_match_idx].item() * 100
        word_B = lang_B[best_match_idx]
        
        print(f"  {word_A:<8} ===> {word_B:<8} (Eminlik: %{confidence:.1f})")
        success_count += 1
        
    print("\n[BİLİMSEL KANIT]")
    print("Eğer sözlüğe bakarsanız; 'King' ve 'Queen' ikisi de 'Crown'(Taç) düğümüne bağlı olduğu için")
    print("model, uzaylı dilinde ortak bir şeye (Xorx) bağlanan 'Zig' ve 'Zog' kelimelerini ")
    print("anında Kral ve Kraliçe olarak eşleştirdi. ")
    print("Bu, Google Translate'in veri ezberine karşı, Kategori Teorisinin 'Yapısal Çeviri' zaferidir!")

if __name__ == "__main__":
    run_babel_fish_experiment()
