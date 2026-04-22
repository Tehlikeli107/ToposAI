import torch
import torch.nn as nn
import torch.nn.functional as F

class ToposHeytingAttention(nn.Module):
    """
    Topos tabanlı Dil Modeli (LLM) Attention Katmanı.
    Geometrik Dot-Product (Q * K^T) yerine, Topos'un içsel mantığı olan 
    Sezgisel Mantık (Intuitionistic Logic / Heyting Algebra) kullanır.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

    def forward(self, x, apply_causal_mask=True):
        batch, seq_len, dim = x.shape
        
        # 1. Topos Mantığında değerler [0, 1] arasında "Doğruluk Dereceleri" (Truth Values) olmalıdır.
        # Bu yüzden Query ve Key'i sigmoid ile 0-1 arasına hapsediyoruz.
        Q = torch.sigmoid(self.q_proj(x)) # "Aradığım mantıksal koşullar"
        K = torch.sigmoid(self.k_proj(x)) # "Bağlamın (Context) sağladığı gerçekler"
        V = self.v_proj(x)
        
        # Boyutları eşleştirme: Q(batch, seq_q, 1, dim), K(batch, 1, seq_k, dim)
        Q_exp = Q.unsqueeze(2)
        K_exp = K.unsqueeze(1)
        
        # 2. Gödel-Dummett Heyting İmplikası (Topos İçsel Mantığı):
        # "Q => K" (Eğer Q ise K) doğruluk değeri:
        # Eğer Q <= K ise Doğruluk = 1.0 (Tamamen Sağlandı)
        # Eğer Q > K ise Doğruluk = K (Kısmi Sağlandı)
        # Klasik attention'daki dot-product'ın yerini bu mantıksal çıkarım alır.
        implication = torch.where(Q_exp <= K_exp, torch.ones_like(K_exp), K_exp)
        
        # 3. Bir token'ın tüm özelliklerinin (feature) sağladığı toplam doğruluk (Conjunction)
        # Normalde minimum (min) alınır ama türevlenebilirlik (gradient) için ortalama (mean) alıyoruz.
        attention_truth = implication.mean(dim=-1) # (batch, seq, seq)
        
        # Causal Mask (Geleceği Görmeyi Engelleme - LLM mantığı)
        if apply_causal_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
            # Topos'ta yanlış/imkansız olan şey 0 doğruluk değerine (False / Bottom) denktir.
            attention_truth = attention_truth.masked_fill(mask == 0, 0.0)
            
        # Elde edilen Mantıksal Doğruluk Değerlerini (Truth Values) Vektörleri toplamak için normalize et
        attn_weights = F.softmax(attention_truth * 10, dim=-1) # 10: Temperature
        
        # 4. Değerleri (Values) Topos Mantığına göre birleştir
        output = torch.matmul(attn_weights, V)
        
        return output, attention_truth

# --- Test Senaryosu ---
def test_topos_llm():
    torch.manual_seed(42)
    
    # 4 Token'lık bir cümle: "Kedi", "Sütü", "Çok", "Sever"
    batch_size = 1
    seq_len = 4
    dim = 8 # Embedding boyutu
    
    # Rastgele embedding'ler (Gerçek bir modelde bunlar Embedding katmanından gelir)
    x = torch.randn(batch_size, seq_len, dim)
    
    # Topos Attention modelini oluştur
    topos_attention = ToposHeytingAttention(dim=dim)
    
    # İleri besleme (Forward Pass)
    out, truth_matrix = topos_attention(x)
    
    print("--- TOPOS LLM: HEYTING ATTENTION TESTİ ---")
    print(f"Girdi Boyutu (Cümle): {x.shape}")
    print(f"Çıktı Boyutu (Bağlamsal Anlam): {out.shape}")
    
    print("\nMantıksal Doğruluk Matrisi (Heyting Implication Truth Values):")
    # Maskelenmiş ve mantıksal olarak hesaplanmış matris
    print(torch.round(truth_matrix[0] * 100) / 100) # Okunabilirlik için yuvarlandı
    
    print("\nFARK NEDİR?")
    print("Klasik LLM (Dot Product) simetriktir (Q*K == K*Q) ve kelimeler arası Geometrik Benzerliğe bakar.")
    print("Topos LLM (Heyting) ASİMETRİKTİR. 'Q mantıksal olarak K'dan çıkarsanabilir mi?' sorusuna bakar.")
    
if __name__ == "__main__":
    test_topos_llm()
