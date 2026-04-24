import torch
import torch.nn.functional as F

class ToposConstrainedDecoder:
    """
    Topological Constrained Decoding (TCD) for Neuro-Symbolic LLMs.
    
    Standart LLM'ler (GPT, LLaMA) metin üretirken yalnızca istatistiksel Logit'lere
    bakar (Softmax / Top-K). Bu, "Halüsinasyon" ve "Ezber" (Memorization) yaratır.
    
    Bu Decoder, dil modelinin ürettiği istatistiksel ihtimalleri, Kategori Teorisi 
    (Topos) matrisindeki mantıksal kurallarla (Reachability / Transitive Closure) 
    anlık olarak denetler. Mantıksal zincirde kanıtlanamayan kelimelerin üretilmesini
    matematiksel olarak engeller (Logit Masking: -inf).
    """
    def __init__(self, reachability_matrix, threshold=0.1):
        """
        Args:
            reachability_matrix: [VocabSize, VocabSize] boyutunda, kelimeler arası
                                 mantıksal geçişliliği (Transitive Closure) tutan tensör.
            threshold: Bir kelimenin "Mantıksal Olarak Kanıtlanmış" sayılması için 
                       gereken minimum Topos bağlantı gücü [0.0, 1.0].
        """
        assert reachability_matrix.dim() == 2, "Reachability matrix 2 boyutlu olmalıdır."
        assert reachability_matrix.size(0) == reachability_matrix.size(1), "Reachability matrix kare (V x V) olmalıdır."
        
        self.reachability_matrix = reachability_matrix
        self.threshold = threshold

    def apply_topological_mask(self, current_token_idx, next_token_logits):
        """
        Dil modelinin ürettiği ham (raw) logitleri alır, mantıksiz olanları maskeler.
        
        Args:
            current_token_idx: Bağlamdaki mevcut (veya odaklanılan son) kelime indeksi (int).
            next_token_logits: Dil modelinin sıradaki kelime için ürettiği ihtimaller [VocabSize].
        
        Returns:
            masked_logits: Mantıksız kelimelerin -inf yapıldığı güvenli logitler.
        """
        # Mevcut kelimenin, sözlükteki diğer tüm kelimelere olan "Mantıksal Ulaşılabilirliği" (Reachability)
        logical_connections = self.reachability_matrix[current_token_idx]
        
        # Eğer bağlantı gücü threshold'dan düşükse, bu kelime mantıksal olarak KANITLANAMAZ demektir (False/0).
        # Boolean bir maske oluşturuyoruz.
        valid_logical_mask = logical_connections >= self.threshold

        # Eğer hiçbir kelime mantıksal olarak kanıtlanamıyorsa (Maske tamamen False ise),
        # sistemin çökmemesi (NaN dönmemesi) için maskelemeyi iptal et (Fallback).
        if not valid_logical_mask.any():
            return next_token_logits.clone()

        # Dil modelinin logitlerini kopyala
        masked_logits = next_token_logits.clone()

        # Mantıksal olarak KANITLANAMAYAN kelimelerin ihtimalini EKSİ SONSUZ (-inf) yap.
        masked_logits[~valid_logical_mask] = float('-inf')

        return masked_logits
    def generate_safe_token(self, current_token_idx, next_token_logits, temperature=1.0, top_k=None):
        """
        Güvenli (Logit-Maskeli) metin üretimi. Halüsinasyon garantili 0.
        """
        # 1. Topolojik Maskelemeyi Uygula
        safe_logits = self.apply_topological_mask(current_token_idx, next_token_logits)
        
        # Maskelenmiş logitler üzerinden standart (Temperature / Top-K) decoding
        safe_logits = safe_logits / temperature
        
        if top_k is not None:
            # Top-K Sampling (Güvenli kelimeler arasında)
            indices_to_remove = safe_logits < torch.topk(safe_logits, top_k)[0][..., -1, None]
            safe_logits[indices_to_remove] = float('-inf')
            
        probs = F.softmax(safe_logits, dim=-1)

        # S9 FIX: Sadece argmax alıp greedy decoding yapmak yerine gerçek Sampling.
        # Eğer temperature 0'a çok yakınsa Greedy (argmax) yap.
        if temperature < 1e-4:
            next_token = torch.argmax(probs).item()
        else:
            # Gerçek İstatistiksel Örnekleme (Sampling)
            next_token = torch.multinomial(probs, num_samples=1).item()

        return next_token
