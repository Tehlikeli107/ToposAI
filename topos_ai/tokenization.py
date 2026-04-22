import torch
from typing import List, Dict

# =====================================================================
# TOPOLOGICAL TOKENIZER (BPE KILLER)
# Problem: Klasik LLM'ler (GPT-4) Byte-Pair Encoding (BPE) kullanır. 
# BPE, heceleri "sadece yan yana sık gelmelerine (İstatistik)" göre birleştirir.
# Çözüm: Topological Tokenizer, harfler arasındaki "Yönlü Morfizmaları" (A->B)
# hesaplar. Eğer 'q' harfi her zaman 'u' harfini gerektiriyorsa (q -> u = 1.0),
# bu ikisini topolojik olarak mühürler. Yani dili istatistikle değil,
# Kategori Teorisi (Nedensellik) ile parse eder.
# =====================================================================

class TopologicalTokenizer:
    """
    [REAL TOPOLOGICAL TOKENIZER]
    BPE (Byte-Pair Encoding) gibi sadece istatistiksel frekansa (sıklığa) bakmaz.
    Karakterlerin veya hecelerin "Birbirini ne kadar zorunlu kıldığına" 
    (Logical Implication / Morphism Strength) bakar. 
    Örn: 'q' harfi her zaman 'u' harfini gerektiriyorsa (P(u|q) ≈ 1.0),
    frekansları düşük olsa bile bunları topolojik bir bütün (qu) olarak mühürler.
    """
    def __init__(self, vocab_size=5000):
        self.target_vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.reverse_vocab = {}
        
    def _compute_topological_morphisms(self, token_list):
        """
        Token listesinde, A -> B geçişinin "Gereklilik Gücünü" (Morphism) hesaplar.
        Güç = P(B | A) = Count(A, B) / Count(A).
        Eğer A varsa, B'nin gelmesi ne kadar 'Zorunlu' (Deterministic)?
        """
        pair_counts = {}
        single_counts = {}
        
        for i in range(len(token_list) - 1):
            A = token_list[i]
            B = token_list[i+1]
            
            single_counts[A] = single_counts.get(A, 0) + 1
            pair_counts[(A, B)] = pair_counts.get((A, B), 0) + 1
            
        morphism_strengths = {}
        for (A, B), count in pair_counts.items():
            # A'dan B'ye giden ok'un gücü (Gereklilik / Implication)
            strength = count / single_counts[A]
            morphism_strengths[(A, B)] = strength
            
        return morphism_strengths

    def train(self, text: str):
        print(f"\n[TOPOLOGICAL TOKENIZER] Eğitiliyor... (Hedef Vocab: {self.target_vocab_size})")
        
        # Başlangıçta her karakter bir tokendır (Base Alphabet)
        unique_chars = sorted(list(set(text)))
        self.vocab = {c: i for i, c in enumerate(unique_chars)}
        self.reverse_vocab = {i: c for i, c in enumerate(unique_chars)}
        
        current_tokens = [c for c in text]
        current_id = len(self.vocab)
        
        merge_count = 0
        while len(self.vocab) < self.target_vocab_size:
            # Harfler/Heceler arasındaki nedensellik (morfizma) gücünü bul
            morphisms = self._compute_topological_morphisms(current_tokens)
            
            if not morphisms:
                break
                
            # BPE'den Farkı: "En çok geçen çifti" DEĞİL, "Birbirini en çok GEREKTİREN (Morphism=1.0) çifti" birleştir.
            # Eşitlik durumunda en çok geçen çifti (Count) tie-breaker olarak kullanmak için filtreleme yaparız.
            best_pair = max(morphisms.items(), key=lambda x: (x[1], x[0]))[0]
            best_strength = morphisms[best_pair]
            
            if best_strength < 0.01: # Artık anlamlı bir gereklilik bağı kalmadıysa dur
                break
                
            # A ve B tokenlarını "AB" olarak topolojik olarak mühürle
            new_token_str = best_pair[0] + best_pair[1]
            self.vocab[new_token_str] = current_id
            self.reverse_vocab[current_id] = new_token_str
            self.merges[best_pair] = new_token_str
            
            # Metni yeni mühürlü tokenlarla güncelle (Merge Pass)
            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if i < len(current_tokens) - 1 and (current_tokens[i], current_tokens[i+1]) == best_pair:
                    new_tokens.append(new_token_str)
                    i += 2
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
                    
            current_tokens = new_tokens
            current_id += 1
            merge_count += 1
            
            if merge_count % 100 == 0:
                print(f"  > {merge_count} Topolojik Birleşme (Merge) yapıldı. Kelimeler nedensellikle bağlanıyor... Sözlük: {len(self.vocab)}")
                
        print(f"✅ Eğitim Tamamlandı. {len(self.vocab)} Topolojik Token icat edildi.")
        print("  Örnek Keşifler (Kesin Nedensellik Taşıyan Heceler):", list(self.vocab.keys())[-10:])

    def encode(self, text: str) -> List[int]:
        """Metni, topolojik olarak keşfedilmiş token ID'lerine çevirir (İleri doğru tarama)."""
        tokens = [c for c in text]
        
        # Eğitilmiş birleşme kurallarını sırayla uygula
        for (A, B), new_token_str in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == A and tokens[i+1] == B:
                    new_tokens.append(new_token_str)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        # Sonucu ID listesine çevir (Eğer bilinmeyen bir harf varsa 0 veya 'UNK' farz et)
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, tokens: List[int]) -> str:
        """Token ID'lerini tekrar insan diline (String) çevirir."""
        return "".join([self.reverse_vocab.get(t, "") for t in tokens])
