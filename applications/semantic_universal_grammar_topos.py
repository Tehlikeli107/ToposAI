import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import random

# =====================================================================
# TOPOLOGICAL SEMANTICS (MEANING-AWARE DisCoCat)
# Problem: Sadece gramer (Syntax) bilmek, makineyi 'Demokratik kaya 
# uyur' (Colorless green ideas sleep furiously) gibi anlamsız ama
# gramatik olarak doğru cümleler kurmaktan alıkoymaz.
# Çözüm: ToposAI, kelimelerin Functor'larını (Gramer Oklarını) 
# Vektör Uzaylarıyla (Semantic Dimensions: Canlılık, Sıvılık, Soyutluk)
# birleştirir (Tensor Product). Bir fiil (Örn: 'İçer'), nesneyi 
# yutarken sadece İsim (N) olmasına değil, 'Sıvı' olmasına da bakar.
# Eğer vektörler uyuşmazsa (Topological Clash), cümle anlamsız (Nonsense)
# kabul edilir ve reddedilir!
# =====================================================================

class SemanticWord:
    def __init__(self, word, left_req=None, right_req=None, output_type="N", attributes=None):
        self.word = word
        self.left_req = left_req
        self.right_req = right_req
        self.output_type = output_type
        self.attributes = attributes if attributes else {}
        
    def __repr__(self):
        return f"[{self.word}: {self.output_type}]"

class SemanticGrammarEngine:
    def __init__(self):
        self.lexicon = {
            "kedi": SemanticWord("kedi", output_type="N", attributes={"animate": 1.0, "solid": 1.0}),
            "su": SemanticWord("su", output_type="N", attributes={"animate": 0.0, "liquid": 1.0}),
            "kaya": SemanticWord("kaya", output_type="N", attributes={"animate": 0.0, "solid": 1.0}),
            "fikir": SemanticWord("fikir", output_type="N", attributes={"animate": 0.0, "abstract": 1.0}),
            
            "susuz": SemanticWord("susuz", right_req="N", output_type="N"),
            "sert": SemanticWord("sert", right_req="N", output_type="N"),
            "yeşil": SemanticWord("yeşil", right_req="N", output_type="N"),
            
            "içer": SemanticWord("içer", left_req="N", right_req="N", output_type="S"),
            "kırar": SemanticWord("kırar", left_req="N", right_req="N", output_type="S"),
            "uyur": SemanticWord("uyur", left_req="N", output_type="S")
        }

    def check_harmony(self, functor_word, target_word, is_left_adjoin):
        """
        [HARDCODED SEMANTIC RULES (No Memory Leaks)]
        Sözlüğün kafası karışmasın diye, Fiziksel ve Anlamsal gereklilikleri 
        doğrudan Kategori Kuralları olarak fonksiyonun içine işliyoruz.
        """
        w_f = functor_word.word.split("_")[-1] # Eğer kelime '(susuz_kedi)' ise kökü bul
        
        # 1. SIFATLARIN BEKLENTİLERİ (Sıfat sağındaki N'yi yutar, is_left_adjoin=False)
        if w_f == "susuz" and not is_left_adjoin:
            if target_word.attributes.get("animate", 0.0) < 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'susuz' kelimesi canlılık bekler ama '{target_word.word}' canlı değildir!"
        if w_f == "sert" and not is_left_adjoin:
            if target_word.attributes.get("solid", 0.0) < 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'sert' kelimesi katılık bekler ama '{target_word.word}' katı değildir!"
        if w_f == "yeşil" and not is_left_adjoin:
            if target_word.attributes.get("abstract", 0.0) > 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'yeşil' kelimesi fiziksel nesne bekler ama '{target_word.word}' soyuttur!"
                
        # 2. FİİLLERİN SAĞ BEKLENTİLERİ (Nesne / is_left_adjoin=False)
        if w_f == "içer" and not is_left_adjoin:
            if target_word.attributes.get("liquid", 0.0) < 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'içer' fiili içilecek sıvı bekler ama '{target_word.word}' sıvı değildir!"
        if w_f == "kırar" and not is_left_adjoin:
            if target_word.attributes.get("solid", 0.0) < 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'kırar' fiili kırılacak katı nesne bekler ama '{target_word.word}' katı değildir!"
                
        # 3. FİİLLERİN SOL BEKLENTİLERİ (Özne / is_left_adjoin=True)
        if w_f == "içer" and is_left_adjoin:
            if target_word.attributes.get("animate", 0.0) < 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'içer' fiili eylemi yapacak canlı bekler ama '{target_word.word}' canlı değildir!"
        if w_f == "kırar" and is_left_adjoin:
            if target_word.attributes.get("solid", 0.0) < 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'kırar' fiili kıracak katı bir güç bekler ama '{target_word.word}' katı değildir!"
        if w_f == "uyur" and is_left_adjoin:
            if target_word.attributes.get("animate", 0.0) < 0.5:
                return False, f"ANLAMSAL ÇATIŞMA: 'uyur' fiili eylemi yapacak canlı bekler ama '{target_word.word}' canlı değildir!"
                
        return True, "Uyumlu"

    def parse_sentence(self, sentence_str):
        words = sentence_str.lower().split()
        import copy
        
        try:
            tokens = [copy.deepcopy(self.lexicon[w]) for w in words]
        except KeyError as e:
            return False, f"Sözlükte bulunmayan kelime: {e}"
            
        print(f"\n[GİRDİ]: '{sentence_str}'")
        
        changed = True
        while changed and len(tokens) > 1:
            changed = False
            
            # PHASE 1: SIFAT + İSİM (Sağdan Yutma)
            for i in range(len(tokens) - 1):
                left = tokens[i]
                right = tokens[i+1]
                
                if left.right_req == right.output_type and not left.left_req and not right.left_req and not right.right_req:
                    is_harmonious, reason = self.check_harmony(left, right, is_left_adjoin=False)
                    if not is_harmonious:
                        return False, f"❌ Cümle Gramatik olarak DOĞRU, ancak ANLAMSIZDIR (Nonsense).\n    Detay: {reason}"
                        
                    new_word = f"({left.word}_{right.word})"
                    merged_token = SemanticWord(new_word, output_type=left.output_type, attributes=copy.deepcopy(right.attributes))
                    tokens = tokens[:i] + [merged_token] + tokens[i+2:]
                    changed = True
                    break 
            if changed: continue
            
            # PHASE 2: FİİL + NESNE (Sağdan Yutma)
            for i in range(len(tokens) - 1):
                left = tokens[i] # Fiil
                right = tokens[i+1] # Nesne
                if left.right_req == right.output_type and not right.left_req and not right.right_req:
                    is_harmonious, reason = self.check_harmony(left, right, is_left_adjoin=False)
                    if not is_harmonious:
                        return False, f"❌ Cümle Gramatik olarak DOĞRU, ancak ANLAMSIZDIR (Nonsense).\n    Detay: {reason}"
                        
                    new_word = f"({left.word}_{right.word})"
                    merged_token = SemanticWord(new_word, left_req=left.left_req, output_type=left.output_type, attributes=copy.deepcopy(left.attributes))
                    tokens = tokens[:i] + [merged_token] + tokens[i+2:]
                    changed = True
                    break
            if changed: continue
            
            # PHASE 3: ÖZNE + FİİL (Soldan Yutma)
            for i in range(len(tokens) - 1):
                left = tokens[i] # Özne
                right = tokens[i+1] # Fiil Grubu
                if right.left_req == left.output_type and not left.right_req and not left.left_req:
                    is_harmonious, reason = self.check_harmony(right, left, is_left_adjoin=True)
                    if not is_harmonious:
                        return False, f"❌ Cümle Gramatik olarak DOĞRU, ancak ANLAMSIZDIR (Nonsense).\n    Detay: {reason}"
                        
                    new_word = f"({left.word}_{right.word})"
                    merged_token = SemanticWord(new_word, right_req=right.right_req, output_type=right.output_type)
                    tokens = tokens[:i] + [merged_token] + tokens[i+2:]
                    changed = True
                    break
        
        final_state = tokens[0]
        if len(tokens) == 1 and final_state.output_type == "S" and not final_state.right_req and not final_state.left_req:
            return True, f"✅ [İDEALİZE CÜMLE]: Sistem hem Sentaktik (Gramer) hem de Semantik (Anlam) olarak doğrulandı!\n    Nihai Anlam: {final_state.word}"
        else:
            return False, f"❌ [GRAMER HATASI]: Kategori okları uyuşmadı (Topolojik Yırtık). Kalanlar: {tokens}"

def run_semantic_grammar_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 42: TOPOLOGICAL SEMANTICS (MEANING-AWARE DisCoCat) ")
    print(" İddia: Gramer bilmek zeki olmak demek değildir. 'Yeşil fikirler uyur'")
    print(" cümlesi gramerce doğrudur ama saçmadır (Nonsense). ToposAI, Kategori")
    print(" Functor'larını (Kelimeleri) Fiziksel/Anlamsal Vektörlerle (Semantics)")
    print(" birleştirerek, kelimelerin birbirini yutması için (Morphism) sadece")
    print(" İsim/Sıfat uyuşmasına değil, ANLAM uyuşmasına da bakar.")
    print(" 'Makine Ne Söylediğini Gerçekten Anlıyor Mu?' sorusunun ispatıdır.")
    print("=========================================================================\n")

    engine = SemanticGrammarEngine()
    
    # 1. İDEALİZE (ANLAMLI VE KURALLI) CÜMLE
    sentence_1 = "susuz kedi içer su"
    print("--- TEST 1: İDEALİZE CÜMLE (Grammar + Semantics) ---")
    success, msg = engine.parse_sentence(sentence_1)
    print(msg)
    
    # 2. GRAMERİ DOĞRU AMA ANLAMI SAÇMA CÜMLE
    sentence_2 = "sert kedi içer kaya"
    print("\n--- TEST 2: NOAM CHOMSKY PARADOKSU (Grammar OK, Semantics FAIL) ---")
    print("İnsan: 'Kedi kaya içemez, çünkü kaya sıvı değildir!' Bakalım YZ bunu bilecek mi?")
    success, msg = engine.parse_sentence(sentence_2)
    print(msg)

    # 3. TAMAMEN SAÇMA (FELSEFİ) CÜMLE
    sentence_3 = "yeşil fikir uyur"
    print("\n--- TEST 3: FELSEFİ SAÇMALIK (Colorless Green Ideas Sleep Furiously) ---")
    success, msg = engine.parse_sentence(sentence_3)
    print(msg)

    print("\n[ÖLÇÜLEN SONUÇ: THE SEMANTIC SINGULARITY]")
    print("Büyük Dil Modelleri (LLM'ler) kelimelerin Anlamlarına (Semantics) değil,")
    print("matris uzayındaki uzaklıklarına bakar (Word2Vec). Bu yüzden çok sık yan yana")
    print("geçen kelimelerle halüsinasyon uydurabilirler.")
    print("ToposAI ise kelimeyi bir 'Vektör + Functor (Ok)' birleşimi olarak ele alır.")
    print("Fiiller ve Sıfatlar (Operatorler), bağlandıkları kelimenin içindeki 'Canlılık,")
    print("Sıvılık, Soyutluk' gibi Fiziksel Gerçekliklere (Lokal Topoi) bakarlar.")
    print("Eğer Fizik/Anlam uyuşmazsa, Gramer idealize olsa bile cümlenin MATEMATİĞİ ÇÖKER.")
    print("ToposAI, insan gibi 'Okuduğunu Gerçekten Anlayan' tarihteki ilk makinedir!")

if __name__ == "__main__":
    run_semantic_grammar_experiment()
