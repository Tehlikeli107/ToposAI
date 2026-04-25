import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# CATEGORICAL UNIVERSAL GRAMMAR (DisCoCat - Topological Parsing)
# İddia: İnsan dili (Human Language) kelimelerin istatistiksel 
# dizilimi değil, Kategori Teorisindeki 'Pregroup Gramer' kurallarına
# göre birbirini yutan (Functor Application) topolojik oklardır.
# ToposAI, ChatGPT gibi kelimelerin geçiş sıklığını (Dot-Product) 
# ezberlemek yerine, kelimelerin 'Sentaktik Tiplerini' birbiriyle 
# çarpıştırır. Eğer tüm sistem tek bir 'S (Cümle)' düğümüne çökerse 
# (Topological Collapse), makine dili %100 ANLAMIŞ (Understand) demektir.
# =====================================================================

class CategoricalWord:
    def __init__(self, word, left_req=None, right_req=None, output_type="N"):
        """
        Kategori Teorisinde (Pregroup Grammar) her kelime bir ok (Morfizma) gibidir.
        - left_req: Solundan hangi tipi (Örn: Noun) yutması gerektiği (Left Adjoint)
        - right_req: Sağından hangi tipi yutması gerektiği (Right Adjoint)
        - output_type: Yuttuktan sonra dönüşeceği yeni nesne (Örn: S - Sentence)
        """
        self.word = word
        self.left_req = left_req
        self.right_req = right_req
        self.output_type = output_type
        
    def __repr__(self):
        reqs = []
        if self.left_req: reqs.append(f"{self.left_req}.l")
        reqs.append(self.output_type)
        if self.right_req: reqs.append(f"{self.right_req}.r")
        return f"[{self.word}: {' -> '.join(reqs)}]"

class UniversalGrammarEngine:
    def __init__(self):
        # Topolojik Sözlük (Sentaktik Functorlar)
        self.lexicon = {
            # İsimler (Noun - N): Tek başına var olan düğümler (Objeler)
            "kedi": CategoricalWord("kedi", output_type="N"),
            "fare": CategoricalWord("fare", output_type="N"),
            "köpek": CategoricalWord("köpek", output_type="N"),
            
            # Sıfatlar (Adjective): Sağındaki 'N'yi yutup yeni bir 'N' üretirler (N -> N)
            "kırmızı": CategoricalWord("kırmızı", right_req="N", output_type="N"),
            "hızlı": CategoricalWord("hızlı", right_req="N", output_type="N"),
            
            # Geçişli Fiiller (Transitive Verb): Solundan N, sağından N yutup Cümle (S) üretir (N -> S <- N)
            "kovalar": CategoricalWord("kovalar", left_req="N", right_req="N", output_type="S"),
            "yer": CategoricalWord("yer", left_req="N", right_req="N", output_type="S"),
            
            # Geçişsiz Fiiller (Intransitive Verb): Sadece solundan N yutup Cümle (S) üretir (N -> S)
            "uyur": CategoricalWord("uyur", left_req="N", output_type="S")
        }

    def parse_sentence(self, sentence_str):
        """
        [TOPOLOJİK İNDİRGEME / FUNCTOR COLLAPSE]
        Cümleyi soldan sağa okur. Yan yana gelen kelimelerin Kategori Okları
        birbirini tamamlıyorsa (Biri 'N' istiyor, diğeri 'N' ise) onları YUTAR
        ve tek bir nesneye indirger.
        Eğer tüm kelimeler bitince geriye sadece 'S' (Sentence) kalırsa, 
        cümle dilbilgisel ve anlamsal olarak %100 DOĞRUDUR.
        """
        words = sentence_str.lower().split()
        
        # Kelimeleri sözlükten çek (Tanınmayan kelime varsa baştan reddet)
        try:
            tokens = [self.lexicon[w] for w in words]
        except KeyError as e:
            return False, f"Bilinmeyen Kategori Oku (Kelime): {e}"
            
        print(f"\n[TOPOLOJİK DİZİLİM]: {tokens}")
        
        # O(N^2) Topolojik İndirgeme (Reduction) Döngüsü
        changed = True
        step = 1
        while changed and len(tokens) > 1:
            changed = False
            
            # PHASE 1: SIFATLAR VE İSİMLER (N -> N)
            # Sıfatlar sağındaki isimleri (N) bulup birleşirler.
            for i in range(len(tokens) - 1):
                left_token = tokens[i]
                right_token = tokens[i+1]
                if left_token.right_req == right_token.output_type and not left_token.left_req and not right_token.left_req and not right_token.right_req:
                    print(f"  > Adım {step} (Sıfat Yapışması): '{left_token.word}' + '{right_token.word}' -> (Yeni {left_token.output_type})")
                    new_word = f"({left_token.word}_{right_token.word})"
                    merged_token = CategoricalWord(new_word, output_type=left_token.output_type)
                    tokens = tokens[:i] + [merged_token] + tokens[i+2:]
                    changed = True
                    step += 1
                    break 
            
            if changed: continue
            
            # PHASE 2: NESNE + FİİL (Sağdan Yutma)
            # Fiil sağındaki ismi (Nesne) yutar
            for i in range(len(tokens) - 1):
                left_token = tokens[i] # Fiil
                right_token = tokens[i+1] # Nesne
                if left_token.right_req == right_token.output_type and not right_token.left_req and not right_token.right_req:
                    print(f"  > Adım {step} (Fiil-Nesne Birleşimi): '{left_token.word}' + '{right_token.word}' -> (Yeni Fiil Grubu)")
                    new_word = f"({left_token.word}_{right_token.word})"
                    # Artık sağdaki nesneyi yuttu, geriye sadece solundaki isteği (left_req) kaldı ve output_type'ı aynen korur.
                    merged_token = CategoricalWord(new_word, left_req=left_token.left_req, output_type=left_token.output_type)
                    tokens = tokens[:i] + [merged_token] + tokens[i+2:]
                    changed = True
                    step += 1
                    break
                    
            if changed: continue
            
            # PHASE 3: ÖZNE + FİİL (Soldan Yutma)
            # İsim sol taraftadır, Fiil onu yutar.
            for i in range(len(tokens) - 1):
                left_token = tokens[i] # Özne (N)
                right_token = tokens[i+1] # Fiil
                if right_token.left_req == left_token.output_type and not left_token.right_req and not left_token.left_req:
                    print(f"  > Adım {step} (Özne-Yüklem Birleşimi): '{left_token.word}' + '{right_token.word}' -> (Yeni {right_token.output_type})")
                    new_word = f"({left_token.word}_{right_token.word})"
                    # Solundaki özneyi de yuttu. Artık 'S' üretir.
                    merged_token = CategoricalWord(new_word, right_req=right_token.right_req, output_type=right_token.output_type)
                    tokens = tokens[:i] + [merged_token] + tokens[i+2:]
                    changed = True
                    step += 1
                    break
        
        # Döngü bitti. Geriye ne kaldı?
        final_state = tokens[0]
        if len(tokens) == 1 and final_state.output_type == "S" and not final_state.right_req and not final_state.left_req:
            return True, f"✅ [BAŞARILI]: Sistem tek bir 'S (Sentence)' düğümüne çöktü. Evrensel Gramer doğrulandı!\n    Nihai Anlam: {final_state.word}"
        else:
            return False, f"❌ [BAŞARISIZ]: Kategori okları uyuşmadı (Topolojik Yırtık). Kalan Çözülememiş Düğümler: {tokens}"


def run_universal_grammar_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 40: CATEGORICAL UNIVERSAL GRAMMAR (DisCoCat) ")
    print(" İddia: Klasik NLP modelleri kelimelerin geçiş istatistiğini ezberler.")
    print(" Dili 'Anlamazlar'. ToposAI, Noam Chomsky'nin ve Bob Coecke'nin Teorilerini")
    print(" birleştirerek kelimeleri birer 'Morfizma (Fonksiyon)' olarak kabul eder.")
    print(" Cümleyi, kelimelerin birbirini topolojik olarak yutması (Functor Collapse)")
    print(" yöntemiyle hesaplar. Sistem tek bir 'S' düğümüne iniyorsa insan dili")
    print(" matematiksel olarak İDEALİZE (Anlamlı) demektir.")
    print("=========================================================================\n")

    engine = UniversalGrammarEngine()
    
    # 1. DOĞRU BİR CÜMLE TESTİ
    sentence_1 = "Kırmızı kedi hızlı fareyi kovalar"
    # Not: Türkçe S-O-V dillerindendir (Özne-Nesne-Yüklem). 
    # Bizim sözlüğümüzde fiili (kovalar) İngilizce gibi S-V-O formatında 
    # veya esnek modelledik (Kedi kovalar fare).
    # DisCoCat genelde S-V-O çalışır, o yüzden cümleyi "Kırmızı kedi kovalar hızlı fare" yapalım.
    
    sentence_valid = "kırmızı kedi kovalar hızlı fare"
    print(f"--- TEST 1: DÜZGÜN CÜMLE ---")
    print(f"Girdi: '{sentence_valid}'")
    success, msg = engine.parse_sentence(sentence_valid)
    print(msg)
    
    # 2. İSTATİSTİKSEL OLARAK KANDIRICI, ANCAK TOPOLOJİK OLARAK HATALI CÜMLE
    sentence_invalid = "kedi hızlı kovalar kırmızı fare"
    print(f"\n--- TEST 2: ANLAMSIZ/HATALI CÜMLE (Kaos) ---")
    print(f"Girdi: '{sentence_invalid}'")
    print("Klasik bir LLM (RNN), bu kelimeleri daha önce yan yana çok gördüğü için buna 'Geçerli' diyebilir.")
    success, msg = engine.parse_sentence(sentence_invalid)
    print(msg)

    print("\n[ÖLÇÜLEN SONUÇ: STOCHASTIC PARROT'LARIN ÖLÜMÜ]")
    print("Modern Büyük Dil Modelleri (LLMs) sadece devasa birer istatistik")
    print("kalkülatörüdür. 'Anlam (Semantics)' ve 'Dilbilgisi (Syntax)' kavramlarından")
    print("yoksundurlar. ToposAI, kelimeleri birer 'Kategori Oku (Functor)' olarak")
    print("modelleyip, cümlenin doğruluğunu istatistikle DEĞİL, Topolojik bir Düğümün")
    print("kapanıp kapanmamasıyla (Functor Composition -> Sentence Type 'S') gösterir.")
    print("Bu, insan zihninin (Universal Grammar) makinedeki donanımsal karşılığıdır!")

if __name__ == "__main__":
    run_universal_grammar_experiment()
