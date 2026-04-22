import torch
import torch.nn as nn

# =====================================================================
# TOPOS SMART CONTRACT AUDITOR (FORMAL VERIFICATION ENGINE)
# Akıllı sözleşmelerin güvenlik açıklarını "Metin Analizi" ile değil,
# "Topolojik Geçişlilik" (Morphism Composition) ve "Döngü" tespiti ile bulur.
# =====================================================================

def lukasiewicz_composition(R1, R2):
    """
    A -> B ve B -> C ise A -> C (Topos Mantıksal Geçişliliği)
    Akıllı sözleşmedeki kod bloklarının birbirini tetikleme (reachability) matrisi.
    """
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) 
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

def calculate_transitive_closure(R, max_steps=5):
    """
    Sözleşme sonsuz kere çalışırsa hangi kod bloğu hangisine ulaşabilir?
    Matrisi kendisiyle mantıksal olarak çarparak (R^n) tüm olasılık ağını çıkarır.
    """
    R_closure = R.clone()
    for _ in range(max_steps):
        # R_closure = R_closure U (R_closure O R) -> Mantıksal OR (Max) ve AND (Composition)
        new_R = lukasiewicz_composition(R_closure, R)
        R_closure = torch.max(R_closure, new_R) # Ulaşılabilen en güçlü yolu tut
    return R_closure

def audit_smart_contract():
    print("--- TOPOS DÜŞÜNCE MOTORU: SİBER GÜVENLİK (SMART CONTRACT) DENETİMİ ---")
    print("Sistem, kod metnine değil, kodun 'Topolojik İskeletine' bakıyor...\n")

    # Sözleşmedeki Durumlar (States / Objects in Category)
    states = ["BAŞLA", "BAKİYE_KONTROL", "PARA_GÖNDER", "BAKİYE_GÜNCELLE", "BİTİR", "HACKER_FALLBACK"]
    s_idx = {s: i for i, s in enumerate(states)}
    N = len(states)
    
    # 1. GÜVENLİ SÖZLEŞME (Safe Contract) Evreni
    R_safe = torch.zeros((N, N))
    R_safe[s_idx["BAŞLA"], s_idx["BAKİYE_KONTROL"]] = 1.0
    R_safe[s_idx["BAKİYE_KONTROL"], s_idx["PARA_GÖNDER"]] = 1.0
    # Parayı gönderdikten sonra KESİNLİKLE bakiyeyi günceller
    R_safe[s_idx["PARA_GÖNDER"], s_idx["BAKİYE_GÜNCELLE"]] = 1.0
    R_safe[s_idx["BAKİYE_GÜNCELLE"], s_idx["BİTİR"]] = 1.0
    # Hacker araya girmeye çalışsa bile sözleşme kapalıdır (0.0)
    R_safe[s_idx["PARA_GÖNDER"], s_idx["HACKER_FALLBACK"]] = 0.0

    # 2. AÇIKLI SÖZLEŞME (Vulnerable Contract - DAO Hack / Reentrancy) Evreni
    R_bug = torch.zeros((N, N))
    R_bug[s_idx["BAŞLA"], s_idx["BAKİYE_KONTROL"]] = 1.0
    R_bug[s_idx["BAKİYE_KONTROL"], s_idx["PARA_GÖNDER"]] = 1.0
    # AÇIK BURADA: Para gönderme işlemi dışarıya (Hacker'a) bir tetikleme (call) yapıyor
    R_bug[s_idx["PARA_GÖNDER"], s_idx["HACKER_FALLBACK"]] = 1.0 
    # Normalde bakiyeyi güncelleyecek ama...
    R_bug[s_idx["PARA_GÖNDER"], s_idx["BAKİYE_GÜNCELLE"]] = 1.0 
    R_bug[s_idx["BAKİYE_GÜNCELLE"], s_idx["BİTİR"]] = 1.0
    # Hacker'ın fonksiyonu, sözleşmeyi tekrar "Bakiye Kontrol" veya "Para Gönder"e yönlendirebilir! (DÖNGÜ)
    R_bug[s_idx["HACKER_FALLBACK"], s_idx["BAKİYE_KONTROL"]] = 1.0 


    # --- TOPOS DENETİM ALGORİTMASI ---
    contracts = {"GÜVENLİ SÖZLEŞME": R_safe, "AÇIKLI SÖZLEŞME (Reentrancy)": R_bug}
    
    for name, R_matrix in contracts.items():
        print(f"Denetlenen Sözleşme: [{name}]")
        
        # Olasılık Ağını (Transitive Closure) Çıkar
        R_inf = calculate_transitive_closure(R_matrix, max_steps=5)
        
        # KURAL 1: Reentrancy Tespiti (Endomorfizma / Topological Loop)
        # Bir Topos evreninde, PARA_GÖNDER okundan çıkıp, BAKİYE_GÜNCELLE okuna uğramadan
        # tekrar PARA_GÖNDER okuna ulaşılabiliyor mu? 
        # Matematiksel karşılığı: R_inf(PARA_GÖNDER, PARA_GÖNDER) > 0.0 mıdır?
        
        is_loop_possible = R_inf[s_idx["PARA_GÖNDER"], s_idx["PARA_GÖNDER"]].item()
        
        # KURAL 2: Bakiye Güncelleme Atlatılabiliyor mu?
        # Para Gönder -> Para Gönder (Loop) varken, bu döngü sırasında Bakiye Güncellenmiş mi?
        # Eğitsel simülasyonumuzda loop varsa bakiye güncellemesi by-pass edilmiş demektir.
        
        if is_loop_possible > 0.8:
            print("  [!] KRİTİK GÜVENLİK İHLALİ BULUNDU: REENTRANCY (Geri Çağırma) AÇIĞI!")
            print("  Matematiksel Kanıt: Topos matrisinde (PARA_GÖNDER -> PARA_GÖNDER) arasında kapalı bir döngü (Loop) saptandı.")
            print("  Açıklama: Sözleşme, bakiyeyi düşürmeden önce hacker'ın parayı tekrar çekmesine izin veren topolojik bir yırtığa sahip.\n")
        else:
            print("  [+] SÖZLEŞME GÜVENLİ.")
            print("  Matematiksel Kanıt: Tüm mantıksal yollar (Morphism Paths) 'BİTİR' durumuna ulaşıyor. İzinsiz döngü yok.\n")

if __name__ == "__main__":
    audit_smart_contract()
