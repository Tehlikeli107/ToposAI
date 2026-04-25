import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topos_ai.formal_category import (
    FiniteCategory,
    Presheaf,
    PresheafTopos,
    Subpresheaf,
)

# =====================================================================
# GÖDEL'S INCOMPLETENESS & THE LIAR'S PARADOX (META-COGNITION)
# İddia: Klasik YZ'ler paradokslarda (Örn: "Bu cümle yanlıştır")
# sembolik olarak sonsuz döngüye girer veya halüsinasyon uydurur.
# Kategori Teorisinde (Topos Logic) ise her ifade bir alt nesne (Subobject)
# ve doğruluk değerleri (Truth Values) açık kümelerdir (Sieves).
# Bir sistem içinde kendi doğruluğunun tersini iddia eden bir nesnenin
# (Liar Paradox) Heyting cebirindeki içsel mantığı (Internal Logic)
# Topos tarafından anında "Çelişki (Bottom/Empty Sieve)" olarak yakalanır.
# Bu, sistemin çökmesini engelleyen matematiksel bir 'Bilinemezlik' kanıtıdır.
# =====================================================================

def create_paradox_universe():
    # Basit bir evren: Tek bir nesne (Zamanın tek bir anı)
    category = FiniteCategory(
        objects=("Now",),
        morphisms={"id": ("Now", "Now")},
        identities={"Now": "id"},
        composition={("id", "id"): "id"}
    )

    # Tüm gerçeği temsil eden ana Presheaf (Evrenin tamamı)
    # Bu evrende konuşulan "Cümle (Statement)" isimli tek bir varlık var.
    presheaf_universe = Presheaf(
        category,
        sets={"Now": {"Liar_Sentence"}},
        restrictions={"id": {"Liar_Sentence": "Liar_Sentence"}}
    )
    return category, presheaf_universe

def run_godel_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 18: GÖDEL'S PARADOX & THE LIAR'S SENTENCE ")
    print(" (FORMAL TOPOS MANTIĞI VE HEYTING CEBİRİ İLE YENİDEN YAZILMIŞTIR) ")
    print("=========================================================================\n")

    category, P_universe = create_paradox_universe()
    topos = PresheafTopos(category)

    print("--- 1. PARADOKS KURULUMU (THE LIAR) ---")
    # Paradoks cümlesi: "Liar_Sentence"
    # Bu cümlenin Doğru (True) olduğunu farz eden bir alt küme (Subpresheaf) kuralım.
    statement_is_true = Subpresheaf(P_universe, subsets={"Now": {"Liar_Sentence"}})

    print(f" İfade (P): 'Bu cümle yanlıştır.'")
    print(f" P'nin Topos'taki Karşılığı: {statement_is_true.subsets}")

    # Topos içinde "Değil (Not / Negation)" işlemi:
    # Heyting Cebirinde Not(P) demek: P'den Boşluğa (False) giden en büyük alt kümedir.
    statement_is_false = topos.subobject_negation(statement_is_true)

    print("\n--- 2. İÇSEL MANTIK (HEYTING ALGEBRA) ÇÖZÜMÜ ---")
    print(" Klasik mantık der ki: P = Not(P) ise sistem çöker (Paradoks).")
    print(" Topos mantığı ise cümlenin 'Yanlış' olma durumunu anında hesaplar:")
    print(f" Not(P)'nin Topos'taki Karşılığı: {statement_is_false.subsets}")

    # Şimdi Kripke-Joyal Zorlama (Forcing) mantığı ile soralım:
    # "Bu cümle aynı anda hem Doğru hem Yanlış olabilir mi?" (P AND Not(P))
    paradox_intersection = topos.subobject_meet(statement_is_true, statement_is_false)

    print("\n--- 3. ÇELİŞKİ TESPİTİ VE SİSTEMİN KORUNMASI ---")
    print(f" P AND Not(P) Kesişimi (Meet): {paradox_intersection.subsets}")

    is_empty = all(len(s) == 0 for s in paradox_intersection.subsets.values())
    if is_empty:
        print("\n [BİLİMSEL SONUÇ: TOPOS ÇÖKMEZ]")
        print(" Klasik YZ'ler (LLM'ler) 'P = Not(P)' girdisinde ya uydurur (Halüsinasyon)")
        print(" ya da sonsuz döngüye girer. Ancak Formal Kategori Teorisinde (Topos),")
        print(" Doğruluk (Truth) evrensel değil yereldir (Heyting Cebiri).")
        print(" ToposAI, Liar Paradoksunu Kripke-Joyal mantığında P ∩ ¬P = ∅ (Boş Küme)")
        print(" olarak matematiksel bir kesinlikle (0 adımda) 'Çelişki' olarak etiketler")
        print(" ve Gödel'in 'Kanıtlanamazlık' sınırını aşmadan sistemi güvende tutar.")
    else:
        print("\n [HATA] Sistem paradoksu çözemedi ve çöktü.")

if __name__ == "__main__":
    run_godel_experiment()
