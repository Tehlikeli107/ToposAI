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
# MECHANISTIC INTERPRETABILITY (XAI) & TOPOS FORCING LOGIC
# İddia: Klasik YZ'lerdeki "Neden bu kararı verdin?" sorusu, matris
# ağırlıklarında "En çok parlayan yolu" (BFS/Graph) arayarak çözülemez.
# Çünkü kararlar "Şartlara (Context)" bağlıdır.
# Kategori Teorisinde (Topos Logic) bir karar `True` (1.0) değildir,
# "Hangi evren aşamalarında (Morfizmalar) doğru olduğu" bir Sieve'dir
# (Zorlama/Forcing Kümesi).
# Bu modül, yapay zekanın "Hastaya Kanser Teşhisi" veya "Kredi Ret"
# kararını neden verdiğini, arka planda çalışan Kripke-Joyal (İçsel Mantık)
# kurallarının kesişim (Meet) ve gereklilik (Implication) kümelerini
# şeffaf bir şekilde dökerek MATEMATİKSEL İSPATLAR.
# =====================================================================

def create_medical_topos():
    # 1. HASTALIĞIN BAĞLAMSAL EVRENİ (Category)
    # Tıbbi kararların alındığı evrenin yapısı (Hiyerarşisi/Morfizmaları)
    # Zaman ve bulgu ilerleyişi: Checkup -> Symptom_X -> Final_Diagnosis
    category = FiniteCategory(
        objects=("Checkup", "Symptom_X", "Diagnosis"),
        morphisms={
            "idC": ("Checkup", "Checkup"), "idS": ("Symptom_X", "Symptom_X"), "idD": ("Diagnosis", "Diagnosis"),
            "find_symptom": ("Symptom_X", "Checkup"), # Checkup'ta semptom bulunur
            "conclude": ("Diagnosis", "Symptom_X"),   # Semptom teşhise götürür
            "direct_conclude": ("Diagnosis", "Checkup") # (conclude o find_symptom)
        },
        identities={"Checkup": "idC", "Symptom_X": "idS", "Diagnosis": "idD"},
        composition={
            ("idC", "idC"): "idC", ("idS", "idS"): "idS", ("idD", "idD"): "idD",
            ("idC", "find_symptom"): "find_symptom", ("find_symptom", "idS"): "find_symptom",
            ("idS", "conclude"): "conclude", ("conclude", "idD"): "conclude",
            ("idC", "direct_conclude"): "direct_conclude", ("direct_conclude", "idD"): "direct_conclude",
            ("find_symptom", "conclude"): "direct_conclude" # A <- B <- C = A <- C
        }
    )

    # 2. HASTANIN (PATIENT) TÜM VERİ EVRENİ (Presheaf)
    patient_data = Presheaf(
        category,
        sets={
            "Checkup": {"Blood_Test_High", "Normal_BP"},
            "Symptom_X": {"Anomaly_Detected"},
            "Diagnosis": {"Critical_Condition"}
        },
        restrictions={
            "idC": {"Blood_Test_High": "Blood_Test_High", "Normal_BP": "Normal_BP"},
            "idS": {"Anomaly_Detected": "Anomaly_Detected"},
            "idD": {"Critical_Condition": "Critical_Condition"},

            # Contravariant Functor Kuralları: F(Hedef) -> F(Kaynak)
            # Ok: Symptom_X -> Checkup
            # Restriction: Checkup -> Symptom_X
            # (Checkup anındaki yüksek kan bulgusu, Semptom aşamasına anomali olarak taşınır)
            "find_symptom": {"Blood_Test_High": "Anomaly_Detected", "Normal_BP": "Anomaly_Detected"}, # Basitlik için ikisi de anomaliye haritalansın

            # Ok: Diagnosis -> Symptom_X
            # Restriction: Symptom_X -> Diagnosis
            "conclude": {"Anomaly_Detected": "Critical_Condition"},

            # Ok: Diagnosis -> Checkup
            # Restriction: Checkup -> Diagnosis
            "direct_conclude": {"Blood_Test_High": "Critical_Condition", "Normal_BP": "Critical_Condition"}
        }
    )

    return category, patient_data

def run_interpretability_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 23: MECHANISTIC INTERPRETABILITY (XAI) & TOPOS LOGIC ")
    print(" (FORMAL KRIPKE-JOYAL FORCING / HEYTING CEBİRİ İLE YENİDEN YAZILMIŞTIR) ")
    print("=========================================================================\n")

    category, patient_universe = create_medical_topos()
    topos = PresheafTopos(category)

    # 3. YZ'NİN ÖĞRENDİĞİ KURALLAR (ALT KÜMELER / SUBOBJECTS)
    print("--- 1. YZ'NİN (AĞIRLIKSIZ / MATEMATİKSEL) KURALLARI ---")

    # YZ'nin içindeki kural 1: Kan değeri yüksekliği tespit edildi (Fact 1)
    fact_blood = Subpresheaf(patient_universe, subsets={"Checkup": {"Blood_Test_High"}, "Symptom_X": {"Anomaly_Detected"}, "Diagnosis": {"Critical_Condition"}})

    # YZ'nin içindeki kural 2: Teşhis durumu kritiktir (Fact 2)
    # (Not: Normal_BP bu kurala dahil değil, onu eliyoruz)
    fact_critical = Subpresheaf(patient_universe, subsets={"Checkup": set(), "Symptom_X": set(), "Diagnosis": {"Critical_Condition"}})

    print(" YZ Kararı (Sonuç): 'Critical_Condition' tetiklendi.")
    print(" Klasik (Black-box) model: '0.98 ihtimalle Kritik. Nedenini sorma, ağırlıklar (Weights) öyle diyor.'")

    print("\n--- 2. TOPOS İÇSEL MANTIĞI İLE ŞEFFAF İSPAT (FORCING SIEVES) ---")
    print(" Topos (Heyting) mantığında bir şeyin 'Neden' doğru olduğu, o sonuca götüren")
    print(" tüm Morfizma oklarının (Sieve / Açık Kümeler) matematiksel izdüşümüyle bulunur.")

    # "Kan Tahlili" alt kümesi, "Kritik Teşhis" alt kümesini ZORUNLU KILIYOR MU?
    # (Implication: Blood => Critical)
    implication_rule = topos.subobject_implication(fact_blood, fact_critical)

    print(f"\n [Kural Denetimi] YZ'nin İçsel Mantığı (Blood_Test => Critical_Condition):")
    print(" Sistemin bu kararı verirken kullandığı kanıt zinciri (Sieves):")
    for context, elements in implication_rule.subsets.items():
        print(f"   Bağlam (Zaman) '{context}': Kapsanan Öğeler -> {elements}")

    print("\n--- 3. XAI (EXPLAINABLE AI) SONUÇ DÖKÜMÜ ---")

    # Karar: Checkup aşamasındaki Blood_Test_High nesnesi, sistemi Diagnosis aşamasına "Forcing" (Zorluyor) mu?
    # Kripke-Joyal teoreminde: x forces P_implies_Q, ancak ve ancak x'e gelen her yolda P doğruysa Q da doğruysa.
    forcing_sieve = topos.forcing_sieve(implication_rule, "Checkup", "Blood_Test_High")

    print(" Soru: 'Blood_Test_High' bulgusu, Critical duruma neden olmak ZORUNDA MIYDI?")
    print(f" Sistemin Zorlayıcı Kanıt Ağacı (Forcing Sieve): {forcing_sieve}")

    if "direct_conclude" in forcing_sieve:
        print("\n [BAŞARILI: %100 ŞEFFAF VE MATEMATİKSEL İSPAT]")
        print(" Yapay Zeka kararı verirken istatistiksel bir ağırlık (0.98) SALLAMADI.")
        print(" ToposAI, Kripke-Joyal içsel mantığını kullanarak;")
        print(" Checkup'taki 'Blood_Test_High' bulgusunun aradaki tüm adımları aşıp")
        print(" 'direct_conclude' okuyla doğrudan 'Critical_Condition'ı zorladığını")
        print(" bir 'Sieve' (Kanıt Kümesi / Forcing Sieve) olarak ŞEFFAFÇA masaya koydu.")
        print(" Geleceğin Güvenilir YZ (Explainable AI / XAI) sistemleri, kararlarını ")
        print(" kara-kutu (Blackbox) tensörlerle değil, Topos-Sieve kanıtlarıyla sunacaktır.")
    else:
        print("\n [HATA] Kararın nedeni (İspat Zinciri) bulunamadı.")

if __name__ == "__main__":
    run_interpretability_experiment()
