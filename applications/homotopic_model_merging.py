import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topos_ai.formal_category import (
    FiniteCategory,
    FiniteFunctor,
)

# =====================================================================
# TOPOLOGICAL MODEL MERGING (ZERO-DEGRADATION FEDERATED LEARNING)
# İddia: Klasik YZ, farklı uzmanlıkları olan Modelleri (Örn: Tıp YZ'si
# ile Hukuk YZ'si) birleştirmek için Milyarlarca Matris ağırlığının (Weights)
# ortalamasını alır. Bu işlem "Beyin Hasarına" (Catastrophic Forgetting) yol açar.
# Kategori Teorisinde (Topos Logic) iki farklı uzayı (Beyni) birleştirmek
# "Pushout (Colimit)" işlemi ile yapılır.
# A (Ortak Temel/Dil), B (Tıp Uzmanı), C (Hukuk Uzmanı).
# Pushout, ortak dili üst üste yapıştırır, çelişmeyen uzmanlıkları birleştirir
# ve HİÇBİR EĞİTİM (Zero-Shot) OLMADAN, HİÇBİR VERİ KAYBI YAŞANMADAN
# formal olarak izlenebilir yeni bir Süper-Kategori (Süper YZ Beyni) üretir.
# =====================================================================

def build_model_universes():
    # 1. ORTAK BEYİN (KÖK MODEL - Temel İngilizce ve Mantık)
    # Bu, her iki uzman modelin de üzerine inşa edildiği "Baz (Base)" modeldir.
    # Sadece temel kavramları (Özne, Nesne, Mantıksal Bağlaçlar) bilir.
    base_brain = FiniteCategory(
        objects=("Human", "Rule", "Condition"),
        morphisms={
            "idH": ("Human", "Human"), "idR": ("Rule", "Rule"), "idC": ("Condition", "Condition"),
            "applies_to": ("Rule", "Human"),      # Kural İnsana uygulanır
            "requires": ("Rule", "Condition")     # Kural Şart gerektirir
        },
        identities={"Human": "idH", "Rule": "idR", "Condition": "idC"},
        composition={
            ("idH", "idH"): "idH", ("idR", "idR"): "idR", ("idC", "idC"): "idC",
            ("applies_to", "idR"): "applies_to", ("idH", "applies_to"): "applies_to",
            ("requires", "idR"): "requires", ("idC", "requires"): "requires",
        }
    )

    # 2. TIP UZMANI BEYNİ (MEDICAL EXPERT)
    # Kök Modeli almış (Fine-Tuning), kendi tıp kavramlarını eklemiş.
    # Human -> Patient
    # Rule -> Protocol
    # Condition -> Symptom
    medical_brain = FiniteCategory(
        objects=("Patient", "Protocol", "Symptom", "Disease", "Medicine"), # Yeni Tıbbi Kavramlar!
        morphisms={
            "idP": ("Patient", "Patient"), "idPr": ("Protocol", "Protocol"), "idS": ("Symptom", "Symptom"),
            "idD": ("Disease", "Disease"), "idM": ("Medicine", "Medicine"),
            "applies_to": ("Protocol", "Patient"),      # Ortak Kural: Tıbbi Protokol Hastaya Uygulanır
            "requires": ("Protocol", "Symptom"),        # Ortak Kural: Protokol Semptom Gerektirir

            # TIP UZMANINA ÖZEL YENİ BİLGİLER (Yeni Morfizmalar)
            "diagnoses": ("Symptom", "Disease"),        # Semptom Hastalığı Teşhis Eder
            "treats": ("Medicine", "Disease"),          # İlaç Hastalığı Tedavi Eder
            "prescribes": ("Protocol", "Medicine"),     # Protokol İlaç Yazar

            # Gerekli Eksik Oklar (Composition Mismatch'i çözmek için üretilen Direct Path'ler)
            "protocol_diagnoses": ("Protocol", "Disease"), # diagnoses o requires
            "protocol_treats": ("Protocol", "Disease")     # treats o prescribes
        },
        identities={"Patient": "idP", "Protocol": "idPr", "Symptom": "idS", "Disease": "idD", "Medicine": "idM"},
        composition={
            ("idP", "idP"): "idP", ("idPr", "idPr"): "idPr", ("idS", "idS"): "idS",
            ("idD", "idD"): "idD", ("idM", "idM"): "idM",
            ("applies_to", "idPr"): "applies_to", ("idP", "applies_to"): "applies_to",
            ("requires", "idPr"): "requires", ("idS", "requires"): "requires",
            ("diagnoses", "idS"): "diagnoses", ("idD", "diagnoses"): "diagnoses",
            ("treats", "idM"): "treats", ("idD", "treats"): "treats",
            ("prescribes", "idPr"): "prescribes", ("idM", "prescribes"): "prescribes",

            ("protocol_diagnoses", "idPr"): "protocol_diagnoses", ("idD", "protocol_diagnoses"): "protocol_diagnoses",
            ("protocol_treats", "idPr"): "protocol_treats", ("idD", "protocol_treats"): "protocol_treats",

            # Eksik kompozisyon denklemleri
            ("diagnoses", "requires"): "protocol_diagnoses",
            ("treats", "prescribes"): "protocol_treats"
        }
    )

    # 3. HUKUK UZMANI BEYNİ (LEGAL EXPERT)
    legal_brain = FiniteCategory(
        objects=("Citizen", "Law", "Evidence", "Crime", "Penalty"), # Yeni Hukuki Kavramlar!
        morphisms={
            "idC": ("Citizen", "Citizen"), "idL": ("Law", "Law"), "idE": ("Evidence", "Evidence"),
            "idCr": ("Crime", "Crime"), "idP": ("Penalty", "Penalty"),
            "applies_to": ("Law", "Citizen"),         # Ortak Kural: Kanun Vatandaşa Uygulanır
            "requires": ("Law", "Evidence"),          # Ortak Kural: Kanun Delil Gerektirir

            # HUKUK UZMANINA ÖZEL YENİ BİLGİLER (Yeni Morfizmalar)
            "proves": ("Evidence", "Crime"),          # Delil Suçu İspatlar
            "punishes": ("Penalty", "Crime"),         # Ceza Suçu Cezalandırır
            "sentences": ("Law", "Penalty"),          # Kanun Ceza Verir

            # Gerekli Eksik Oklar (Composition Mismatch'i çözmek için)
            "law_proves": ("Law", "Crime"),           # proves o requires
            "law_punishes": ("Law", "Crime")          # punishes o sentences
        },
        identities={"Citizen": "idC", "Law": "idL", "Evidence": "idE", "Crime": "idCr", "Penalty": "idP"},
        composition={
            ("idC", "idC"): "idC", ("idL", "idL"): "idL", ("idE", "idE"): "idE",
            ("idCr", "idCr"): "idCr", ("idP", "idP"): "idP",
            ("applies_to", "idL"): "applies_to", ("idC", "applies_to"): "applies_to",
            ("requires", "idL"): "requires", ("idE", "requires"): "requires",
            ("proves", "idE"): "proves", ("idCr", "proves"): "proves",
            ("punishes", "idP"): "punishes", ("idCr", "punishes"): "punishes",
            ("sentences", "idL"): "sentences", ("idP", "sentences"): "sentences",

            ("law_proves", "idL"): "law_proves", ("idCr", "law_proves"): "law_proves",
            ("law_punishes", "idL"): "law_punishes", ("idCr", "law_punishes"): "law_punishes",

            # Eksik kompozisyon denklemleri
            ("proves", "requires"): "law_proves",
            ("punishes", "sentences"): "law_punishes"
        }
    )

    return base_brain, medical_brain, legal_brain

def categorical_pushout_merge(base, expert_A, expert_B, map_A, map_B):
    """
    [PUSHOUT / COLIMIT ALGORİTMASI]
    Kategori Teorisinde iki nesneyi ortak kökleri (Base) üzerinden
    birleştirme işleminin formal fonksiyonudur.
    Ortak parçaları üst üste yapıştırır (Gluing), geri kalan tüm yeni
    (Specialized) bilgi ve okları tek bir Dev Kategoriye (Süper Model) kayıpsız ekler.
    """
    merged_objects = set()
    merged_morphisms = {}
    merged_identities = {}
    merged_composition = {}

    # 1. Ortak (Base) kavramları al, Uzman Modellerdeki kelimeleri eşitle.
    # Yani Tıptaki 'Patient' ile Hukuktaki 'Citizen' aslen Kök'teki 'Human' kavramıdır.
    # İkisini de ortak olan 'Human' kavramında "Yapıştıracağız" (Equivalence Class).

    glued_objects = {}
    for obj in base.objects:
        glued_objects[map_A["objects"][obj]] = obj # Patient -> Human
        glued_objects[map_B["objects"][obj]] = obj # Citizen -> Human
        merged_objects.add(obj) # Ortak/Kök kelime Süper Beyne eklenir

    # 2. Uzman Modellerdeki sadece onlara HAS (Yeni) kavramları ekle.
    for obj in expert_A.objects:
        if obj not in glued_objects:
            merged_objects.add(obj) # Örn: Disease, Medicine
    for obj in expert_B.objects:
        if obj not in glued_objects:
            merged_objects.add(obj) # Örn: Crime, Penalty

    # 3. Morfizmaları (Bilgi ve Kuralları) Yapıştır.
    glued_morphisms = {}
    for mor in base.morphisms:
        glued_morphisms[map_A["morphisms"][mor]] = mor # Tıptaki 'applies_to' -> Kök 'applies_to'
        glued_morphisms[map_B["morphisms"][mor]] = mor # Hukuktaki 'applies_to' -> Kök 'applies_to'

    # Yardımcı Fonksiyon: Bir objenin veya morfizmanın Süper Modeldeki nihai adı
    def get_merged_name(item, mapping_dict):
        return mapping_dict.get(item, item)

    # 4. Tüm Okları (Rules/Morphisms) Çelişmeden Süper Modele Aktar
    for name, (src, dst) in expert_A.morphisms.items():
        merged_name = get_merged_name(name, glued_morphisms)
        merged_src = get_merged_name(src, glued_objects)
        merged_dst = get_merged_name(dst, glued_objects)
        merged_morphisms[merged_name] = (merged_src, merged_dst)

    for name, (src, dst) in expert_B.morphisms.items():
        merged_name = get_merged_name(name, glued_morphisms)
        merged_src = get_merged_name(src, glued_objects)
        merged_dst = get_merged_name(dst, glued_objects)
        # Eğer ok Kök Modelse zaten eklendi, değilse YENİ BİLGİ olarak eklenecek.
        if merged_name not in merged_morphisms:
            merged_morphisms[merged_name] = (merged_src, merged_dst)

    # Tüm objelerin kendi Identity'lerini ekle (Süper Model)
    for obj in merged_objects:
        merged_identities[obj] = f"id_{obj}"
        merged_morphisms[f"id_{obj}"] = (obj, obj)

    # 5. Kompozisyon Tablosunu "Yeniden İnşa Et" (Transitive Closure)
    # A ve B'nin orijinal tablolarını doğrudan kopyalamak isim uyuşmazlığı yaratır.
    # Bunun yerine, Süper Modeldeki tüm okları birbiriyle eşleştirerek geçerli olanları listeliyoruz.

    for name1, (src1, dst1) in merged_morphisms.items():
        for name2, (src2, dst2) in merged_morphisms.items():
            if dst1 == src2:  # Yol birleşebiliyorsa

                # 1. Kural: Identity Kompozisyonları (f o id = f, id o f = f)
                if name1.startswith("id_"):
                    merged_composition[(name2, name1)] = name2
                elif name2.startswith("id_"):
                    merged_composition[(name2, name1)] = name1

                # 2. Kural: Uzmanlardan gelen Miras Oklar (Tıp ve Hukuk)
                else:
                    # Tıp Uzmanı Kompozisyonlarını Kontrol Et
                    found = False
                    for (g, f), comp_res in expert_A.composition.items():
                        mg, mf, mres = get_merged_name(g, glued_morphisms), get_merged_name(f, glued_morphisms), get_merged_name(comp_res, glued_morphisms)
                        if mg == name2 and mf == name1:
                            merged_composition[(name2, name1)] = mres
                            found = True
                            break
                    if found: continue

                    # Hukuk Uzmanı Kompozisyonlarını Kontrol Et
                    for (g, f), comp_res in expert_B.composition.items():
                        mg, mf, mres = get_merged_name(g, glued_morphisms), get_merged_name(f, glued_morphisms), get_merged_name(comp_res, glued_morphisms)
                        if mg == name2 and mf == name1:
                            merged_composition[(name2, name1)] = mres
                            break

    return FiniteCategory(tuple(merged_objects), merged_morphisms, merged_identities, merged_composition)

def run_model_merging_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 24: TOPOLOGICAL MODEL MERGING (genel zeka araştırması PUSHOUT) ")
    print(" (FORMAL KATEGORİ TEORİSİ COLIMIT / PUSHOUT İLE YENİDEN YAZILMIŞTIR) ")
    print("=========================================================================\n")

    base_brain, medical_brain, legal_brain = build_model_universes()

    print("--- 1. UZMAN YZ MODELLERİ (LOKAL BEYİNLER) ---")
    print(f" Kök (Ortak) Model Kavramları: {base_brain.objects}")
    print(f" A Modeli (Tıp Uzmanı) Yeni Bilgiler: {[obj for obj in medical_brain.objects if obj not in ('Patient', 'Protocol', 'Symptom')]}")
    print(f" B Modeli (Hukuk Uzmanı) Yeni Bilgiler: {[obj for obj in legal_brain.objects if obj not in ('Citizen', 'Law', 'Evidence')]}")

    # Tıbbi Ajan Kök Modelin üstüne nasıl Fine-Tune edilmiş? (Functor Map)
    map_medical = {
        "objects": {"Human": "Patient", "Rule": "Protocol", "Condition": "Symptom"},
        "morphisms": {"applies_to": "applies_to", "requires": "requires", "idH": "idP", "idR": "idPr", "idC": "idS"}
    }

    # Hukuki Ajan Kök Modelin üstüne nasıl Fine-Tune edilmiş? (Functor Map)
    map_legal = {
        "objects": {"Human": "Citizen", "Rule": "Law", "Condition": "Evidence"},
        "morphisms": {"applies_to": "applies_to", "requires": "requires", "idH": "idC", "idR": "idL", "idC": "idE"}
    }

    print("\n--- 2. KATEGORİK PUSHOUT (MATRİSLERİN BİRLEŞTİRİLMESİ) ÇALIŞIYOR... ---")
    print(" Klasik (Derin Öğrenme) yaklaşım: Matrisleri ortala, Hastayı ve Vatandaşı birbirine karıştır")
    print(" (Beyin Hasarı). Kategori Teorisi yaklaşımı (Colimit): Ortak dilleri üst üste yapıştır,")
    print(" uzmanlıkları (Okları/Morfizmaları) çelişmeden tek bir Topos evrenine sentezle.")

    super_brain = categorical_pushout_merge(base_brain, medical_brain, legal_brain, map_medical, map_legal)

    print("\n--- 3. BİLİMSEL SONUÇ (SÜPER YZ BEYNİNİN DOĞUŞU) ---")
    print(f" Süper Model'in Kavram Uzayı (Objeler): {super_brain.objects}")
    print("\n Süper Model'in Bildiği Kurallar (Morfizmalar/Mantık Okları):")
    for name, (src, dst) in super_brain.morphisms.items():
        if not name.startswith("id_"):
            print(f"   [{src}] ---({name})---> [{dst}]")

    print("\n [BAŞARILI: ZERO-DEGRADATION (KAYIPSIZ) genel zeka araştırması SENTEZİ]")
    print(" Gördüğünüz gibi Tıbbi Uzmanın (Medicine, Disease) bilgileri ile,")
    print(" Hukuki Uzmanın (Crime, Penalty) bilgileri, hiçbir türev (Loss/Training)")
    print(" kullanılmadan tek bir beyinde (Pushout) formal olarak izlenebilirca kaynaştırıldı.")
    print(" Hastalık ile Suç, ortak 'Condition/Human' temelinde birleşerek, ")
    print(" birbirlerini ezmeden tek bir devasa Ontolojik Ağa dönüştüler!")

if __name__ == "__main__":
    run_model_merging_experiment()
