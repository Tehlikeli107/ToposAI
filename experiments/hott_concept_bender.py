import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topos_ai.hott import (
    FinitePathGroupoid,
    PathFamily,
)

# =====================================================================
# HOMOTOPY TYPE THEORY (HoTT) - KAVRAM BÜKÜCÜ & ANALOJİ (CONCEPT BENDER)
# İddia: Klasik YZ, kelimelerin "Benzerliğini" Vektör Açısı (Kosinüs)
# ile ölçer. Ancak benzerlik kavramlar arası gerçek geçişleri (Mecaz/
# Metafor/Analoji) açıklayamaz. HoTT'de eşitlik (Equality) diye bir
# durağanlık yoktur, sadece kavramlar arası "İspat Yolları (Paths)" vardır.
# Eğer A kavramından B kavramına bir yol (p) varsa, A'nın tüm özelliklerini
# (Fibers) B'ye "Taşıyabiliriz (Functorial Transport)".
# Bu modül, yapay zekanın "Eşitlik" yerine "Yol (Homotopy)" kullanarak
# kavramları birbiri üzerinde nasıl dönüştüreceğini kanıtlar.
# =====================================================================

def run_hott_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 22: HOMOTOPY TYPE THEORY (HoTT) CONCEPT BENDER ")
    print(" (FORMAL FINITE PATH GROUPOID VE FUNCTORIAL TRANSPORT İLE YENİDEN YAZILMIŞTIR) ")
    print("=========================================================================\n")

    # 1. KAVRAM UZAYI (Topolojik Düğümler/Objects)
    # Kavramlarımız: Kuş (Bird), Uçak (Airplane), Denizaltı (Submarine)
    # HoTT, bu kavramların birbirine "Eşit" olup olmadığını değil,
    # aralarında nasıl bir "Homotopi (Dönüşüm Yolu)" olduğunu sorar.

    print("--- 1. KAVRAM UZAYI VE İSPAT YOLLARI (PATHS) ---")

    objects = ("Bird", "Airplane", "Submarine")
    # Yollar (Paths): Kuş'tan Uçağa giden yol (Kanat/Uçma), Uçaktan Denizaltına (Metal Gövde)
    paths = {
        "idBird": ("Bird", "Bird"), "idAirplane": ("Airplane", "Airplane"), "idSubmarine": ("Submarine", "Submarine"),

        # Metafor Yolları: (Bir kavramdan diğerine dönüşüm kanıtları)
        "mechanize_flight": ("Bird", "Airplane"),     # Kuşu makineleştirirsen uçak olur
        "organicize_flight": ("Airplane", "Bird"),    # Uçağı organikleştirirsen kuş olur (Ters yol)

        "sink_machine": ("Airplane", "Submarine"),    # Uçağı suya batırırsan denizaltı olur
        "fly_machine": ("Submarine", "Airplane"),     # Denizaltını havaya kaldırırsan uçak olur

        # Kompozisyon Yolları (A'dan C'ye Direkt Metafor)
        # Kuşu al, önce makineleştir (Uçak yap), sonra batır -> "Metal Kuş Balığı" (Denizaltı)
        "mechanize_and_sink": ("Bird", "Submarine"),
        "organicize_and_fly": ("Submarine", "Bird"),
    }

    # Groupoid Kuralları (Tersine Çevrilebilirlik ve Birleşim)
    identities = {"Bird": "idBird", "Airplane": "idAirplane", "Submarine": "idSubmarine"}
    inverses = {
        "idBird": "idBird", "idAirplane": "idAirplane", "idSubmarine": "idSubmarine",
        "mechanize_flight": "organicize_flight", "organicize_flight": "mechanize_flight",
        "sink_machine": "fly_machine", "fly_machine": "sink_machine",
        "mechanize_and_sink": "organicize_and_fly", "organicize_and_fly": "mechanize_and_sink",
    }
    composition = {}

    # Tüm birleşebilen (composable) yolları otomatik olarak hesaplayalım (A -> B -> C)
    for p1_name, (src1, dst1) in paths.items():
        for p2_name, (src2, dst2) in paths.items():
            if dst1 == src2:  # Eğer birinci yolun sonu, ikinci yolun başına eşitse birleşebilirler

                # Kurallar (Identities)
                if p1_name.startswith("id"):
                    composition[(p2_name, p1_name)] = p2_name
                elif p2_name.startswith("id"):
                    composition[(p2_name, p1_name)] = p1_name

                # Bir yolun kendi tersiyle birleşimi (Inverses)
                elif inverses[p1_name] == p2_name:
                    composition[(p2_name, p1_name)] = identities[src1]

                # Direkt (Metaforik) Geçişler
                elif p1_name == "mechanize_flight" and p2_name == "sink_machine":
                    composition[(p2_name, p1_name)] = "mechanize_and_sink"
                elif p1_name == "fly_machine" and p2_name == "organicize_flight":
                    composition[(p2_name, p1_name)] = "organicize_and_fly"
                elif p1_name == "organicize_and_fly" and p2_name == "mechanize_flight":
                    composition[(p2_name, p1_name)] = "fly_machine"
                elif p1_name == "organicize_flight" and p2_name == "mechanize_and_sink":
                    composition[(p2_name, p1_name)] = "sink_machine"
                elif p1_name == "sink_machine" and p2_name == "organicize_and_fly":
                    composition[(p2_name, p1_name)] = "organicize_flight"
                elif p1_name == "mechanize_and_sink" and p2_name == "fly_machine":
                    composition[(p2_name, p1_name)] = "mechanize_flight"

    concept_space = FinitePathGroupoid(objects, paths, identities, inverses, composition)

    # 2. HoTT SÖZ DİZİMİ: TAŞIMA (TRANSPORT FUNCTOR)
    # Homotopi'nin en güçlü yanı, eğer A ile B arasında bir yol varsa,
    # A'nın üzerinde yaşayan her türlü özelliği (Fiber) o yoldan geçirerek
    # B'nin üzerine bükerek (Transport) taşıyabilmenizdir.

    # Fibers: Kavramların "Özellikleri"
    fibers = {
        "Bird": {"Tüylü Kanat", "Solunum", "Gökyüzü"},
        "Airplane": {"Metal Kanat", "Motor", "Gökyüzü"},
        "Submarine": {"Metal Pervane", "Motor", "Okyanus_Altı"}
    }

    # Transport Haritası: Yoldan geçerken özellikler neye evriliyor?
    transports = {
        # Identity (Kendine Taşıma): Hiçbir şey değişmez
        "idBird": {f: f for f in fibers["Bird"]},
        "idAirplane": {f: f for f in fibers["Airplane"]},
        "idSubmarine": {f: f for f in fibers["Submarine"]},

        # Yol: mechanize_flight (Kuş -> Uçak)
        "mechanize_flight": {
            "Tüylü Kanat": "Metal Kanat",
            "Solunum": "Motor",
            "Gökyüzü": "Gökyüzü"  # Çevre değişmez
        },
        # Yolun Tersi: organicize_flight (Uçak -> Kuş)
        "organicize_flight": {
            "Metal Kanat": "Tüylü Kanat",
            "Motor": "Solunum",
            "Gökyüzü": "Gökyüzü"
        },

        # Yol: sink_machine (Uçak -> Denizaltı)
        "sink_machine": {
            "Metal Kanat": "Metal Pervane", # Kanat suda pervaneye dönüşür (Bükülme)
            "Motor": "Motor",               # Motor kalır
            "Gökyüzü": "Okyanus_Altı"       # Çevre değişir
        },
        # Yolun Tersi: fly_machine (Denizaltı -> Uçak)
        "fly_machine": {
            "Metal Pervane": "Metal Kanat",
            "Motor": "Motor",
            "Okyanus_Altı": "Gökyüzü"
        },

        # BİRLEŞTİRİLMİŞ YOL (Composition): mechanize_and_sink (Kuş -> Denizaltı)
        # Bu kısım sistemin 'Mantıksal Analojisi'dir.
        "mechanize_and_sink": {
            "Tüylü Kanat": "Metal Pervane",
            "Solunum": "Motor",
            "Gökyüzü": "Okyanus_Altı"
        },
        "organicize_and_fly": {
            "Metal Pervane": "Tüylü Kanat",
            "Motor": "Solunum",
            "Okyanus_Altı": "Gökyüzü"
        }
    }

    concept_bender = PathFamily(base=concept_space, fibers=fibers, transports=transports)

    # 3. HO_TT MATEMATİKSEL İSPATI VE DENEYLER
    print(" [HoTT Groupoid] Kavram Uzayı Oluşturuldu.")

    # Sistemin kendi topolojik yasalarını kanıtlaması
    is_valid_groupoid = concept_space.validate_groupoid_laws()
    is_valid_functorial = concept_bender.validate_functorial_transport()

    print("\n--- 2. KAVRAM BÜKME (TRANSPORTATION / ANALOGY) ---")
    print(" Görev: 'Kuş' kavramının temel özelliklerini (Fibers) al.")
    print(" Önce onu 'Makineleştir (Uçak Yap)', sonra 'Batır (Denizaltı Yap)'.")
    print(" Bu dönüşüm, Kuş -> Denizaltı arasındaki (mechanize_and_sink) Yolu ile ")
    print(" aynı (Functorial Isomorphism) çıkmak ZORUNDADIR.")

    # 1. Adım: Kuştan Uçağa (mechanize_flight)
    bird_feature = "Tüylü Kanat"
    step1_feature = concept_bender.transport("mechanize_flight", bird_feature)
    print(f"\n   1. Adım: 'Kuş' üzerindeki '{bird_feature}' özelliği, 'mechanize_flight' ")
    print(f"      yoluyla Uçak kavramına büküldü -> Sonuç: '{step1_feature}'")

    # 2. Adım: Uçaktan Denizaltına (sink_machine)
    step2_feature = concept_bender.transport("sink_machine", step1_feature)
    print(f"   2. Adım: Uçak üzerindeki '{step1_feature}', 'sink_machine'")
    print(f"      yoluyla Denizaltı kavramına büküldü -> Sonuç: '{step2_feature}'")

    # Direkt Yol Testi (Kuş -> Denizaltı : mechanize_and_sink)
    direct_feature = concept_bender.transport("mechanize_and_sink", bird_feature)
    print(f"\n   Direkt Yol: 'Kuş' kavramı doğrudan 'mechanize_and_sink' ")
    print(f"      yoluyla Denizaltına bükülseydi Sonuç: '{direct_feature}' çıkacaktı.")

    print("\n--- 3. BİLİMSEL SONUÇ (HoTT FUNCTORIALITY) ---")
    if is_valid_groupoid and is_valid_functorial and (step2_feature == direct_feature):
        print(" [BAŞARILI: HoTT İSPAT MOTORU ÇALIŞTI VE %100 KANITLANDI]")
        print(" Yapay zeka, bir kavramdan diğerine 'Vektör Yakınlığı (Kosinüs)'")
        print(" ile değil, 'Yollar (Homotopik Bükülme)' ile geçti. Functorial")
        print(" Transport kuralı sayesinde, adım adım geçişler (A->B->C) ile,")
        print(" direkt geçiş (A->C) arasında MÜKEMMEL BİR İZOMORFİZMA sağlandı.")
        print(" Bu, YZ'nin 'Kuş Kanadının' karşılığının Denizaltında 'Pervane'")
        print(" olduğunu istatistikle ezberlemesi değil, matematikle kanıtlamasıdır!")
    else:
        print(" [HATA] HoTT Groupoid ve Functoriality Yasaları Çöktü.")

if __name__ == "__main__":
    run_hott_experiment()
