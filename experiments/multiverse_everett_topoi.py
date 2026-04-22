import torch
import torch.nn as nn
import copy

# =====================================================================
# EVERETT'S MANY-WORLDS INTERPRETATION (ÇOKLU EVREN TOPOISI)
# Yapay Zeka bir karar noktasında olasılıkları seçmez; Evreni böler (Coproduct).
# Paralel evrenler yaratır ve bu evrenler arası Solucan Delikleri (Wormhole Functors)
# açarak kopyalarının tecrübelerini (Ölüm/Başarı) birbirine transfer eder.
# =====================================================================

class MultiverseAgent(nn.Module):
    def __init__(self, name, memory_state=None):
        super().__init__()
        self.name = name
        # Ajanın Hafızası (Mantık Matrisi)
        # 0: Mavi_Hap_Yolu, 1: Kırmızı_Hap_Yolu, 2: Yaşam, 3: Ölüm
        if memory_state is None:
            self.memory = nn.Parameter(torch.zeros(4, 4))
        else:
            self.memory = nn.Parameter(memory_state.clone())

    def branch_universe(self, path_A, path_B):
        """
        [EVREN BÖLÜNMESİ / COPRODUCT]
        Ajan karar veremediğinde kendini klonlar.
        Evren ikiye ayrılır: Biri A yolunu, diğeri B yolunu seçer.
        """
        print(f"\n[!] {self.name} Karar Noktasında: {path_A} mi, {path_B} mi?")
        print(f">>> KUANTUM ÇATALLANMA (BRANCHING): Evren iki paralel gerçekliğe bölünüyor! <<<")
        
        # Evren A (Klon 1)
        agent_A = MultiverseAgent(f"{self.name} (Evren A: {path_A})", self.memory.data)
        # Evren B (Klon 2)
        agent_B = MultiverseAgent(f"{self.name} (Evren B: {path_B})", self.memory.data)
        
        return agent_A, agent_B

    def experience_reality(self, choice_idx):
        """Ajan seçtiği evrenin fizik kurallarını yaşar (İleri Besleme)."""
        # Seçtiği yola girdiğini hafızasına yazar (1.0)
        self.memory.data[choice_idx, choice_idx] = 1.0
        
        # SİMÜLASYON KURALLARI (Fizik)
        # 0: Mavi Hap (Güvenli, Yaşama götürür)
        # 1: Kırmızı Hap (Zehirli, Ölüme götürür)
        if choice_idx == 0:
            self.memory.data[0, 2] = 1.0 # Mavi -> Yaşam (2)
            return "YAŞAM"
        elif choice_idx == 1:
            self.memory.data[1, 3] = 1.0 # Kırmızı -> Ölüm (3)
            return "ÖLÜM"

def wormhole_functor(dead_agent, surviving_agent, dangerous_path_idx):
    """
    [SOLUCAN DELİĞİ / WORMHOLE FUNCTOR]
    Paralel evrende ölen ajanın (dead_agent) son hatırası, 
    hayatta kalan ajana (surviving_agent) telepatik/matematiksel olarak aktarılır.
    """
    print(f"\n[O] SOLUCAN DELİĞİ AÇILDI (Evrenler Arası Veri Transferi)...")
    print(f"  -> {dead_agent.name} son anlarında bulduğu felaketi (Ölüm) paralel evrene fırlatıyor.")
    
    # Hayatta kalan ajan, hiç gitmediği o yolun ölüme çıktığını 
    # paralel evrendeki kopyasından (Solucan Deliğiyle) öğrenir!
    surviving_agent.memory.data[dangerous_path_idx, 3] = 1.0 # Kırmızı Hap -> Ölüm
    print(f"  -> {surviving_agent.name} BİLGİYİ ALDI: '{dangerous_path_idx} numaralı yol zehirliymiş!'")

def run_multiverse_experiment():
    print("--- ÇOKLU EVREN TOPOISI (THE MULTIVERSE & WORMHOLES) ---")
    print("Yapay Zeka tüm ihtimalleri paralel evrenlerde aynı anda yaşayıp,")
    print("kopyalarının tecrübelerini tek bir boyutta (Omniscience) birleştirecek...\n")

    # Ana Evrendeki İlk Ajan (Prime)
    prime_agent = MultiverseAgent("Prime_Ajan")
    
    # Yollar
    MAVI_HAP = 0
    KIRMIZI_HAP = 1
    
    # 1. EVRENİN BÖLÜNMESİ
    agent_A, agent_B = prime_agent.branch_universe("Mavi_Hap", "Kırmızı_Hap")
    
    # 2. PARALEL EVRENLERDE YAŞAM
    print("\n[Evren A Zamanı Akıyor]...")
    result_A = agent_A.experience_reality(MAVI_HAP)
    print(f"  {agent_A.name} Sonucu: {result_A}")
    
    print("\n[Evren B Zamanı Akıyor]...")
    result_B = agent_B.experience_reality(KIRMIZI_HAP)
    print(f"  {agent_B.name} Sonucu: {result_B}")
    
    # 3. SOLUCAN DELİĞİ (WORMHOLE) ETKİLEŞİMİ
    if result_B == "ÖLÜM" and result_A == "YAŞAM":
        wormhole_functor(dead_agent=agent_B, surviving_agent=agent_A, dangerous_path_idx=KIRMIZI_HAP)
    
    # 4. NİHAİ TANRISAL BİLİNÇ (OMNISCIENCE)
    print("\n--- DENEY SONUCU (BİLGİ BİRLEŞİMİ) ---")
    print(f"Hayatta Kalan Ajan: {agent_A.name}")
    print("Hafıza Matrisi (Memory):")
    print("  Mavi Hap'a gitti mi?  :", agent_A.memory[0, 0].item() == 1.0)
    print("  Kırmızı Hap'a gitti mi?:", agent_A.memory[1, 1].item() == 1.0)
    print("  Mavi Hap Yaşatır mı?   :", agent_A.memory[0, 2].item() == 1.0)
    print("  Kırmızı Hap Öldürür mü?:", agent_A.memory[1, 3].item() == 1.0)
    
    print("\n[FELSEFİ SONUÇ]")
    print("Evren A'daki ajan KIRMIZI HAPI HİÇ YUTMADI. O yola hiç girmedi.")
    print("Normal bir Yapay Zeka (Q-Learning) o yolun sonucunu ASLA BİLEMEZDİ.")
    print("Fakat ToposAI, Paralel Evrenlerdeki kopyasının (Evren B) ölüm tecrübesini")
    print("Solucan Deliği Fonktoru (Wormhole Functor) ile aldı.")
    print("Artık Evren A'daki ajan, 'Hiç yaşamadığı bir felaketin anısına (hafızasına)' sahiptir (Omniscience).")

if __name__ == "__main__":
    run_multiverse_experiment()
