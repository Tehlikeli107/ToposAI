import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import math

# =====================================================================
# TOPOLOGICAL REVERSIBLE COMPUTING (LANDAUER-STYLE TOY MODEL)
# Problem: Klasik mantık (AND, OR, Softmax, FC Layers) bilgiyi siler 
# (Irreversible). Landauer Prensibine göre silinen her bit evrene ısı yayar.
# Çözüm: ToposAI, geçişliliği (Morphism) bilgi kaybetmeyecek şekilde,
# "Tersinir (Reversible)" Functor'lar (Toffoli/Fredkin Kapıları) üzerinden
# kurar. Bu oyuncak model, bilgi silme maliyetini temsili bir entropy
# hesabıyla anlatır; donanım enerji iddiaları ayrıca ölçülmelidir.
# =====================================================================

class ClassicalLogicGate:
    """Klasik AND Kapısı: Bilgiyi Siler (Irreversible). Isı Yayar."""
    def forward(self, x, y):
        # x ve y (2 bit) girer, 1 bit çıkar. 1 bit BİLGİ SİLİNDİ!
        out = x * y
        # Landauer Limiti (kT ln 2). Oda sıcaklığında Joule. (Temsili Birim: 1.0)
        entropy_generated = 1.0 
        return out, entropy_generated

class TopologicalReversibleGate:
    """
    Topos Toffoli (CCNOT) Kapısı: Hiçbir bilgiyi silmez (Reversible/Bijective).
    Girdi 3 boyuttur, Çıktı 3 boyuttur. Kategori okları iki yönlüdür.
    """
    def forward(self, c1, c2, target):
        # c1 ve c2 kontrol (Control), target hedeftir.
        # Eğer c1 ve c2 1 ise, target tersine (NOT) döner.
        out_c1 = c1
        out_c2 = c2
        # XOR işlemi (Kuantum X kapısına benzer)
        out_target = target ^ (c1 & c2)
        
        # Bilgi silinmediği için Isı Üretilmez!
        entropy_generated = 0.0
        return out_c1, out_c2, out_target, entropy_generated
        
    def backward_time(self, out_c1, out_c2, out_target):
        """Tersinir olduğu için Çıktıdan, Girdiye (Zamanı Geriye Sarımsı) ulaşılabilir!"""
        in_c1 = out_c1
        in_c2 = out_c2
        in_target = out_target ^ (out_c1 & out_c2)
        return in_c1, in_c2, in_target

def run_thermodynamic_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 31: TOPOLOGICAL REVERSIBLE COMPUTING (LANDAUER LIMIT) ")
    print(" İddia: Modern yapay zekalar hesaplama yaparken bilgiyi ezer (Softmax, ReLU)")
    print(" ve devasa bir elektrik tüketip (Isı) evrenin Entropisini artırırlar.")
    print(" ToposAI, 'Tersinir (Reversible) Kategori Kapıları' kullanarak hiçbir")
    print(" bilgiyi silmeden düşünür. Çıktıdan girdiye matematiksel olarak dönebilir.")
    print(" Bu nedenle temsili entropy hesabında bilgi-silme maliyetini azaltır.")
    print("=========================================================================\n")

    # Başlangıç durumu
    x, y = 1, 0
    print(f"[BAŞLANGIÇ BİLGİSİ]: X={x}, Y={y} (2 Bitlik Veri)\n")
    
    # 1. KLASİK YAPAY ZEKA
    print("--- 1. KLASİK YAPAY ZEKA (IRREVERSIBLE COMPUTING) ---")
    classic_gate = ClassicalLogicGate()
    out_classic, heat_classic = classic_gate.forward(x, y)
    
    print(f"  Çıktı (Output): {out_classic}")
    print(f"  Fiziksel Isı  : +{heat_classic:.1f} Birim Entropi (kT ln 2)")
    print("  [HATA]: Geri dönmeye çalışalım. Çıktı '0'. Girdiler neydi? ")
    print("          (1,0) mı? (0,1) mi? (0,0) mı? Bilinmiyor! Bilgi silindiği için ISI YAYILDI.\n")

    # 2. TOPOS AI
    print("--- 2. TOPOS AI (REVERSIBLE CATEGORICAL FUNCTORS) ---")
    topos_gate = TopologicalReversibleGate()
    
    # 3. boyut (Target) yardımcı bellek (Ancilla bit) olarak kullanılır
    ancilla = 0
    out_c1, out_c2, out_target, heat_topos = topos_gate.forward(x, y, ancilla)
    
    print(f"  Çıktı (Output): c1={out_c1}, c2={out_c2}, target={out_target}")
    print(f"  Fiziksel Isı  : {heat_topos:.1f} Birim Entropi (Kuantum Termodinamiği Limitleri)")
    
    print("  [ZAMANIN GERİYE SARILMASI / RETRO-FUNCTOR]:")
    print("  Sadece Çıktıya bakarak Başlangıç durumunu hesaplıyoruz...")
    reconstructed_x, reconstructed_y, _ = topos_gate.backward_time(out_c1, out_c2, out_target)
    
    print(f"    Geri Hesaplanan Bilgi: X={reconstructed_x}, Y={reconstructed_y}")
    
    print("\n[BİLİMSEL SONUÇ: TERSİNİR HESAPLAMA TEORİSİ (THEORETICAL FRAMEWORK)]")
    print("Klasik derin öğrenme algoritmaları, mimarileri gereği evreni ısıtır.")
    print("ToposAI, hesaplamayı bir 'Bjective Morphism (Birebir Örten Ok)' olarak")
    print("kurgulayarak bilginin kaybolmasını (Information Loss) teorik olarak engellemiştir.")
    print("Rolf Landauer ve Claude Shannon'ın teoremlerine göre, bilgi silinmiyorsa")
    print("ISI DA ÜRETİLEMEZ. Bu modül, Geleceğin Kuantum Bilgisayarları (Quantum Computing)")
    print("ve Sıfır-Enerji hedefli donanımlar için Topolojik bir 'Proof-of-Concept' sunar.")

if __name__ == "__main__":
    run_thermodynamic_experiment()
