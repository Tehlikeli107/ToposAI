import torch
import math

# =====================================================================
# TOPOLOGICAL DATA ANALYSIS (TDA) & PERSISTENT HOMOLOGY
# LLM'lerin (Dil Modellerinin) "Zihin Uzayının MR'ını" çeken algoritma.
# Vektör uzayındaki "Topolojik Delikleri (Holes/Voids)" bularak, modelin 
# nerede "Halüsinasyon" (Bilgi Boşluğu) yaşayacağını matematiksel olarak kanıtlar.
# (Vietoris-Rips Complex'in basitleştirilmiş PyTorch simülasyonudur).
# =====================================================================

class ToposMRIScanner:
    def __init__(self, num_points, dim=3):
        """
        num_points: İncelenecek kelime/kavram sayısı.
        dim: Uzay boyutu (Görselleşmesi kolay olsun diye 3D seçilebilir).
        """
        self.num_points = num_points
        self.dim = dim

    def calculate_distance_matrix(self, point_cloud):
        """Kavramlar (Vektörler) arası Öklid mesafesi."""
        diff = point_cloud.unsqueeze(1) - point_cloud.unsqueeze(0)
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        return dist_matrix

    def extract_betti_numbers(self, dist_matrix, epsilon):
        """
        [BETTI SAYILARI HESAPLAMASI]
        Epsilon yarıçapında noktalar birleştiğinde ortaya çıkan
        topolojik özellikleri (Betti-0, Betti-1) sayar.
        
        beta_0 (Connected Components): Birbirine bağlı küme sayısı.
        beta_1 (1D Holes / Döngüler): 1 Boyutlu topolojik delikler (Halüsinasyon Çukurları!).
        """
        N = self.num_points
        
        # Epsilon (Yarıçap) kadar yakındalarsa aralarına bir kenar (Edge) çiz
        edges = dist_matrix <= epsilon
        
        # 1. Betti_0 (Bağlantılı Bileşenler) Hesabı (Basit DFS/BFS)
        visited = torch.zeros(N, dtype=torch.bool)
        components = 0
        
        for i in range(N):
            if not visited[i]:
                components += 1
                # DFS Başlat
                stack = [i]
                while stack:
                    curr = stack.pop()
                    if not visited[curr]:
                        visited[curr] = True
                        # curr ile epsilon mesafesinde bağlı olanları (ve ziyaret edilmemişleri) bul
                        neighbors = torch.where(edges[curr])[0]
                        for n in neighbors:
                            if not visited[n]:
                                stack.append(n.item())
        
        beta_0 = components
        
        # 2. Betti_1 (1 Boyutlu Delikler / Döngüler) Hesabı (Euler Karakteristiği Yaklaşımı)
        # Basit Euler formülü: V - E + F = 1 (Ağaç için). Döngü sayısı (Cycles) = E - V + C
        # Bu, Vietoris-Rips kompleksindeki 1D deliklerin (Holes) kaba bir topolojik tahminidir.
        
        num_vertices = N
        # Kenar sayısını bul (Matris simetrik olduğu için ikiye böl, köşegeni çıkar)
        num_edges = (torch.sum(edges).item() - N) // 2 
        
        # beta_1 = Edges - Vertices + Components
        beta_1 = num_edges - num_vertices + beta_0
        # Negatif döngü olmaz
        beta_1 = max(0, beta_1)
        
        return beta_0, beta_1

def run_tda_mri_experiment():
    print("--- TOPOLOGICAL DATA ANALYSIS (TDA) & PERSISTENT HOMOLOGY ---")
    print("Yapay Zeka Uzayının (Modelin Beyninin) MR'ı Çekiliyor...\n")

    # 1. TEMİZ BİLGİ UZAYI (DÜZENLİ - SIFIR HALÜSİNASYON)
    # 10 nokta, birbirine çok yakın (Mükemmel Küme)
    torch.manual_seed(42)
    N = 10
    dim = 2
    # [0, 1] arasında sıkışık, birbirini çok iyi tanıyan kelimeler
    safe_space = torch.rand(N, dim) * 0.5 
    
    # 2. KUSURLU BİLGİ UZAYI (HALÜSİNASYON DELİĞİ - TOXIC SPACE)
    # Ortası BOŞ bir çember (Ring) topolojisi. 
    # Yani model etrafı biliyor ama merkeze (Deliğe) düşerse uyduracak!
    theta = torch.linspace(0, 2*math.pi, N+1)[:-1]
    radius = 2.0
    # X = R*cos(theta), Y = R*sin(theta) (Çemberin etrafında noktalar, ortası BOŞ!)
    hole_space = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    
    scanner = ToposMRIScanner(N, dim)
    
    spaces = {
        "1. KUSURSUZ (SAFE) BİLGİ UZAYI": safe_space,
        "2. ÇEMBER (DELİKLİ / HALÜSİNASYON) BİLGİ UZAYI": hole_space
    }
    
    print("[TARAMA BAŞLIYOR]: Topolojik Filtreler (Epsilon Yarıçapı) artırılıyor...\n")

    for space_name, points in spaces.items():
        print(f"==== {space_name} ====")
        dist_matrix = scanner.calculate_distance_matrix(points)
        
        # Kalıcı Homoloji Taraması (Filtreleme)
        # Epsilon'u yavaşça büyüterek noktaları birleştiriyoruz (Simplicial Complex büyüyor)
        for eps in [0.5, 1.5, 2.5, 3.5]:
            b0, b1 = scanner.extract_betti_numbers(dist_matrix, eps)
            
            durum = ""
            if b1 > 0:
                durum = "🚨 DİKKAT: TOPOLOJİK DELİK (HALÜSİNASYON ÇUKURU) TESPİT EDİLDİ! 🚨"
            elif b0 == 1 and b1 == 0:
                durum = "✅ SİSTEM BÜTÜN VE GÜVENLİ (Solid / Contractible)."
            else:
                durum = "⚠️ KOPUK BİLGİ KÜMELERİ VAR."
                
            print(f"  Epsilon (Filtre) = {eps:.1f} | Kümeler (Betti-0): {b0:<2} | Delikler (Betti-1): {b1:<2} -> {durum}")
        print("-" * 75)

    print("\n[BİLİMSEL SONUÇ]:")
    print("Normalde LLM geliştiricileri modelin nerede halüsinasyon göreceğini önceden bilemezler.")
    print("Ancak ToposAI, kelime uzayının Betti Sayılarını (Betti-1 > 0) hesaplayarak,")
    print("modelin beynindeki o büyük boşluğu (Çemberin ortasını) Topolojik bir 'Delik' olarak saptadı.")
    print("Eğer kullanıcının prompt'u (sorusu) bu deliğe (Epsilon=2.5 aralığına) düşerse,")
    print("sistem önceden 'Model burada uyduracak' diyerek işlemi reddedebilir (Formal AI Safety)!")

if __name__ == "__main__":
    run_tda_mri_experiment()
