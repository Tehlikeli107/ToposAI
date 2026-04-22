import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UltrametricHopfieldMemory(nn.Module):
    """
    Spin-Glass tabanlı Ultrametrik Çağrışımlı Bellek (Ultrametric Associative Memory).
    Bellek durumları, kökten yapraklara doğru dallanan bir rastgele yürüyüş (branching random walk)
    ile inşa edilir. Bu sayede bellekler arası uzaklık, En Alt Ortak Ata (Lowest Common Ancestor - LCA)
    derinliğine bağlı bir ultrametrik uzay oluşturur.
    """
    def __init__(self, dim, depth, branching_factor, beta=1.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.branching_factor = branching_factor
        self.beta = beta # Modern Hopfield Softmax sıcaklık (inverse temperature) katsayısı
        
        # Toplam yaprak bellek (gerçek anılar) sayısı = branching_factor ** depth
        self.num_leaves = branching_factor ** depth
        
        # Hiyerarşik ağaç düğümlerinin inşası
        # Her seviyedeki vektörler o derinliğe özgü dallanma detaylarını (delta) tutar.
        self.tree_deltas = nn.ParameterList()
        for d in range(depth + 1):
            num_nodes = branching_factor ** d
            # Derinlik arttıkça varyans azalır, bu da ana hatlardan (kök) ince detaylara (yapraklar)
            # geçişi (spin-glass enerji havzalarını) simüle eder.
            std = 1.0 / math.sqrt(dim * (d + 1))
            self.tree_deltas.append(nn.Parameter(torch.randn(num_nodes, dim) * std))

    def get_all_nodes_at_depth(self, d):
        """Belirli bir derinlikteki tüm mutlak bellek durumlarını hesaplar (kökten d'ye kadar toplam)."""
        current_paths = self.tree_deltas[0] # (1, dim) - Kök
        
        for level in range(1, d + 1):
            # Bir üst seviyedeki düğümleri alt dallara kopyala
            expanded = current_paths.unsqueeze(1).repeat(1, self.branching_factor, 1)
            expanded = expanded.view(-1, self.dim)
            
            # Alt dalın kendi spesifik detayını (delta) ekle
            current_paths = expanded + self.tree_deltas[level]
            
        return current_paths

    def get_leaf_memories(self):
        """En alt seviyedeki (yaprak) tam çözünürlüklü bellekleri döndürür."""
        return self.get_all_nodes_at_depth(self.depth)

    def global_retrieval(self, query):
        """
        Geleneksel Modern Hopfield yaklaşımı ile (O(N) karmaşıklık) yapraklar üzerinde çağrışım.
        query: (batch_size, dim)
        """
        memories = self.get_leaf_memories() # (num_leaves, dim)
        
        # Benzerlik skoru hesapla (Dot product ultrametrik örtüşmeyi yansıtır)
        scores = torch.matmul(query, memories.T) # (batch_size, num_leaves)
        
        # Softmax (Energy Landscape)
        attn = F.softmax(scores * self.beta, dim=-1) # (batch_size, num_leaves)
        retrieved = torch.matmul(attn, memories) # (batch_size, dim)
        
        return retrieved, attn

    def hierarchical_retrieval(self, query):
        """
        Ultrametrik doğayı kullanan Top-Down (O(log N) karmaşıklık) çağrışım.
        Geniş enerji havzalarından (üst düğümler) spesifik anılara (yapraklar) doğru daralır.
        query: (batch_size, dim)
        """
        batch_size = query.size(0)
        device = query.device
        
        # Kökten başlıyoruz. Kökün indeksi 0.
        # current_indices: Her query için ağaçtaki aktif dalın indeksini tutar.
        current_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        current_node_states = self.tree_deltas[0].repeat(batch_size, 1) # (batch_size, dim)
        
        for d in range(1, self.depth + 1):
            # Seçili düğümlerin alt çocuklarını al
            child_start_idx = current_indices * self.branching_factor
            
            # Çocukların indekslerini oluştur: (batch_size, branching_factor)
            offsets = torch.arange(self.branching_factor, device=device).unsqueeze(0).repeat(batch_size, 1)
            child_indices = child_start_idx.unsqueeze(1) + offsets
            
            # Çocukların delta vektörlerini çek ve mutlak durumlarını hesapla
            child_deltas = self.tree_deltas[d][child_indices] # (batch_size, branching_factor, dim)
            child_states = current_node_states.unsqueeze(1) + child_deltas # (batch_size, branching_factor, dim)
            
            # Hangi alt dal query'ye daha yakın?
            # query: (batch_size, 1, dim), child_states: (batch_size, branching_factor, dim)
            scores = torch.sum(query.unsqueeze(1) * child_states, dim=-1) # (batch_size, branching_factor)
            
            # En uyumlu dalı (Hard-Attention / Winner-Take-All) seç
            best_child_local_idx = torch.argmax(scores, dim=-1) # (batch_size,)
            
            # Seçilen çocukların mutlak indekslerini ve durumlarını güncelle
            current_indices = child_start_idx + best_child_local_idx
            
            # Seçilen dalların state'ini çıkar
            batch_range = torch.arange(batch_size, device=device)
            current_node_states = child_states[batch_range, best_child_local_idx]
            
        return current_node_states, current_indices

# Basit bir test senaryosu
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Boyut: 64, Derinlik: 4 (Kök + 4 seviye), Dallanma: 3 (Her düğümün 3 çocuğu var)
    # Toplam kapasite = 3^4 = 81 farklı ultrametrik yaprak bellek
    model = UltrametricHopfieldMemory(dim=64, depth=4, branching_factor=3, beta=2.0)
    
    # 2 adet rastgele girdi (query)
    queries = torch.randn(2, 64)
    
    print(f"Toplam Yaprak Bellek Kapasitesi: {model.num_leaves}")
    
    # 1. Global (Klasik O(N)) Çağrışım
    global_res, attn = model.global_retrieval(queries)
    print("\nGlobal Retrieval Çıktı Boyutu:", global_res.shape)
    
    # 2. Hiyerarşik (Ultrametrik O(log N)) Çağrışım
    hierarchical_res, indices = model.hierarchical_retrieval(queries)
    print("Hiyerarşik Retrieval Çıktı Boyutu:", hierarchical_res.shape)
    print("Hiyerarşik Retrieval İndeksleri (81 yaprak içinden):", indices.tolist())
    
    # İki aramanın benzerliğini kontrol edelim (Ultrametrik uzayda top-down yaklaşım global'e çok yakınsar)
    cos_sim = F.cosine_similarity(global_res, hierarchical_res, dim=-1)
    print(f"\nGlobal ve Hiyerarşik arama arasındaki Kosinüs Benzerliği: {cos_sim.tolist()}")
