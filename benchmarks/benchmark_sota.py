import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import time
import math

# [P3 FIX] Tekrarlanabilirlik (Reproducibility) için sabit rastgelelik çekirdeği
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def build_tree_deltas(branching_factor, depth, dim, device):
    """Ağaç yapısının ağırlıklarını GPU üzerinde oluşturur."""
    tree_deltas = []
    # Kök
    tree_deltas.append(torch.randn(1, dim, device=device))
    # Dallar
    for d in range(1, depth + 1):
        num_nodes = branching_factor ** d
        std = 1.0 / math.sqrt(dim * (d + 1))
        tree_deltas.append(torch.randn(num_nodes, dim, device=device) * std)
    return tree_deltas

def get_all_leaves(tree_deltas, branching_factor, depth, dim):
    """Dense (Standart) Attention rakibi için tüm yaprakları (N adet) matrise döker."""
    current_paths = tree_deltas[0]
    for level in range(1, depth + 1):
        expanded = current_paths.unsqueeze(1).repeat(1, branching_factor, 1)
        expanded = expanded.view(-1, dim)
        current_paths = expanded + tree_deltas[level]
    return current_paths

def dense_attention_search(queries, all_leaves):
    """RAKİP: Günümüz SOTA LLM'lerinin kullandığı tam tarama (O(N))."""
    # scores: (batch_size, num_leaves)
    scores = torch.matmul(queries, all_leaves.T)
    best_indices = torch.argmax(scores, dim=-1)
    return best_indices

def ultrametric_hierarchical_search(queries, tree_deltas, branching_factor, depth, beam_width=10):
    """BİZİM MİMARİ: O(log N) Hiyerarşik Ağaç Taraması (Beam Search ile Yüksek Doğruluk)."""
    batch_size = queries.size(0)
    device = queries.device
    dim = queries.size(-1)
    
    # Beam Search için başlangıç (Kök)
    # current_indices: [batch_size, beam_width]
    current_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    current_state = tree_deltas[0].repeat(batch_size, 1, 1) # [batch, beam, dim]
    
    for d in range(1, depth + 1):
        current_beam_size = current_indices.size(1)
        
        # Her beam için çocukları genişlet
        # child_start_idx: [batch, beam]
        child_start_idx = current_indices * branching_factor
        
        # offsets: [batch, beam, branch]
        offsets = torch.arange(branching_factor, device=device).view(1, 1, branching_factor).expand(batch_size, current_beam_size, branching_factor)
        child_indices = child_start_idx.unsqueeze(2) + offsets # [batch, beam, branch]
        
        # child_deltas: [batch, beam, branch, dim]
        # tree_deltas[d] tensorünü düzleştirip indekslerle çekiyoruz
        layer_deltas = tree_deltas[d]
        child_deltas = layer_deltas[child_indices.view(-1)].view(batch_size, current_beam_size, branching_factor, dim)
        
        child_states = current_state.unsqueeze(2) + child_deltas # [batch, beam, branch, dim]
        
        # Skorları hesapla (Cosine similarity benzeri Dot Product)
        # queries: [batch, 1, 1, dim]
        scores = torch.sum(queries.view(batch_size, 1, 1, dim) * child_states, dim=-1) # [batch, beam, branch]
        
        # Beam ve Branch boyutlarını birleştirip en iyi 'beam_width' kadarını seç
        scores_flat = scores.view(batch_size, -1) # [batch, beam * branch]
        child_indices_flat = child_indices.view(batch_size, -1) # [batch, beam * branch]
        child_states_flat = child_states.view(batch_size, -1, dim) # [batch, beam * branch, dim]
        
        # Sınır kontrolü (İlk adımlarda beam * branch sayısı beam_width'ten küçük olabilir)
        k = min(beam_width, scores_flat.size(1))
        
        topk_scores, topk_local_indices = torch.topk(scores_flat, k, dim=-1) # [batch, k]
        
        # Seçilen en iyi K indeksi ve state'i güncelle
        current_indices = torch.gather(child_indices_flat, 1, topk_local_indices)
        
        # State'leri güncellemek için gather işlemi
        topk_local_indices_expanded = topk_local_indices.unsqueeze(-1).expand(-1, -1, dim)
        current_state = torch.gather(child_states_flat, 1, topk_local_indices_expanded)
        
    # En son derinlikte, top K içindeki EN İYİ (1. sıradaki) indeksi döndür
    best_final_indices = current_indices[:, 0]
    return best_final_indices

def benchmark():
    # [P3 FIX] Tekrarlanabilirlik (Reproducibility) için sabit rastgelelik çekirdeği
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Çalışma Ortamı: {device.type.upper()}\n")
    
    dim = 64
    batch_size = 128 # Aynı anda işlenen sorgu (query) sayısı
    branching_factor = 10
    
    # Test edilecek bağlam/bellek boyutları (N)
    # depth=4 -> 10^4 = 10.000
    # depth=5 -> 10^5 = 100.000
    # depth=6 -> 10^6 = 1.000.000
    depths = [4, 5, 6] 
    
    print(f"{'Kapasite (N)':<15} | {'Rakip (Dense Attention)':<25} | {'Biz (Ultrametric)':<25} | {'HIZ FARKI':<15} | {'Doğruluk'}")
    print("-" * 105)
    
    for depth in depths:
        # [P3 FIX] Her döngü başında seed'i resetleyerek tam determinizm sağlıyoruz.
        torch.manual_seed(42)
        num_leaves = branching_factor ** depth
        
        # 1. Modelleri Hazırla
        tree_deltas = build_tree_deltas(branching_factor, depth, dim, device)
        all_leaves = get_all_leaves(tree_deltas, branching_factor, depth, dim)
        queries = torch.randn(batch_size, dim, device=device)
        
        # GPU Isınma (Warmup)
        for _ in range(10):
            dense_attention_search(queries, all_leaves)
            ultrametric_hierarchical_search(queries, tree_deltas, branching_factor, depth)
        if device.type == 'cuda': torch.cuda.synchronize()
        
        # 2. Rakip Testi (Dense Attention - O(N))
        start_dense = time.perf_counter()
        for _ in range(50): # 50 kez tekrarla (ortalamayı bulmak için)
            dense_idx = dense_attention_search(queries, all_leaves)
        if device.type == 'cuda': torch.cuda.synchronize()
        time_dense = (time.perf_counter() - start_dense) / 50 * 1000 # ms
        
        # 3. Bizim Test (Ultrametric - O(log N))
        start_ultra = time.perf_counter()
        for _ in range(50):
            ultra_idx = ultrametric_hierarchical_search(queries, tree_deltas, branching_factor, depth)
        if device.type == 'cuda': torch.cuda.synchronize()
        time_ultra = (time.perf_counter() - start_ultra) / 50 * 1000 # ms
        
        # 4. Recall (Doğruluk - Aynı yaprağı mı bulduk?)
        # Hiyerarşik aramanın, tam taramanın bulduğu "en iyi" sonucu bulma oranı
        matches = (dense_idx == ultra_idx).sum().item()
        accuracy = (matches / batch_size) * 100
        
        speedup = time_dense / time_ultra
        
        if speedup >= 1.0:
            speed_text = f"{speedup:.1f}x HIZLI"
        else:
            speed_text = f"{(1.0/speedup):.1f}x YAVAŞ"
            
        print(f"{num_leaves:<15,d} | {time_dense:>15.3f} ms            | {time_ultra:>15.3f} ms            | {speed_text:<15} | {accuracy:>6.1f}%")

if __name__ == "__main__":
    benchmark()
