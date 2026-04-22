import torch
import time
import math

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

def ultrametric_hierarchical_search(queries, tree_deltas, branching_factor, depth):
    """BİZİM MİMARİ: O(log N) Hiyerarşik Ağaç Taraması."""
    batch_size = queries.size(0)
    device = queries.device
    dim = queries.size(-1)
    
    current_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    current_state = tree_deltas[0].repeat(batch_size, 1)
    
    for d in range(1, depth + 1):
        child_start_idx = current_indices * branching_factor
        offsets = torch.arange(branching_factor, device=device).unsqueeze(0).repeat(batch_size, 1)
        child_indices = child_start_idx.unsqueeze(1) + offsets
        
        child_deltas = tree_deltas[d][child_indices] # (batch, branch, dim)
        child_states = current_state.unsqueeze(1) + child_deltas # (batch, branch, dim)
        
        scores = torch.sum(queries.unsqueeze(1) * child_states, dim=-1) # (batch, branch)
        best_child = torch.argmax(scores, dim=-1)
        
        current_indices = child_start_idx + best_child
        current_state = child_states[torch.arange(batch_size, device=device), best_child]
        
    return current_indices

def benchmark():
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
        
        print(f"{num_leaves:<15,d} | {time_dense:>15.3f} ms            | {time_ultra:>15.3f} ms            | {speedup:>7.1f}x HIZLI | {accuracy:>6.1f}%")

if __name__ == "__main__":
    benchmark()
