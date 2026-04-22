import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# TOPOS VISUALIZER (AÇIKLANABİLİR YAPAY ZEKA - XAI)
# Kategori Teorisi (Topos) matrislerini, insanların ve araştırmacıların 
# anlayabileceği Yönlü Çizgelere (Directed Graphs / Morphisms) çevirir.
# =====================================================================

class ToposVisualizer:
    def __init__(self, entities, threshold=0.3):
        """
        entities: Evrendeki varlıkların listesi (Örn: ["Kedi", "Süt", "Köpek"])
        threshold: Çizilecek okun (Morphism) minimum doğruluk gücü [0.0 - 1.0]
        """
        self.entities = entities
        self.threshold = threshold

    def plot_universe(self, relation_matrix, title="Topos Universe (Local Truth)"):
        """
        Tek bir evrenin (Topos) içindeki mantıksal bağıntıları çizer.
        relation_matrix: [N, N] boyutunda doğruluk (0-1) tensörü/numpy dizisi
        """
        if isinstance(relation_matrix, torch.Tensor):
            R = relation_matrix.detach().cpu().numpy()
        else:
            R = np.array(relation_matrix)
            
        G = nx.DiGraph()
        
        # Düğümleri (Kavramları) ekle
        for i, entity in enumerate(self.entities):
            G.add_node(i, label=entity)
            
        # Okları (Morfizmaları) ekle
        N = len(self.entities)
        for i in range(N):
            for j in range(N):
                weight = R[i, j]
                if weight > self.threshold:
                    # Okun kalınlığı doğruluk gücüne (weight) bağlı
                    G.add_edge(i, j, weight=weight)
                    
        # Çizim Ayarları
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=1.5, seed=42) # Düğümleri yay (spring) mantığıyla dağıt
        
        # Düğüm Çizimi
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black', linewidths=2)
        
        # Etiket Çizimi
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
        
        # Ok (Morphism) Çizimi
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges] # Çizgi kalınlığı
        
        # Çelişkileri/Güçlü bağları renklerle ayır (1.0'a yakınsa Yeşil, Düşükse Kırmızımsı)
        edge_colors = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, width=weights, 
            arrowsize=20, edge_color=edge_colors, 
            edge_cmap=plt.cm.RdYlGn, edge_vmin=0.0, edge_vmax=1.0, 
            connectionstyle="arc3,rad=0.1" # Kavisli oklar (A->B ve B->A çakışmasın diye)
        )
        
        # Ağırlık (Doğruluk) Değerlerini okların üzerine yaz
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=10, label_pos=0.3)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_sheaf_gluing(self, R_A, R_B, glued_R, conflict_score, name_A="Tanık 1", name_B="Tanık 2"):
        """
        İki çelişen evrenin (Local Topoi) kesişimini ve Gluing (Yapıştırma) sonucunu yan yana çizer.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        matrices = [(R_A, name_A, 'lightblue'), (R_B, name_B, 'lightgreen'), (glued_R, f"Glued Reality\n(Conflict: {conflict_score:.2f})", 'gold')]
        
        for ax_idx, (matrix, title, node_color) in enumerate(matrices):
            if isinstance(matrix, torch.Tensor):
                R = matrix.detach().cpu().numpy()
            else:
                R = np.array(matrix)
                
            G = nx.DiGraph()
            for i, entity in enumerate(self.entities):
                G.add_node(i, label=entity)
                
            N = len(self.entities)
            for i in range(N):
                for j in range(N):
                    weight = R[i, j]
                    if weight > self.threshold:
                        G.add_edge(i, j, weight=weight)
                        
            pos = nx.circular_layout(G)
            
            nx.draw_networkx_nodes(G, pos, ax=axes[ax_idx], node_size=2000, node_color=node_color, edgecolors='black')
            labels = nx.get_node_attributes(G, 'label')
            nx.draw_networkx_labels(G, pos, labels, ax=axes[ax_idx], font_size=10, font_weight='bold')
            
            edges = G.edges()
            weights = [G[u][v]['weight'] * 2 for u, v in edges]
            edge_colors = [G[u][v]['weight'] for u, v in edges]
            
            nx.draw_networkx_edges(
                G, pos, ax=axes[ax_idx], edgelist=edges, width=weights, 
                arrowsize=15, edge_color=edge_colors, 
                edge_cmap=plt.cm.RdYlGn, edge_vmin=0.0, edge_vmax=1.0,
                connectionstyle="arc3,rad=0.1"
            )
            
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=axes[ax_idx], font_color='red', font_size=8)
            
            axes[ax_idx].set_title(title, fontsize=14, fontweight='bold')
            axes[ax_idx].axis('off')
            
        plt.tight_layout()
        plt.show()

# --- TEST (KULLANIM ÖRNEĞİ) ---
if __name__ == "__main__":
    # Test 1: Self-Modifying AI (Paradigma Kayması) Sonrası Evreni Çizelim
    print("Topos Visualizer Test Ediliyor (Matplotlib Arayüzü Açılacak)...")
    
    entities = ["Şövalye", "Ejderha", "Prenses", "İcat_X_Diplomasi"]
    
    # Varsayılan Öğrenilmiş Matris (self_modifying_ai.py'nin Epoch 200 Sonucu)
    R_learned = torch.tensor([
        [1.0, 0.0, 0.0, 0.9], # Şövalye -> İcat_X'e gidiyor (0.9), Ejderhaya gitmiyor (0.0)
        [0.0, 1.0, 1.0, 0.0], # Ejderha -> Prensesi tutuyor (1.0)
        [0.0, 0.0, 1.0, 0.0], # Prenses 
        [0.0, 0.0, 0.8, 1.0]  # İcat_X -> Prensese ulaşıyor (0.8)
    ])
    
    visualizer = ToposVisualizer(entities, threshold=0.2)
    visualizer.plot_universe(R_learned, title="2-Category Paradigm Shift\n(Şövalye Ejderha'yı Es Geçip 'X' Üzerinden Prensese Ulaştı)")
    
    print("Çizim penceresi kapatıldığında program sonlanacaktır.")
