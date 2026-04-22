import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ultrametric_hopfield import UltrametricHopfieldMemory

def make_hierarchical_data(depth=3, branching=3, dim=64, samples_per_leaf=100, noise_std=0.5):
    """
    Hiyerarşik (ağaç) yapısına sahip sentetik veri üretir.
    Ortak ataları paylaşan sınıflar, uzayda birbirlerine daha yakındır.
    """
    torch.manual_seed(42)
    centers = [torch.zeros(dim)]
    
    # Hiyerarşik merkezleri (cluster centers) oluştur
    for d in range(1, depth + 1):
        new_centers = []
        for c in centers:
            for b in range(branching):
                # Derinlik arttıkça merkeze olan sapma azalır (ince detaylar)
                offset = torch.randn(dim) * (2.0 / d)
                new_centers.append(c + offset)
        centers = new_centers
        
    # Her yaprak (sınıf) için etrafında gürültülü veri örnekleri oluştur
    X, y = [], []
    for i, c in enumerate(centers):
        samples = c.unsqueeze(0) + torch.randn(samples_per_leaf, dim) * noise_std
        X.append(samples)
        y.append(torch.full((samples_per_leaf,), i, dtype=torch.long))
        
    return torch.cat(X), torch.cat(y)

class UltrametricClassifier(nn.Module):
    def __init__(self, input_dim, dim, depth, branching):
        super().__init__()
        # Girdiyi bellek uzayına (query) dönüştüren kodlayıcı ağ
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )
        # Hiyerarşik bellek katmanımız (Prototip / Sınıf merkezleri olarak davranacak)
        self.memory = UltrametricHopfieldMemory(dim=dim, depth=depth, branching_factor=branching)

    def forward(self, x):
        query = self.encoder(x)
        leaves = self.memory.get_leaf_memories()
        
        # Sınıflandırma için query ile tüm yapraklar arasındaki (O(N)) benzerliği ölç
        # Bu işlem türevlenebilir (differentiable) olduğu için eğitimde kullanılır.
        scores = torch.matmul(query, leaves.T) 
        return scores, query

def train_and_evaluate():
    dim = 32
    depth = 3
    branching = 3
    # Toplam sınıf (yaprak) sayısı = 3^3 = 27
    num_classes = branching ** depth 
    
    print(f"Veri Seti Hazırlanıyor... (Sınıf Sayısı: {num_classes})")
    X, y = make_hierarchical_data(depth=depth, branching=branching, dim=dim)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = UltrametricClassifier(input_dim=dim, dim=dim, depth=depth, branching=branching)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\nEğitim Başlıyor (Tam Arama - O(N) ile)...")
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            scores, _ = model(batch_X)
            loss = criterion(scores, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(scores, dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        acc = 100 * correct / total
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(dataloader):.4f} | Eğitim Doğruluğu: {acc:.2f}%")
        
    print("\n--- Çıkarım (Inference) Testi ---")
    model.eval()
    with torch.no_grad():
        # Test için verinin tamamını (veya bir kısmını) kullanalım
        query = model.encoder(X)
        
        # 1. Klasik (Global) Arama Doğruluğu (O(N) Karmaşıklık)
        leaves = model.memory.get_leaf_memories()
        global_scores = torch.matmul(query, leaves.T)
        global_preds = torch.argmax(global_scores, dim=-1)
        global_acc = 100 * (global_preds == y).sum().item() / y.size(0)
        
        # 2. Hiyerarşik (Ağaç Üzerinden) Arama Doğruluğu (O(log N) Karmaşıklık)
        # 27 sınıfın tamamıyla çarpım yapmak yerine, 3 derinlik x 3 dal = sadece 9 çarpım yapar!
        _, hierarchical_preds = model.memory.hierarchical_retrieval(query)
        hierarchical_acc = 100 * (hierarchical_preds == y).sum().item() / y.size(0)
        
        print(f"Klasik (Global) Arama Doğruluğu:    {global_acc:.2f}% (Tüm sınıflar tarandı)")
        print(f"Hiyerarşik (Top-Down) Arama Doğruluğu: {hierarchical_acc:.2f}% (Ağaç üzerinden rotalandı)")
        print("\nSonuç:")
        if hierarchical_acc > 90:
            print("BAŞARILI! Model, ağacın kökünden yapraklara doğru verinin hiyerarşik yapısını öğrenmiş ve hızlı aramaya olanak sağlamıştır.")
        else:
            print("Model hiyerarşiyi tam oturtamadı. (Eğitim parametreleri ayarlarına ihtiyaç olabilir)")

if __name__ == "__main__":
    train_and_evaluate()
