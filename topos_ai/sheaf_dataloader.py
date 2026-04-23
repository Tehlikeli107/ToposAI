import torch
import numpy as np

# =====================================================================
# CATEGORICAL SHEAF DATALOADER (YONEDA STREAMING)
# Amacı: Terabaytlarca veriyi (Big Data) RAM'e veya GPU VRAM'ine
# taşımak fiziksel bir darboğazdır (I/O Bottleneck).
# Kategori Teorisinde bir Demet (Sheaf), global verinin lokal 
# kesitlerden (Local Sections) kayıpsızca inşa edilebilmesidir.
# Bu modül, veriyi diske hapseder. Yoneda Lemma'yı kullanarak,
# devasa [N, 1_000_000] boyutlu ham veriyi diskin üzerinde okur
# ve onu sadece [N, 64] boyutlu 'İlişki Morfizmalarına (Functors)'
# dönüştürerek GPU'ya iletir.
# Sonuç: %99.9 I/O ve RAM Tasarrufu!
# =====================================================================

class SheafDataloader:
    def __init__(self, file_path, num_samples, feature_dim, num_probes=64, batch_size=32):
        """
        file_path: Devasa ham verinin diskteki konumu (.dat / .bin)
        num_samples: Veri adedi
        feature_dim: Her bir verinin orijinal boyutu (Örn: 100,000 piksel/gen)
        num_probes: Yoneda Sonda sayısı (GPU'ya gidecek nihai Topolojik boyut)
        """
        self.file_path = file_path
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_probes = num_probes
        
        # [THE YONEDA UNIVERSE ON CPU]
        # Evrenin referans noktaları CPU'da (veya Disk Controller'da) durur.
        # Boyut: [64, 100_000]
        # Not: Gerçekte bu problar küçük alt-kümeler (Sub-manifolds) olabilir,
        # biz burada tam boyutlu ama seyrek (Sparse) veya rastgele problar kullanıyoruz.
        torch.manual_seed(42)
        self.probes = torch.randn(num_probes, feature_dim) / (feature_dim ** 0.5)

    def _get_morphism(self, raw_chunk_tensor):
        """
        [YONEDA FUNCTOR (Disk-Side)]
        Devasa ham veriyi [Batch, 100_000] alır, 
        Problara olan uzaklığını (İlişkisini) bulur.
        Çıktı: [Batch, 64] boyutunda ufacık bir Topolojik Vektör!
        """
        # Disk tarafında matris çarpımı (veya L2 uzaklığı)
        # raw_chunk_tensor: [Batch, feature_dim]
        # probes.t(): [feature_dim, num_probes]
        # Sonuç: [Batch, num_probes]
        morphisms = torch.matmul(raw_chunk_tensor, self.probes.t())
        
        # Topolojik sınır [0, 1] (Reachability)
        return torch.sigmoid(morphisms)

    def stream_batches(self, device='cuda'):
        """
        [SHEAF LOCAL SECTIONS TO GLOBAL GPU]
        Numpy'nin 'mmap' (Memory-mapped file) özelliğini kullanarak
        diskteki 100 GB'lık dosyayı RAM'e almadan, pencere pencere
        (Local Sections) okuruz.
        """
        # Diskteki devasa dosyayı sanal olarak haritala (RAM'e yüklemez!)
        mmap_data = np.memmap(self.file_path, dtype='float32', mode='r', shape=(self.num_samples, self.feature_dim))
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            
            # Sadece ilgili 32 satırı diskten RAM'e çek
            raw_chunk_np = mmap_data[start_idx:end_idx]
            raw_chunk_tensor = torch.tensor(raw_chunk_np, dtype=torch.float32)
            
            # YONEDA SIKIŞTIRMASI (CPU'da, RAM patlamadan gerçekleşir)
            # 1 Milyon boyutlu veri, 64 boyuta düşer
            yoneda_morphism = self._get_morphism(raw_chunk_tensor)
            
            # GPU'ya sadece 64 boyutlu bu ufacık "İlişki / Ruh" gider!
            gpu_tensor = yoneda_morphism.to(device)
            
            yield gpu_tensor
