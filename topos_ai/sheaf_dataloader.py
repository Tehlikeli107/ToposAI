import numpy as np
import torch


class SheafDataloader:
    """
    Memory-mapped dataloader with probe-based local-section projections.

    Raw feature rows stay on disk through NumPy memmap. Each batch is projected
    onto a smaller probe basis before it is moved to the requested device. This
    is a practical streaming/projection demo rather than a categorical proof of
    lossless sheaf reconstruction.
    """

    def __init__(self, file_path, num_samples, feature_dim, num_probes=64, batch_size=32):
        self.file_path = file_path
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_probes = num_probes

        torch.manual_seed(42)
        self.probes = torch.randn(num_probes, feature_dim) / (feature_dim**0.5)

    def _get_morphism(self, raw_chunk_tensor):
        """
        Project a raw batch to probe activations in `[0, 1]`.
        """
        morphisms = torch.matmul(raw_chunk_tensor, self.probes.t())
        return torch.sigmoid(morphisms)

    def stream_batches(self, device="cuda"):
        """Yield projected batches from a memory-mapped float32 matrix."""
        mmap_data = np.memmap(
            self.file_path,
            dtype="float32",
            mode="r",
            shape=(self.num_samples, self.feature_dim),
        )

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            raw_chunk_np = mmap_data[start_idx:end_idx]
            raw_chunk_tensor = torch.tensor(raw_chunk_np, dtype=torch.float32)
            yoneda_morphism = self._get_morphism(raw_chunk_tensor)
            yield yoneda_morphism.to(device)
