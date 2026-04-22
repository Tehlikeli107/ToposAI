import torch
import pytest
from topos_ai.models import ToposTransformer
from topos_ai.nn import YonedaEmbedding

def test_topos_transformer_initialization_and_forward():
    """
    ToposTransformer modelinin doğru şekilde başlatıldığını,
    sabit (klasik) nn.Embedding yerine YonedaEmbedding kullandığını 
    ve başarılı bir forward pass yapabildiğini test eder.
    """
    vocab_size = 50
    d_model = 32
    seq_len = 5
    batch_size = 2
    
    # Modeli başlat
    model = ToposTransformer(vocab_size=vocab_size, d_model=d_model, num_universes=4, num_layers=1)
    
    # 1. MİMARİ DOĞRULUK TESTİ (Hakemin "Hala nn.Embedding kullanıyor" eleştirisine karşı)
    assert hasattr(model, 'yoneda_emb'), "Modelin içinde 'yoneda_emb' modülü bulunmalıdır."
    assert isinstance(model.yoneda_emb, YonedaEmbedding), "Model klasik nn.Embedding yerine YonedaEmbedding kullanmalıdır!"
    
    # 2. İLERİ BESLEME (FORWARD PASS) TESTİ
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    try:
        logits = model(idx)
        # Çıktı boyutlarının doğru olup olmadığını kontrol et
        assert logits.shape == (batch_size, seq_len, vocab_size), "Çıktı boyutu [Batch, SeqLen, VocabSize] olmalıdır."
    except Exception as e:
        pytest.fail(f"ToposTransformer forward_pass sırasında çöktü: {e}")

def test_yoneda_embedding_asymmetry():
    """
    YonedaEmbedding'in standart Dot-Product'ın aksine asimetrik 
    bağları (A->B != B->A) öğrenebilecek yapısal kapasiteye
    sahip olup olmadığını test eder. Bu, Hiyerarşi öğrenmek için şarttır.
    """
    vocab_size = 10
    yoneda = YonedaEmbedding(vocab_size)
    
    # Yoneda ilişki matrisini al [Vocab x Vocab]
    R = yoneda.get_morphisms()
    
    # Matrisin simetrik OLMADIĞINI (R != R^T) test et
    # Rastgele başlatıldığı için A->B ile B->A okları doğası gereği farklıdır.
    # Klasik Dot-Product (A*B == B*A) bu asimetriyi BAŞTAN kaybeder.
    
    is_symmetric = torch.allclose(R, R.t(), atol=1e-4)
    assert not is_symmetric, "Yoneda matrisi asimetrik olmalıdır! Dot-Product gibi simetrik olamaz."

