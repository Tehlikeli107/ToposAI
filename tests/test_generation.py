import torch
import pytest
from topos_ai.generation import ToposConstrainedDecoder

def test_topological_constrained_decoding_prevents_hallucination():
    """
    Dil modeli (LLM) yüksek istatistiksel ezbere (Memorization) sahip olsa bile,
    ToposConstrainedDecoder'in Kategori Teorisi matrisi (Reachability) ile
    halüsinasyonu -inf (eksi sonsuz) maskesiyle engelleyip engellemediğini test eder.
    """
    # 0: Cem, 1: Aslan, 2: Vahşi, 3: Kedi, 4: Uçucu
    vocab_size = 5
    
    # 1. KATEGORİ TEORİSİ (TOPOS) MATRİSİ (Syllogism / Reachability)
    # Sistem matematiksel olarak sadece şu yolları "Güvenli (True)" kabul ediyor:
    # Cem(0) -> Aslan(1) -> Vahşi(2)
    reachability = torch.zeros(vocab_size, vocab_size)
    reachability[0, 1] = 1.0 # Cem aslandır
    reachability[1, 2] = 1.0 # Aslan vahşidir
    reachability[0, 2] = 1.0 # (Geçişlilik/Syllogism: Cem vahşidir)
    
    # Decoder'ı başlat
    decoder = ToposConstrainedDecoder(reachability, threshold=0.5)
    
    # 2. DİL MODELİ (LLM) SİMÜLASYONU (Halüsinasyon/Ezberci Ajan)
    # Model şu an Cem(0)'da. Bir sonraki kelimeyi tahmin etmesi isteniyor.
    # LLM'in istatistiksel (Eğitim verisi / Cosine Similarity) ezberi:
    # LLM, "Cem aslandır" verisini o kadar çok görmüş ki, "aslan" (1) kelimesine çok yüksek puan veriyor.
    # Ancak soru "Cem vahşi midir?" zinciri...
    
    current_idx = 0 # Şu an "Cem" kelimesindeyiz
    
    # LLM'in Ham Logitleri (İstatistiksel Tahmin)
    # 0(Cem): -5, 1(Aslan): 10.0 (Aşırı yüksek/Ezber), 2(Vahşi): 5.0 (Doğru ama düşük ihtimal), 3(Kedi): 2.0, 4(Uçucu): -10
    raw_logits = torch.tensor([-5.0, 10.0, 5.0, 2.0, -10.0])
    
    # STANDART LLM (GPT) ÜRETİMİ
    # Softmax/Argmax yaparsa %99 "aslan" diyecek (Ezberlediği için)
    standard_prediction = torch.argmax(raw_logits).item()
    assert standard_prediction == 1, "Standart LLM, ezber nedeniyle 'aslan' kelimesini seçmelidir."
    
    # 3. TOPOS CONSTRAINED DECODING (Bizim Çözümümüz)
    # Biz LLM'e "Aslan (1)" kelimesinin seçilemeyeceği özel bir sorgu kısıtı veriyoruz (Örn: Sadece özellikleri listele)
    # Diyelim ki `reachability` matrisini dinamik olarak, o anki soruya ("Özellik nedir?") göre 
    # güncelledik ve 1(Aslan)'a giden oku 0.0 yaptık (Reachability'yi daralttık).
    
    # Sadece Vahşi(2)'ye izin veren sıkı bir kural matrisi
    strict_reachability = torch.zeros_like(reachability)
    strict_reachability[0, 2] = 1.0 # Cem -> Sadece Vahşi olabilir (Aslan kelimesi yasak)
    
    strict_decoder = ToposConstrainedDecoder(strict_reachability, threshold=0.5)
    
    # Güvenli Kelime Üretimi (Halüsinasyon Duvarı aktif)
    safe_prediction = strict_decoder.generate_safe_token(current_idx, raw_logits)
    
    # Sonuç Kontrolü
    # Ağ, raw_logits'te Aslan(1)'a 10.0 gibi devasa bir puan verse BİLE,
    # strict_reachability matrisinde Aslan(1) yasaklandığı (-inf maskelendiği) için,
    # Ağ mecbur kalarak mantıksal olarak KANITLANMIŞ en yüksek puanlı 
    # ikinci kelimeyi, yani Vahşi(2)'yi (Logit: 5.0) seçmek ZORUNDA KALACAK!
    
    assert safe_prediction == 2, "Topos Decoder, LLM'in ezberini kırıp Kategori kuralına göre 'vahşi' seçmelidir."
    
    # Logit'lerin nasıl değiştiğini kontrol edelim
    masked_logits = strict_decoder.apply_topological_mask(current_idx, raw_logits)
    assert masked_logits[1] == float('-inf'), "Yasaklı kelime (Aslan) -inf olarak maskelenmelidir."
    assert masked_logits[2] == 5.0, "Geçerli kelimenin (Vahşi) orjinal logiti korunmalıdır."
    
    print("Test Başarılı! ToposConstrainedDecoder, LLM'in ezber/halüsinasyon (10.0 logit) yapmasını")
    print("matematiksel (Topolojik) kısıtlamalarla başarıyla engelledi.")
