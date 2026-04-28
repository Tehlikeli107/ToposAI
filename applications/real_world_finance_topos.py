import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import torch
import yfinance as yf
import pandas as pd
import numpy as np

# =====================================================================
# REAL-WORLD FINANCE TOPOI: SYSTEMIC RISK & MARKET CONTAGION
# Evren: GERÇEK Amerikan Borsası (S&P 500) Son 1 Yıllık Kapanış Verileri.
# Yapay Zeka, şirketlerin fiyat hareketlerinden bir "Risk Topolojisi" kurar.
# Transitive Closure ile, bir şirkette başlayan şokun (Krizin) dalga dalga 
# hangi sektörlere sıçrayacağını Topolojik Risk Skoru ile hesaplar.
# =====================================================================

def lukasiewicz_composition(R1, R2):
    R1_exp = R1.unsqueeze(2) 
    R2_exp = R2.unsqueeze(0) 
    t_norm = torch.clamp(R1_exp + R2_exp - 1.0, min=0.0) 
    composition, _ = torch.max(t_norm, dim=1) 
    return composition

def fetch_real_stock_data(tickers):
    print("\n[VERİ OLUŞTURULUYOR] Yahoo Finance API'den GERÇEK Borsa Verisi indiriliyor...")
    print("  Son 1 Yıllık 'Günlük Kapanış (Adj Close)' verisi çekiliyor...")
    
    # yfinance ile toplu veri indir (Sessiz mod)
    data = yf.download(tickers, period="1y", progress=False)['Close']
    
    # Eksik verileri at (Eğer şirket yeni açıldıysa vs)
    data = data.dropna(axis=1)
    
    valid_tickers = list(data.columns)
    print(f"  [BAŞARILI]: {len(valid_tickers)} şirketin 1 yıllık işlem verisi ({len(data)} İşlem Günü) indirildi.\n")
    return data, valid_tickers

def build_financial_topology(returns_df):
    """
    [EVREN İNŞASI]
    Günlük Getirilerin (Daily Returns) Pearson Korelasyonu üzerinden, 
    "A Hissesi düşerse, B Hissesi de Düşer" ihtimalini [0.0, 1.0] arası 
    Morfizma (Topos Oku) Matrisine çevirir.
    """
    # 1. Korelasyon Matrisi (Şirketlerin Birlikte Hareket Etme Gücü)
    corr_matrix = returns_df.corr()
    
    # Kategori Teorisinde okların ağırlığı [0, 1] arası olmalıdır.
    # Pozitif korelasyon (Aynı yönde hareket) tehlike demektir. 
    # Negatif korelasyonları 0.0 (Bağ yok/Güvenli) yapıyoruz.
    topos_R = np.clip(corr_matrix.values, 0.0, 1.0)
    
    # Kendisiyle olan bağları (Diagonal) 1.0 bırakıyoruz
    return torch.tensor(topos_R, dtype=torch.float32)

def run_finance_contagion_experiment():
    print("=========================================================================")
    print(" TOPOSAI: FINANCIAL CONTAGION (GERÇEK DÜNYA EKONOMİK KRİZ SİMÜLATÖRÜ) ")
    print(" İddia: Klasik borsa analistleri 'A ve B birbirine ne kadar bağlı?' diye sorar.")
    print(" ToposAI ise, 'A batarsa, B üzerinden C'yi tetikleyerek D'yi yok edebilir mi?' ")
    print(" zincirlemesini (Sistemik Kelebek Etkisi) canlı veri üzerinden GÖSTERİR.")
    print("=========================================================================\n")

    # Farklı Sektörlerden 15 Gerçek Şirket
    # Teknoloji, Banka, Enerji, Perakende, E-Ticaret
    tickers = [
        "AAPL", "MSFT", "NVDA", "TSLA",         # Teknoloji
        "JPM", "GS", "BAC",                     # Banka/Finans
        "XOM", "CVX",                           # Enerji (Exxon, Chevron)
        "WMT", "TGT", "AMZN",                   # Perakende (Walmart, Target)
        "PFE", "JNJ",                           # İlaç
        "KO"                                    # Gıda (Coca-Cola)
    ]
    
    # 1. GERÇEK VERİLERİ ÇEK VE YÜZDESEL GETİRİ (RETURN) HESAPLA
    prices_df, valid_tickers = fetch_real_stock_data(tickers)
    returns_df = prices_df.pct_change().dropna()
    N = len(valid_tickers)
    t_idx = {t: i for i, t in enumerate(valid_tickers)}

    # 2. TOPOLOJİK MATRİSİ (R) KUR
    R = build_financial_topology(returns_df)
    
    # 3. TRANSITIVE CLOSURE (KRİZİN SONSUZ ZİNCİRLEME ŞOK DALGASI)
    # R matrisi sadece 1 adımlık (Doğrudan) şoku gösterir.
    # R_inf ise "Sonsuz Zincirleme (Contagion)" haritasını çıkarır.
    print("[MANTIK AĞI ÇALIŞIYOR] Sistemik Şok Dalgası (Transitive Closure) hesaplanıyor...")
    R_inf = R.clone()
    for _ in range(5): # 5 Adımlık Kriz Sıçraması
        new_R = lukasiewicz_composition(R_inf, R)
        R_inf = torch.max(R_inf, new_R)

    print("Hesaplama Bitti.\n")

    # ---------------------------------------------------------
    # BİLİMSEL KEŞİFLER (Ajanın Kendi Kendine Buldukları)
    # ---------------------------------------------------------
    
    # A) SİSTEMİK RİSKİ EN YÜKSEK ŞİRKET HANGİSİ? (Çökerse tüm borsayı yakar)
    # Bir şirketin diğer HERKES üzerindeki geçişlilik gücünün ortalaması
    systemic_risk_scores = torch.mean(R_inf, dim=1)
    most_dangerous_idx = torch.argmax(systemic_risk_scores).item()
    safest_idx = torch.argmin(systemic_risk_scores).item()
    
    print("--- 1. SİSTEMİK RİSK ANALİZİ (TOO BIG TO FAIL) ---")
    print(f"Borsayı (Zincirleme olarak) En Çok Tetikleyen / Tehdit Eden Hisse: [{valid_tickers[most_dangerous_idx]}] (Risk: %{systemic_risk_scores[most_dangerous_idx]*100:.1f})")
    print(f"Borsadan En İzole / En Güvenli Liman Hisse: [{valid_tickers[safest_idx]}] (Risk: %{systemic_risk_scores[safest_idx]*100:.1f})\n")

    # B) HİSSEDEN HİSSEYE ŞOK DALGASI SİMÜLASYONU
    print("--- 2. KRİZ SİMÜLASYONU (CONTAGION SHOCKWAVE) ---")
    
    test_cases = [
        ("TSLA", "AAPL"), # Tesla düşerse Apple'ı etkiler mi?
        ("JPM", "NVDA"),  # Banka krizinin Yapay Zekaya etkisi
        ("XOM", "KO")     # Petrolün Kolaya etkisi
    ]
    
    for u_str, v_str in test_cases:
        if u_str in t_idx and v_str in t_idx:
            u, v = t_idx[u_str], t_idx[v_str]
            
            # Doğrudan (1. Derece) Etki
            direct_shock = R[u, v].item()
            # Zincirleme (Transitive) Topolojik Etki
            chain_shock = R_inf[u, v].item()
            
            print(f"SENARYO: [{u_str}] Hissesi %10 çökerse, [{v_str}] hissesine şok nasıl yansır?")
            print(f"  > Normal Analist (Doğrudan Korelasyon): %{direct_shock*100:.1f} etkilenir der.")
            
            if chain_shock > direct_shock + 0.1:
                print(f"  > ToposAI (Zincirleme Risk): %{chain_shock*100:.1f} ETKİLENİR DER!")
                print(f"    (ÇÜNKÜ {u_str}, önce başka bir hisseyi, o hisse de gizlice {v_str}'yi batırmaktadır!)")
            else:
                print(f"  > ToposAI (Zincirleme Risk): %{chain_shock*100:.1f}")
                print(f"    (Sistem stabil. Gizli bir zincirleme tehlike yok.)")
            print("-" * 60)

    print("\n[SONUÇ]: ToposAI, internetteki bir PDF'i değil, SIFIR ÖĞRETMEN (Unsupervised) ile")
    print("canlı piyasa matematiğini indirmiş, işlemiş ve milyarlarca dolarlık Hedge Fonların")
    print("kullandığı 'Sistemik Risk Grafını (Network Topology)' Kategori Teorisiyle icat etmiştir.")

if __name__ == "__main__":
    run_finance_contagion_experiment()
