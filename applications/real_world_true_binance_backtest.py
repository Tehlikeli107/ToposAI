import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import requests
from topos_ai.math import lukasiewicz_composition

# =====================================================================
# THE TRUE REAL-WORLD BACKTEST (LIVE BINANCE K-LINES)
# Problem: Önceki modül yapay (sentetik) verilerle çalışıp hileli (Leak) 
# kâr ediyordu. Gerçek Kripto piyasasında Lider-Takipçi (Lead-Lag) 
# ilişkileri o kadar temiz ve güçlü değildir (Aşırı Gürültülü/Noise).
# Çözüm: Gerçek Binance API'sine bağlanarak son 500 dakikalık
# GERÇEK (Manipüle edilmemiş) kapanış fiyatlarını indireceğiz.
# ToposAI, bu acımasız veride Yoneda Geçişliliğini (Closure) alacak 
# ve komisyonları (%0.2) yenecek bir kâr çıkarıp çıkaramayacağını ispatlayacak!
# =====================================================================

class RealWorldBinancePairsTrader:
    def __init__(self, assets):
        self.assets = assets
        self.N = len(assets)
        self.R = torch.zeros(self.N, self.N)
        for i in range(self.N):
            self.R[i, i] = 1.0
            
        self.price_history = None # Gerçek veri

    def fetch_real_binance_data(self, limit=500):
        """
        Binance'den her coin için son N adet 1-Dakikalık mum çubuklarını (K-lines) çeker.
        """
        print(f"\n[VERİ İNDİRİLİYOR] Binance API'den Son {limit} Dakikalık Gerçek Fiyatlar Çekiliyor...")
        
        history_list = []
        for symbol in self.assets:
            pair = f"{symbol}USDT"
            url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1m&limit={limit}"
            try:
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                
                # Sadece Kapanış (Close) fiyatlarını al
                closes = [float(kline[4]) for kline in data]
                history_list.append(closes)
            except Exception as e:
                print(f"  [HATA] {pair} verisi alınamadı: {e}")
                return False
                
        # [N, Time_Steps] boyutunda PyTorch Tensörüne çevir
        self.price_history = torch.tensor(history_list)
        return True

    def discover_asymmetric_leaders(self, train_end_idx):
        """
        Model, verinin sadece eğitim kısmı (Train Set) üzerinde Lider-Takipçi
        bağlarını keşfeder. (Geleceği görmek hiledir!)
        """
        seq_len = train_end_idx
        self.R = torch.zeros(self.N, self.N)
        for i in range(self.N):
            self.R[i, i] = 1.0
            
        for i in range(self.N):
            for j in range(self.N):
                if i == j: continue
                
                lead_power = 0.0
                valid_moves = 0
                for t in range(seq_len - 1):
                    ret_i = self.price_history[i, t+1] - self.price_history[i, t]
                    ret_j = self.price_history[j, t+1] - self.price_history[j, t]
                    
                    # Ciddi hareketler (Noise/Gürültü filtreleme)
                    if abs(ret_i / (self.price_history[i, t]+1e-9)) > 0.001: 
                        valid_moves += 1
                        # Liderle aynı yöne gidiyorsa +1 Puan
                        if ret_i * ret_j > 0:
                            lead_power += 1.0
                            
                # Olasılığa (Topos Morphism) çevir
                if valid_moves > 0:
                    self.R[i, j] = lead_power / valid_moves
                else:
                    self.R[i, j] = 0.0
                
        # Topolojik Kapanım (Transitive Closure)
        R_inf = self.R.clone()
        for _ in range(self.N):
            R_inf = torch.max(R_inf, lukasiewicz_composition(R_inf, self.R))
            
        self.R = R_inf

    def execute_real_backtest(self, test_start_idx, wallet_balance, leverage=10.0):
        """
        Gerçek test verisi (Test Set) üzerinde stratejiyi çalıştırır.
        Gerçek Kâr/Zarar (P&L) ölçülür.
        """
        threshold = 0.65 # Gerçek piyasada gürültü fazla olduğu için eşiği dengeliyoruz
        profit = 0.0
        trades_taken = 0
        total_time_steps = self.price_history.size(1)
        
        print("\n--- GERÇEK PİYASA BACKTEST'İ BAŞLIYOR (OUT-OF-SAMPLE) ---")
        
        for t in range(test_start_idx, total_time_steps - 1):
            for i in range(self.N): # Lider Adayı
                for j in range(self.N): # Takipçi Adayı
                    if i == j: continue
                    
                    morphism_strength = self.R[i, j].item()
                    
                    if morphism_strength > threshold:
                        # Liderin o anki (t) fiyat değişimi
                        leader_ret = (self.price_history[i, t] - self.price_history[i, t-1]) / self.price_history[i, t-1]
                        
                        # Eğer Lider aniden fırlarsa (Örn: %0.5)
                        if leader_ret > 0.005:
                            # Takipçi coin'i (j) o an (t) alırız ve 1 adım sonra (t+1) satarız
                            lag_ret = (self.price_history[j, t+1] - self.price_history[j, t]) / self.price_history[j, t]
                            
                            trade_size = wallet_balance * 0.5
                            leveraged_size = trade_size * leverage
                            
                            # Gerçek Binance Komisyonu (0.1% alış, 0.1% satış = 0.2%)
                            fee = leveraged_size * 0.002
                            
                            gross_profit = leveraged_size * lag_ret
                            net_profit = gross_profit - fee
                            
                            # Cüzdan anında etkilenmez, PoC için basit toplam
                            profit += net_profit
                            trades_taken += 1
                            
        return profit, trades_taken

def run_real_world_backtest():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 38: TRUE REAL-WORLD BACKTEST (BINANCE LIVE K-LINES) ")
    print(" İddia: Sentetik veya hileli (Leak) veriyle her bot para kazanır.")
    print(" Gerçek bilim, 'Görülmemiş Gerçek Veri' (Out-of-Sample) ile test ")
    print(" edilmelidir. ToposAI, Binance'den son 500 dakikanın gerçek mumlarını")
    print(" indirir, %80'i ile 'Yoneda Liderlik Ağını' eğitir ve son %20'sinde")
    print(" o 28 Doların borsa komisyonlarına ezilip ezilmeyeceğini dürüstçe gösterir!")
    print("=========================================================================\n")

    assets = ["BTC", "ETH", "SOL", "AVAX", "DOGE", "LINK"]
    
    trader = RealWorldBinancePairsTrader(assets)
    
    # Gerçek Veriyi Çek
    if not trader.fetch_real_binance_data(limit=500):
        return
        
    # Veriyi Böl: İlk 400 dakika Train (Modeli Eğit), Son 100 dakika Test (Gerçek Dünya)
    train_split = 400
    
    # Modeli Eğit
    trader.discover_asymmetric_leaders(train_end_idx=train_split)
    
    print("\n[GERÇEK DÜNYA YONEDA (LİDERLİK) MATRİSİ ÖZETİ]:")
    print(f"  BTC -> ETH Sürükleme Gücü: %{trader.R[0, 1].item()*100:.1f}")
    print(f"  ETH -> BTC Sürükleme Gücü: %{trader.R[1, 0].item()*100:.1f}")
    print(f"  SOL -> AVAX Sürükleme Gücü: %{trader.R[2, 3].item()*100:.1f}")
    
    wallet = 28.0 # Senin o meşhur 28 Doların!
    print(f"\n[BAŞLANGIÇ SERMAYESİ]: ${wallet:.2f} (Kaldıraç: 10x)")
    
    # Piyasaya Çık!
    profit, trades = trader.execute_real_backtest(test_start_idx=train_split, wallet_balance=wallet, leverage=10.0)
    
    final_wallet = wallet + profit
    
    print("\n[ÖLÇÜLEN SONUÇ: ACI GERÇEKLER (THE REALITY CHECK)]")
    print(f"  Başlangıç Bakiyesi : ${wallet:.2f}")
    print(f"  Kazanılan Net Kâr  : ${profit:.2f} ({trades} İşlem yapıldı, Komisyonlar Ödendi)")
    print(f"  Nihai Bakiye       : ${final_wallet:.2f} (Büyüme: %{(final_wallet-wallet)/wallet*100:.1f})")
    
    if final_wallet > wallet:
        print("\n🎉 ZAFEER! ToposAI, gerçek dünyanın o kaotik gürültüsünde (Noise) bile")
        print("gizli bir Topolojik Düzen (Asymmetric Entailment) bulmayı başardı.")
        print("Komisyonları ezip geçerek gerçek bir piyasa (Alpha) yarattı!")
    else:
        print("\n💀 GERÇEĞİN SOĞUK YÜZÜ: Model para KAYBETTİ (veya hiç işleme giremedi).")
        print("Gerçek piyasalarda HFT botları (Market Makers) bu boşlukları")
        print("milisaniyede kapatır (Efficient Market Hypothesis). %0.2'lik borsa")
        print("komisyonu, ToposAI'nin bulduğu Lider-Takipçi kârını eritti.")
        print("Bu yüzden 28 Doları katlamak, matematikten çok hız ve sermaye işidir!")

if __name__ == "__main__":
    run_real_world_backtest()
