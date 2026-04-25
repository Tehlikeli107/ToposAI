import torch
import torch.nn as nn
import torch.nn.functional as F

from topos_ai.optics import Lens

def test_lens_axioms():
    print("=========================================================================")
    print(" KATEGORİ TEORİSİ 'OPTİKLERİ' (LENSES) İHLAL VE YIKIM TESTİ ")
    print("=========================================================================\n")
    
    torch.manual_seed(42)
    
    # 1. YENİ 'STRICT' LENS TESTİ (Kanıtı)
    print("--- 1. STRICT LENS KANITLAMASI (YAPISAL BÜTÜNLÜK KORUMASI) ---")
    print("İddia: Lens yasaları %100 kusursuz, sıfır hatalı (0.00) olmak ZORUNDADIR.")
    print("Katı (Strict) Topos Optikleri, 'Get/Put' yasalarını sıfır hatayla sağlar mı?\n")
    
    DIM_S = 64 # Bütün (State) Boyutu
    DIM_A = 16 # Parça (Focus) Boyutu
    START_IDX = 10 # Odak başlangıcı
    
    # Bizim yeni yazdığımız "Strict" (Kesin) Lens
    strict_lens = Lens(dim_s=DIM_S, start_idx=START_IDX, dim_a=DIM_A)
    
    State = torch.rand(1, DIM_S, requires_grad=True)
    
    # Yasa 1: Get-Put (Okuduğunu geri yazarsan Bütün değişmemelidir)
    # S = Put(S, Get(S))
    Read_A = strict_lens.get(State)
    Put_A_Back = strict_lens.put(State, Read_A)
    
    get_put_error = torch.max(torch.abs(State - Put_A_Back)).item()
    print(f"Yasa 1 (Get-Put) Hatası: {get_put_error:.8f}")
    
    # Yasa 2: Put-Get (Yazdığını okursan aynı yazdığın parçayı bulmalısın)
    # x = Get(Put(S, x))
    New_A = torch.rand(1, DIM_A, requires_grad=True)
    State_with_New_A = strict_lens.put(State, New_A)
    Read_New_A = strict_lens.get(State_with_New_A)
    
    put_get_error = torch.max(torch.abs(New_A - Read_New_A)).item()
    print(f"Yasa 2 (Put-Get) Hatası: {put_get_error:.8f}")
    
    # Backprop (Öğrenilebilirlik) Testi
    print("\n--- 2. LENS GERİ YAYILIM (BACKPROPAGATION) TESTİ ---")
    print("Katı Lensler yapay zekanın (Gradyanların) akışını durduruyor mu?")
    loss = State_with_New_A.sum()
    loss.backward()
    
    if get_put_error == 0.0 and put_get_error == 0.0 and State.grad is not None:
        print(f"Bütün (State) Gradyan Gücü: {State.grad.abs().sum().item():.4f}")
        print(f"Parça (New_A) Gradyan Gücü: {New_A.grad.abs().sum().item():.4f}")
        print("\n[BAŞARILI] HATA 0.0! Sistem gerçek bir LENS (Ortogonal Optik) haline getirildi.")
        print("[BAŞARILI] Gradyan akışı sorunsuz. Sistem geriye doğru %100 öğrenebiliyor.\n")
    else:
        print("\n[BAŞARISIZ] Lens kuralları korunamadı veya türevler kilitlendi.")

    print("=========================================================================")
    print(" LENS KUSURU KANITLANMIŞTIR. SİSTEM %100 'STRICT' MATEMATİĞE GEÇİRİLECEKTİR.")
    print("=========================================================================")

if __name__ == '__main__':
    test_lens_axioms()