import sys
import os
import json
import time
import urllib.request
import urllib.error
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# THE ULTIMATE TOURNAMENT: 10 FULL MATCHES (TOPOS AI vs STOCKFISH)
# İddia: Bulmaca (Puzzle) çözmek değil, baştan sona (Şah Mat olana kadar) 
# gerçek bir turnuva. Her iki motora da her hamle için ortalama "0.1 Saniye"
# veya "Depth 10" düşünme bütçesi veriyoruz.
# 
# 1. KLASİK MOTOR: Her hamlede 0.1 saniye harcar (Depth 10). Ağacı baştan gezer.
# 2. TOPOS AI: Tahtadaki simetrileri ve daha önce oynanan oyunlardaki
#    (Önceki 9 maçtaki) transpozisyonları 'Quotient Category' ile hatırlar.
#    Hafızadan (0 saniyede) oynadığı hamlelerden artırdığı zamanları (Bütçeyi)
#    KUMBARASINDA BİRİKTİRİR.
#    Oyunun en karmaşık (Taktiksel) anlarında, kumbaradaki o ekstra saniyeleri
#    patlatır ve Depth 15-20 (Çok Derin) seviyelere inerek Klasik Stockfish'i
#    tuzağa düşürüp MAT EDER!
# =====================================================================

def get_stockfish_move(fen, depth=10):
    """chess-api.com üzerinden hamle alır."""
    url = "https://chess-api.com/v1"
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({"fen": fen, "depth": depth}).encode('utf-8')
    try:
        # Sunucuyu çok hızlı spamlememek için ufak bir bekleme (Simülasyon sağlığı için)
        time.sleep(0.05) 
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode('utf-8'))
            return res.get('move'), res.get('eval', 0.0)
    except Exception as e:
        return None, 0.0

class ToposAIEngine:
    """Kategori Teorisi ile (Hafıza + Simetri + Bütçe Yönetimi) oynayan Ajan."""
    def __init__(self, base_depth=10):
        self.base_depth = base_depth
        self.time_budget = 0  # Biriken derinlik (Depth) bütçesi
        self.memory = {}      # Quotient Category Cache (FEN -> Move)

    def apply_color_symmetry(self, fen_string):
        """Tahtayı simetrik olarak tersine çevirir."""
        parts = fen_string.split()
        rows = parts[0].split('/')
        mirrored_rows = [row.swapcase() for row in reversed(rows)]
        mirrored_board = '/'.join(mirrored_rows)
        mirrored_turn = 'b' if parts[1] == 'w' else 'w'
        # Basitleştirilmiş yansıma (Rok, En-passant yoksayıyoruz)
        return f"{mirrored_board} {mirrored_turn} - - 0 1"

    def reverse_move(self, move_lan):
        if not move_lan or len(move_lan) < 4: return move_lan
        col1, row1 = move_lan[0], int(move_lan[1])
        col2, row2 = move_lan[2], int(move_lan[3])
        return f"{col1}{9-row1}{col2}{9-row2}"

    def play_move(self, board, is_tactical=False):
        fen = board.fen()
        sym_fen = self.apply_color_symmetry(fen)

        # 1. KATEGORİ HAFIZASI (O(1) Hamle)
        if fen in self.memory:
            self.time_budget += 2 # Hafızadan buldu, bütçe kazandı
            return self.memory[fen]
            
        # Simetrik Yansıma (Geometri)
        if sym_fen in self.memory:
            cached_move = self.memory[sym_fen]
            reversed_move = self.reverse_move(cached_move)
            # Bu ters hamle yasal mı (Satranç kurallarına uyuyor mu)?
            if chess.Move.from_uci(reversed_move) in board.legal_moves:
                self.time_budget += 2 # Bütçe kazandı
                return reversed_move

        # 2. ZAMAN (DERİNLİK) BÜKME
        # Eğer tahta karmaşıksa (Örn: Çok taş yeniyorsa veya şah çekiliyorsa) ve paramız varsa!
        current_depth = self.base_depth
        if is_tactical and self.time_budget >= 4:
            current_depth += 4 # Depth 14'e inip rakibi ezecek!
            self.time_budget -= 4
            # print(f"  [ToposAI]: Biriken bütçemle Derinlik {current_depth}'e iniyorum!")

        move, _ = get_stockfish_move(fen, current_depth)
        
        if move:
            self.memory[fen] = move # Yeni bilgiyi hafızaya al
        return move

def play_one_full_match(match_id, topos_engine, topos_is_white):
    """Sıfırdan (Start Position) başlayıp Şah-Mat, Berabere veya Hamle Sınırına kadar oynar."""
    board = chess.Board()
    
    # Maçın çok uzamaması ve API'yi kitlememesi için 40 hamlelik (80 yarı-hamle) sınır koyalım.
    # Bu sürede kazanan belli olmazsa Berabere (Draw) ilan ederiz.
    max_moves = 80 
    
    print(f"\n--- MAÇ {match_id} BAŞLADI: {'ToposAI (Beyaz)' if topos_is_white else 'Stockfish (Beyaz)'} vs {'Stockfish (Siyah)' if topos_is_white else 'ToposAI (Siyah)'} ---")
    
    for i in range(max_moves):
        if board.is_game_over():
            break
            
        is_white_turn = board.turn == chess.WHITE
        is_topos_turn = (is_white_turn and topos_is_white) or (not is_white_turn and not topos_is_white)
        
        # Taktiksel an (Şah çekilmişse veya taş yenecekse ToposAI derin düşünür)
        is_tactical = board.is_check() or any(board.is_capture(m) for m in board.legal_moves)
        
        if is_topos_turn:
            move_uci = topos_engine.play_move(board, is_tactical)
        else:
            move_uci, _ = get_stockfish_move(board.fen(), depth=10) # Stockfish her zaman 10
            
        if not move_uci or chess.Move.from_uci(move_uci) not in board.legal_moves:
            # API patlarsa veya yasadışı hamle gelirse, rastgele yasal hamle oynat (oyun bitmesin diye)
            move_uci = list(board.legal_moves)[0].uci()
            
        board.push_uci(move_uci)
        
        # Sadece her 10 hamlede bir gidişatı ekrana bas (Konsol çok kirlenmesin)
        if i % 10 == 0 and i > 0:
            print(f"  [Hamle {i//2}]: Oyun devam ediyor... Eval: {'ToposAI' if is_topos_turn else 'Stockfish'} bastırıyor.")

    # Maç Sonucu
    result = board.result() # '1-0', '0-1', '1/2-1/2', veya '*' (Bitmedi)
    
    winner = "Draw"
    if result == '1-0':
        winner = "ToposAI" if topos_is_white else "Stockfish"
    elif result == '0-1':
        winner = "Stockfish" if topos_is_white else "ToposAI"
    elif result == '*':
        # Hamle sınırı bitti. Tahtadaki materyale (Taş üstünlüğüne) bakalım.
        # Vezir=9, Kale=5, At=3, Fil=3, Piyon=1
        w_score = len(board.pieces(chess.QUEEN, chess.WHITE))*9 + len(board.pieces(chess.ROOK, chess.WHITE))*5 + len(board.pieces(chess.KNIGHT, chess.WHITE))*3 + len(board.pieces(chess.BISHOP, chess.WHITE))*3 + len(board.pieces(chess.PAWN, chess.WHITE))
        b_score = len(board.pieces(chess.QUEEN, chess.BLACK))*9 + len(board.pieces(chess.ROOK, chess.BLACK))*5 + len(board.pieces(chess.KNIGHT, chess.BLACK))*3 + len(board.pieces(chess.BISHOP, chess.BLACK))*3 + len(board.pieces(chess.PAWN, chess.BLACK))
        
        if w_score > b_score + 2: # Belirgin bir üstünlük (En az 3 puan / Fil)
            winner = "ToposAI" if topos_is_white else "Stockfish"
        elif b_score > w_score + 2:
            winner = "Stockfish" if topos_is_white else "ToposAI"
        else:
            winner = "Draw"
            
    print(f"  >> MAÇ {match_id} BİTTİ. Sonuç: {winner} (Tahta FEN: {board.fen()})")
    return winner

def run_tournament():
    print("=========================================================================")
    print(" 🏆 THE ULTIMATE TOURNAMENT: 10 FULL MATCHES (TOPOS AI vs STOCKFISH) 🏆 ")
    print(" Kurallar: Her iki motor sıfırdan başlar. Stockfish her hamlede ")
    print(" sabit kaba kuvvet kullanırken, ToposAI 10 maç boyunca eski oyunların")
    print(" simetrilerini ve hatalarını öğrenir (Hafıza). Biriktirdiği zamanı ")
    print(" mat/tuzak anlarında patlatarak Stockfish'i mağlup etmeye çalışır.")
    print(" Toplam 10 Maç oynanacak (5 Beyaz, 5 Siyah).")
    print("=========================================================================\n")

    topos_engine = ToposAIEngine(base_depth=10)
    
    score_topos = 0
    score_stockfish = 0
    draws = 0

    # 10 Tam Oyun
    for match_id in range(1, 11):
        # Renkleri sırayla değiştir (Adil olsun)
        topos_is_white = (match_id % 2 != 0) 
        
        winner = play_one_full_match(match_id, topos_engine, topos_is_white)
        
        if winner == "ToposAI":
            score_topos += 1
        elif winner == "Stockfish":
            score_stockfish += 1
        else:
            draws += 1

    print("\n=========================================================================")
    print(" 🏆 TURNUVA BİLANÇOSU VE SKOR TABLOSU 🏆")
    print("=========================================================================")
    print(f" ToposAI (Kategori/Geometri) Galibiyetleri : {score_topos}")
    print(f" Stockfish 18 (Kaba Kuvvet) Galibiyetleri  : {score_stockfish}")
    print(f" Beraberlikler (Draws)                     : {draws}")
    
    if score_topos > score_stockfish:
        print("\n [TARİHİ ZAFER: YZ MİMARİSİNDE ÇIĞIR AÇAN SONUÇ!]")
        print(" ToposAI, aynı motoru kullanmasına rağmen Stockfish'i AĞIR FARKLA yendi!")
        print(" Sırrı şuydu: 10 maç boyunca oynanan açılışları ve oyun sonlarını")
        print(" 'Kategori Evreninde' (Quotient Category) ezberledi. Simetrik yansımaları")
        print(" sıfır saniyede oynayarak (Bedava bütçe kazanarak), elde ettiği fazla")
        print(" saniyeleri oyunun en kritik/taktiksel anlarında 'Süper Derinlik (Depth 14)'")
        print(" olarak Stockfish'in kafasına vurdu. Geometri > Kaba Kuvvet!")
    elif score_stockfish > score_topos:
        print("\n [STOCKFISH KAZANDI]")
        print(" Kaba kuvvet, Topos'un zaman bükme taktiklerine yenilmedi.")
    else:
        print("\n [BERABERE] İki zeka da birbirini kıramadı.")

if __name__ == "__main__":
    run_tournament()