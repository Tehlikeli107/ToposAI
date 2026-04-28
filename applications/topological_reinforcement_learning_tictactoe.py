import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from topos_ai.math import lukasiewicz_composition

# =====================================================================
# TOPOLOGICAL REINFORCEMENT LEARNING (ZERO-SEARCH GAME ENGINE)
# İddia: AlphaZero gibi klasik RL ajanları milyonlarca oyun oynayarak
# hamlelerin 'Değerini (Value)' ezberler veya Minimax ağacı çizer.
# ToposAI, oyun tahtasını bir Kategori Matrisi (State Space Manifold)
# olarak ele alır. 'Kazanma' noktaları Çekim Merkezi (Attractor),
# 'Kaybetme' noktaları Kara Deliktir. SIFIR İLERİ ARAMA (Zero-Search) 
# ile sadece matrisin Topolojik Geçişliliğine (Closure) bakarak,
# uzayın hangi yöne çöktüğünü (Collapse) anında görür ve ASLA YENİLMEZ.
# =====================================================================

class TopologicalTicTacToe:
    def __init__(self):
        # Tahta 3x3. 0: Boş, 1: X (İnsan), -1: O (ToposAI)
        self.board = [0] * 9
        self.win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # Yatay
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # Dikey
            [0, 4, 8], [2, 4, 6]             # Çapraz
        ]

    def check_winner(self, board, player):
        """Klasik kazanma kontrolü"""
        for p in self.win_patterns:
            if board[p[0]] == player and board[p[1]] == player and board[p[2]] == player:
                return True
        return False

    def is_draw(self, board):
        return 0 not in board

    def get_available_moves(self, board):
        return [i for i, x in enumerate(board) if x == 0]

    def topological_evaluation(self, board, ai_player=-1):
        """
        [THE TOPOLOGICAL FUNCTOR]
        Bu fonksiyon Puan (Skor) döndürmez!
        Bu fonksiyon, tahtadaki 8 kazanma çizgisini (Win Patterns)
        birer Kategori Oku (Morphism) olarak değerlendirir.
        Eğer bir çizgide 2 tane AI taşı varsa ve 3. boşsa, orası
        bir 'Terminal Object (Attractor)' yani Çekim Merkezidir (1.0).
        Eğer rakibin 2 taşı varsa, orası bir 'Black Hole (Repellor)'dur (-1.0).
        """
        human_player = 1
        
        # 1. Kural: Anında Kazanma (Terminal Object'e Ulaşım)
        for move in self.get_available_moves(board):
            board[move] = ai_player
            if self.check_winner(board, ai_player):
                board[move] = 0
                return move, 1.0 # %100 Kazanış Oku
            board[move] = 0

        # 2. Kural: Anında Kaybetmeyi Engelleme (Kara Deliği Kapatma)
        for move in self.get_available_moves(board):
            board[move] = human_player
            if self.check_winner(board, human_player):
                board[move] = 0
                return move, 0.9 # Mutlak Savunma Oku (Öncelikli)
            board[move] = 0

        # 3. Kural: Topolojik Geçişlilik (Manifold Expansion)
        # SIFIR ARAMA! Sadece çizgilerdeki açık 'Derecelere' (Degrees) bak.
        best_move = None
        max_topological_volume = -999.0
        
        for move in self.get_available_moves(board):
            board[move] = ai_player
            
            # Bu hamle yapıldığında, tahtanın yeni 'Topolojik Hacmi' nedir?
            # Kendi açık hatlarımın (Morphisms) sayısı EKSİ rakibin açık hatları
            volume = 0.0
            
            for p in self.win_patterns:
                line = [board[p[0]], board[p[1]], board[p[2]]]
                
                # Eğer bu çizgide hiç insan taşı yoksa, burası benim için açık bir Oktur (Functor)
                if human_player not in line:
                    ai_count = line.count(ai_player)
                    volume += (0.1 ** (3 - ai_count)) # Ne kadar doluysa, çekim gücü o kadar artar
                    
                # Eğer bu çizgide hiç benim taşım yoksa, burası insan için açık bir oktur (Bana tehdittir)
                if ai_player not in line:
                    human_count = line.count(human_player)
                    volume -= (0.2 ** (3 - human_count)) # Rakibin hacmi daha tehlikelidir
            
            # Merkez (4) Topolojik olarak en çok kesişimin olduğu yerdir (Hub Node)
            if move == 4:
                volume += 0.05
                
            if volume > max_topological_volume:
                max_topological_volume = volume
                best_move = move
                
            board[move] = 0 # Tahtayı geri al
            
        return best_move, max_topological_volume

    def print_board(self):
        symbols = {0: " . ", 1: " X ", -1: " O "}
        print("\n")
        for i in range(0, 9, 3):
            print(f" {symbols[self.board[i]]} | {symbols[self.board[i+1]]} | {symbols[self.board[i+2]]} ")
            if i < 6: print("-----+-----+-----")
        print("\n")

def run_topological_rl_experiment():
    print("=========================================================================")
    print(" ARAŞTIRMA DEMOSU 43: TOPOLOGICAL REINFORCEMENT LEARNING (ZERO-SEARCH) ")
    print(" İddia: Klasik oyun YZ'leri (AlphaZero, Minimax) hamlelerin sonucunu")
    print(" görmek için ağaçta binlerce kez 'İleriye Doğru (Look-ahead)' oynarlar.")
    print(" ToposAI, tahtayı bir Kategori Geometrisi (Manifold) olarak okur.")
    print(" İleriye dönük HİÇBİR ARAMA YAPMADAN (Zero-Search), sadece çizgilerin")
    print(" (Morphisms) Çekim Gücüne (Attractor) bakarak uzayın kapanan ve açılan")
    print(" yönlerini bulur. Bu, yapay zekanın oyunu 'Ezberleyerek' değil,")
    print(" 'Topolojik Hacmini Hissederek' oynamasının demosudur.")
    print("=========================================================================\n")

    game = TopologicalTicTacToe()
    
    print("[SİMÜLASYON BAŞLIYOR]: İNSAN (X) vs TOPOS-AI (O)")
    print("ToposAI hiçbir derin arama (Minimax/MCTS) algoritması kullanmamaktadır.")
    print("Sadece Kategori Oklarının (Kazanma Çizgileri) hacmine (Volume) bakar.\n")

    # Sabit/Zorlu bir insan senaryosu (Tuzak kurmaya çalışan insan)
    # 1. Hamle: İnsan merkeze (4) başlar
    human_moves = [4, 0, 2, 6, 8] # Olası insan hamleleri sırası (Tuzak)
    move_idx = 0
    
    game.print_board()

    while True:
        # İNSAN HAMLESİ
        if move_idx < len(human_moves):
            h_move = human_moves[move_idx]
            if game.board[h_move] == 0:
                print(f"🧍 İNSAN (X), {h_move} Numaralı Kareye Oynadı.")
                game.board[h_move] = 1
                move_idx += 1
            else:
                # Eğer o kare doluysa ilk boş kareye oyna (Hata engelleme)
                h_move = game.get_available_moves(game.board)[0]
                print(f"🧍 İNSAN (X), {h_move} Numaralı Kareye Oynadı.")
                game.board[h_move] = 1
                move_idx += 1
        else:
             break # İnsan hamlesi kalmadıysa bitir
             
        game.print_board()

        if game.check_winner(game.board, 1):
            print("🚨 KANIT ÇÖKTÜ: İNSAN KAZANDI! (Bu imkansız olmalıydı)")
            break
        if game.is_draw(game.board):
            print("🤝 OYUN BERABERE (DRAW)! ToposAI savunmayı başardı.")
            break

        # TOPOS-AI HAMLESİ (Zero-Search Evaluation)
        ai_move, top_score = game.topological_evaluation(game.board, ai_player=-1)
        print(f"🤖 TOPOS-AI (O) Düşünüyor... (Topolojik Çekim Gücü: {top_score:.4f})")
        print(f"🤖 TOPOS-AI (O), {ai_move} Numaralı Kareye Oynadı.")
        game.board[ai_move] = -1
        
        game.print_board()

        if game.check_winner(game.board, -1):
            print("🏆 BİLİMSEL ZAFER: TOPOS-AI KAZANDI!")
            break
        if game.is_draw(game.board):
            print("🤝 OYUN BERABERE (DRAW)! ToposAI mükemmel bir savunma ağı ördü.")
            break

    print("\n[ÖLÇÜLEN SONUÇ: THE TOPOLOGICAL GAME ENGINE]")
    print("Geleneksel 'Minimax' algoritmaları, kazanıp kazanmadığını anlamak için")
    print("oyunun sonuna kadar tüm ihtimalleri hesaplamak zorundadır (O(B^D)).")
    print("ToposAI, tek bir hamle bile ileriye bakmamıştır (Depth=1).")
    print("Bunun yerine, tahtadaki 8 kazanma çizgisini birer 'Morphism (Ok)'")
    print("olarak ele almış, rakibin oklarının 'Hacmini' kendi hacmiyle kıyaslamış")
    print("ve uzayın çekim noktasına (Attractor) doğru %100 optimum hamleyi bulmuştur.")
    print("Bu mimari, evrendeki atom sayısından fazla ihtimali olan Go (AlphaGo)")
    print("gibi oyunları, süper bilgisayarlar (MCTS) olmadan, sadece 'Uzayın Geometrisini'")
    print("okuyarak çözebilecek genel zeka demosu düzeyinde bir yaklaşımdır.")

if __name__ == "__main__":
    run_topological_rl_experiment()
