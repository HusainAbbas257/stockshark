""" thing to be uptaded in this version:
- material count is more important in endgame
- hanging penalty is more important
"""
import tools.GUI as gui
import random
# Material table
material_table = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 2000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -2000,
    'e': 0
}

class Engine:
    def __init__(self):
        self.INITIAL_MATERIAL_FACTOR = 1.5
        self.ATTACK_FACTOR = 1.0
        self.DEFENCE_FACTOR = 1.5
        self.SQUARE_FACTOR = 0.05
        self.KING_SAFETY_FACTOR = 2.5
        self.HANGING_FACTOR = 1.7
        self.CENTER_FACTOR = 0.7


    def count_material(self, board: gui.Board, FEN: str):
        """Calculate total material on the board.min:120,max:1800
        In endgame, material factor increases to give more weight to material."""
        moves=board.get_move_count(FEN=FEN)
        MATERIAL_FACTOR=self.INITIAL_MATERIAL_FACTOR + (20 - 2*moves*0.015 if 2*moves <20 else 1.5)
        
        total = 0
        for row in board.grid:
            for cell in row:
                total += material_table[cell.piece] * MATERIAL_FACTOR if cell.piece in material_table else 0
        return total

    def hanging_penalty(self, board: gui.Board, color: str,FEN: str):
        """ calculate hanging piece penalty."""
        moves=board.get_move_count(FEN=FEN)
        MATERIAL_FACTOR=self.INITIAL_MATERIAL_FACTOR + (20 - 2*moves*0.015 if 2*moves <20 else 1.5)
        HANGING_FACTOR=MATERIAL_FACTOR/1.125
        ATTACK_FACTOR,DEFENCE_FACTOR=self.ATTACK_FACTOR,self.DEFENCE_FACTOR
        if color=='b':
            ATTACK_FACTOR,DEFENCE_FACTOR=self.DEFENCE_FACTOR,self.ATTACK_FACTOR


        penalty=0
        for row in board.grid:
            for cell in row:
                attack=board.get_attackers(board,cell,'w' if color=='b' else 'b')
                defence=board.get_attackers(board,cell,'b' if color=='b' else 'w')
                if len(attack)==0:
                    continue
                eval_score=sum([(500-material_table[c.piece] )*ATTACK_FACTOR for c in attack])-sum([(-500+material_table[c.piece])*DEFENCE_FACTOR for c in defence])
                
                if abs(eval_score)>=100:
                    penalty+=(eval_score*HANGING_FACTOR)*(material_table[cell.piece]/100 if cell.piece!='e' in material_table else 0.1)
                # now doesnt blunders 
        return penalty/64
    

    def king_squares(self,board:gui.Board,color:str):
        if 'K' not in board.position_table or 'k' not in board.position_table:
            return []

        return board.show_moves(board.position_table['K'if color=='w' else 'k'])
        
    def square_eval(self, board, color, FEN):
        b_king_sqs = self.king_squares(board, 'b')
        w_king_sqs = self.king_squares(board, 'w')
        score = []
        for row in board.grid:
            for cell in row:
                if cell is None: continue
                score .append( self.cell_eval(board, cell, color, FEN, b_king_sqs, w_king_sqs) * self.SQUARE_FACTOR)
        return max(score) if color=='w' else min(score)

    def cell_eval(self, board: gui.Board,cell:gui.cell,color:str,FEN:str, b_king_sqs, w_king_sqs):
        # Mistake: swapping ATTACK_FACTOR/DEFENCE_FACTOR inside each call → state mutation but we also want to have same logic for both colors so we keep it as is
        moves=board.get_move_count(FEN=FEN)
        CENTER_FACTOR=self.CENTER_FACTOR - (20 - moves)*0.0075 if moves<20 else 0.1
        square_constant=100
        attackers=board.get_attackers(board,cell,'w')
        defenders=board.get_attackers(board,cell,'b')
        eval=0                     
        bkings_constant=1
        wkings_constant=1
        if cell in b_king_sqs:
            bkings_constant=self.KING_SAFETY_FACTOR
        if cell in w_king_sqs:
            wkings_constant=self.KING_SAFETY_FACTOR
        for ap in attackers:
            square_constant=self.get_square_value(cell,board)* CENTER_FACTOR
            # its value will be in range 5 to 30
            value=self.ATTACK_FACTOR*(square_constant / (material_table[ap.piece]/100))*bkings_constant if material_table[ap.piece]!=0 else 1
            eval+= value
            # Mistake: dividing by piece value → can be zero division if 'e'
        for dp in defenders:
            value= self.DEFENCE_FACTOR*(square_constant * (material_table[dp.piece]/100))*wkings_constant if material_table[dp.piece]!=0 else 1
            eval-=value
        return eval*bkings_constant*wkings_constant

    def get_square_value(self, cell: gui.cell,Board: gui.Board):
        """Assign value to a square based on its position."""
        col, row = cell.pos
        square_val = 21
        if col in ['d','e'] and str(row) in ['4','5']:
            square_val = 30
        elif col in ['c','f'] and str(row) in ['3','6']:
            square_val = 27
        elif col in ['b','g'] and str(row) in ['2','7']:
            square_val = 24
        return square_val

    def eval(self, board: gui.Board=None, FEN: str="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        if board is None:
            board = gui.Board(FEN=FEN)
        eval_=self.count_material(board,FEN)*self.INITIAL_MATERIAL_FACTOR
        eval_ += self.hanging_penalty(board, board.turn,FEN=FEN) * self.HANGING_FACTOR
        eval_+=self.square_eval(board,board.turn,FEN=FEN)
        return eval_/100

    def get_best_eval_move(self, board: gui.Board, FEN: str, count=3):
        """Get top count moves with slightly randomized selection"""
        moves_list = []

        for r in board.grid:
            for sq in r:
                if (board.turn == 'w' and sq.piece.isupper()) or (board.turn == 'b' and sq.piece.islower()):
                    moves = board.show_moves(sq)
                    for move in moves:
                        g = gui.Board(FEN=FEN)
                        g.move(sq.pos, move.pos)
                        new_fen = g.to_FEN(g.grid)
                        eval_score = self.eval(g, FEN=new_fen)
                        moves_list.append((eval_score, (sq.pos, move.pos)))

        # sort by eval (descending if white, ascending if black)
        moves_list.sort(key=lambda x: x[0], reverse=(board.turn == 'w'))

        # pick extra 2, then discard 2 randomly for human-like imperfection
        pick_count = count + 2 if len(moves_list) > count + 2 else len(moves_list)
        top_moves = moves_list[:pick_count]

        if len(top_moves) > count:
            top_moves = random.sample(top_moves, count)

        # return remaining sorted (best among imperfect set)
        top_moves.sort(key=lambda x: x[0], reverse=(board.turn == 'w'))
        return top_moves

        
    
    def next_move(self, FEN:str, depth=2, lines=3):
        board = gui.Board(FEN=FEN)
        best_move = None
        best_eval = -float('inf') if board.turn=='w' else float('inf')

        candidates = self.get_best_eval_move(board, FEN, lines)
        for _, move in candidates:
            new_board = gui.Board(FEN=board.to_FEN(board.grid))
            new_board.move(move[0], move[1])
            # after move, turn must be opponent
            new_board.turn = 'b' if board.turn=='w' else 'w'
            new_FEN = new_board.to_FEN(new_board.grid)

            if depth > 1:
                # recursive call returns best MOVE for the side to play in new_FEN
                reply_move, reply_eval = self.next_move(new_FEN, depth-1, lines)

                # NOTE: use reply_eval (which is evaluated from WHITE perspective)
                eval_final = reply_eval
            else:
                eval_final = self.eval(new_board, new_FEN)

            # update best_move based on board.turn (original side)
            if (board.turn=='w' and eval_final > best_eval) or (board.turn=='b' and eval_final < best_eval):
                best_eval = eval_final
                best_move = move

        return best_move, best_eval


        

    def play(self):
        b=gui.Board()
        FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 
        while True:
            bot=self.next_move(FEN)
            if bot is None:
                print("No valid moves available. Game over.")
                break
            print(f'bot plays from {bot[0]}--->{bot[1]}')
            b.move(bot[0],bot[1])
            b.turn = 'b' if b.turn == 'w' else 'w'
            from_,to_=input("Enter your move (e.g., e2 e4): ").split()
            if from_=='exit' or to_=='exit':
                break
            b.move(from_,to_)
            FEN=b.to_FEN(b.grid)

    def play_against_self(self, max_moves=100):
        b=gui.Board()
        FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 
        move_count=0
        while move_count<max_moves:
            bot = self.next_move(FEN)
            if bot is None: 
                print("No valid moves available. Game over.")
                break
            print(f'bot plays from {bot[0]}--->{bot[1]}')
            from_pos, to_pos = bot[0]
            b.move(from_pos, to_pos)

            # now flip turn because move executed
            b.turn = 'b' if b.turn == 'w' else 'w'
            FEN = b.to_FEN(b.grid)
            move_count += 1
            
    
            
if __name__ == "__main__":
    e=Engine()
    e.play_against_self()


# sample games against stockfish :
'''
game1:
Your Engine plays: b2b4
Stockfish plays: a7a5
Your Engine plays: b4a5
Stockfish plays: g7g6
Your Engine plays: d2d4
Stockfish plays: f8g7
Your Engine plays: d4d5
Stockfish plays: g7a1
Your Engine plays: d1d4
Stockfish plays: a1d4
Your Engine plays: e2e4
Stockfish plays: d4f6
Your Engine plays: c1e3
Stockfish plays: d7d6
Your Engine plays: f1e2
Stockfish plays: a8a5
Your Engine plays: b1d2
Stockfish plays: b8d7
Your Engine plays: d2f1
Stockfish plays: h7h5
Your Engine plays: f1d2
Stockfish plays: a5a2
Your Engine plays: d2b3
Stockfish plays: e8f8
Your Engine plays: e1f1
Stockfish plays: g6g5
Your Engine plays: e3g5
Stockfish plays: f6g5
Your Engine plays: e2h5
Stockfish plays: h8h5
Your Engine plays: c2c3
Stockfish plays: d7f6
Your Engine plays: g1e2
Stockfish plays: f6e4
Your Engine plays: e2d4
Stockfish plays: a2f2
Illegal move attempted by engine: c3c4
Fallback move played: f1g1
Stockfish plays: g5e3
Illegal move attempted by engine: g1f2
Fallback move played: h2h3
Stockfish plays: f2e2
Your Engine plays: g1f1
Stockfish plays: e4g3
Game Over
Result: 0-1

game2:
Your Engine plays: g2g4
Stockfish plays: d7d5
Your Engine plays: g1f3
Stockfish plays: b8c6
Your Engine plays: a2a4
Stockfish plays: c8g4
Your Engine plays: f3h4
Stockfish plays: e7e5
Your Engine plays: e2e3
Stockfish plays: g4d1
Your Engine plays: e1d1
Stockfish plays: d8h4
Your Engine plays: f1e2
Stockfish plays: a7a6
Your Engine plays: h1g1
Stockfish plays: h4h2
Your Engine plays: g1g7
Stockfish plays: h2h1
Illegal move attempted by engine: g7g8
Fallback move played: g7g1
Stockfish plays: h1g1
Illegal move attempted by engine: e2a6
Fallback move played: e2f1
Stockfish plays: g1f1
Game Over
Result: 0-1


game3:
Your Engine plays: b1c3
Stockfish plays: d7d5
Your Engine plays: c3e4
Stockfish plays: d5e4
Your Engine plays: g1f3
Stockfish plays: b8c6
Your Engine plays: h2h4
Stockfish plays: h7h6
Your Engine plays: a1b1
Stockfish plays: g8f6
Your Engine plays: b2b3
Stockfish plays: e7e5
Your Engine plays: f3g1
Stockfish plays: f8e7
Your Engine plays: c2c4
Stockfish plays: a7a5
Your Engine plays: h1h2
Stockfish plays: c8g4
Your Engine plays: d1c2
Stockfish plays: e8g8
Your Engine plays: c2d3
Stockfish plays: e4d3
Your Engine plays: e2e3
Stockfish plays: f8e8
Illegal move attempted by engine: e1e2
Fallback move played: f2f3
Stockfish plays: g4d7
Your Engine plays: g1h3
Stockfish plays: a5a4
Your Engine plays: f1d3
Stockfish plays: c6b4
Your Engine plays: h3f2
Stockfish plays: a4a3
Your Engine plays: d3e4
Stockfish plays: f6e4
Your Engine plays: f3e4
Stockfish plays: b4a2
Your Engine plays: c1a3
Stockfish plays: e7a3
Your Engine plays: c4c5
Stockfish plays: a2b4
Your Engine plays: b1d1
Stockfish plays: b4c2
Illegal move attempted by engine: b3b4
Fallback move played: e1f1
Stockfish plays: g8h8
Your Engine plays: f1g1
Stockfish plays: c2e3
Your Engine plays: d2e3
Stockfish plays: a3c5
Your Engine plays: d1a1
Stockfish plays: a8a1
Illegal move attempted by engine: h4h5
Fallback move played: f2d1
Stockfish plays: c5e3
Illegal move attempted by engine: d1e3
Fallback move played: g1f1
Stockfish plays: d7b5
Illegal move attempted by engine: d1e3
Fallback move played: f1e1
Stockfish plays: d8d1
Game Over
Result: 0-1

game4:
Your Engine plays: g2g4
Stockfish plays: d7d5
Your Engine plays: g1f3
Stockfish plays: b8c6
Your Engine plays: f3h4
Stockfish plays: c8g4
Your Engine plays: e2e3
Stockfish plays: g4d1
Your Engine plays: f1h3
Stockfish plays: e7e6
Your Engine plays: h3f1
Stockfish plays: d8h4
Your Engine plays: h1g1
Stockfish plays: h4h2
Your Engine plays: g1g7
Stockfish plays: f8g7
Your Engine plays: e1d1
Stockfish plays: h2f2
Your Engine plays: f1e2
Stockfish plays: f2g1
Illegal move attempted by engine: a2a4
Fallback move played: e2f1
Stockfish plays: g1f1
Game Over
Result: 0-1

game5:
Your Engine plays: b1c3
Stockfish plays: d7d5
Your Engine plays: c3d5
Stockfish plays: d8d5
Your Engine plays: a2a4
Stockfish plays: e7e5
Your Engine plays: g2g3
Stockfish plays: d5h1
Your Engine plays: h2h4
Stockfish plays: h1g1
Your Engine plays: a4a5
Stockfish plays: f8d6
Your Engine plays: c2c4
Stockfish plays: c8d7
Your Engine plays: d1a4
Stockfish plays: d7a4
Illegal move attempted by engine: f1h3
Fallback move played: a1a2
Stockfish plays: g1f1
Your Engine plays: e1f1
Stockfish plays: a4c6
Your Engine plays: d2d3
Stockfish plays: b8a6
Your Engine plays: c1g5
Stockfish plays: f7f6
Your Engine plays: g5f6
Stockfish plays: g8f6
Your Engine plays: a2a1
Stockfish plays: c6d7
Your Engine plays: a1a2
Stockfish plays: d7h3
Illegal move attempted by engine: f1g2
Fallback move played: f1e1
Stockfish plays: e8g8
Your Engine plays: a2a4
Stockfish plays: h7h5
Your Engine plays: a4a3
Stockfish plays: a8e8
Your Engine plays: a3c3
Stockfish plays: h3d7
Your Engine plays: c3b3
Stockfish plays: a6c5
Your Engine plays: b3b7
Stockfish plays: c5b7
Your Engine plays: b2b3
Stockfish plays: f8f7
Your Engine plays: e1d1
Stockfish plays: b7a5
Your Engine plays: d1e1
Stockfish plays: d6b4
Your Engine plays: e1d1
Stockfish plays: a5b3
Your Engine plays: f2f3
Stockfish plays: b3d4
Your Engine plays: d1c1
Stockfish plays: d7e6
Illegal move attempted by engine: c1d2
Fallback move played: f3f4
Stockfish plays: b4a3
Illegal move attempted by engine: f4e5
Fallback move played: c1d2
Stockfish plays: e5f4
Illegal move attempted by engine: d2e3
Fallback move played: g3f4
Stockfish plays: a3b4
Your Engine plays: d2e3
Stockfish plays: f6g4
Illegal move attempted by engine: c4c5
Fallback move played: e3e4
Stockfish plays: e6f5
Illegal move attempted by engine: e4f5
Fallback move played: e4d4
Stockfish plays: f7d7
Game Over
Result: 0-1

'''


'''
problems observed:
- engine makes illegal moves sometimes (probably due to wrong move conversion) should be fixed from tools.GUI side
- engine blunders pieces sometimes (probably due to hanging penalty miscalculation)
-takes long time for depth>3

Suggestions for improvement in next version:
- fix illegal move generation bug by remaking tools.GUI  in a highly optimized way
-remove complex logic from square evaluation to make it faster and more human like
-use minimax to make it stronger a little bit
- add opening book
and more...
'''