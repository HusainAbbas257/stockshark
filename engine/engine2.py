""" thing to be uptaded in this version:
- material count is more important in endgame
- hanging penalty is more important
"""
import tools.GUI as gui
import random
# Material table
material_table = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
    'e': 0
}

class Engine:
    def __init__(self):
        self.INITIAL_MATERIAL_FACTOR = 1.2
        self.ATTACK_FACTOR = 1.1
        self.DEFENCE_FACTOR=0.9
        self.SQUARE_FACTOR = 0.25
        self.KING_SAFETY_FACTOR = 1.2
        self.CENTER_FACTOR = 0.7
        self.HANGING_FACTOR = 1.0 

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
                    penalty+=eval_score*HANGING_FACTOR
                
        return penalty/64
    

    def king_squares(self,board:gui.Board,color:str):
        if 'K' not in board.position_table or 'k' not in board.position_table:
            return []

        return board.show_moves(board.position_table['K'if color=='w' else 'k'])
        # Mistake: assumes position_table['K'] exists, can crash if king captured which ofcoarce will not happen in real chess

    def square_eval(self, board, color, FEN):
        b_king_sqs = self.king_squares(board, 'b')
        w_king_sqs = self.king_squares(board, 'w')
        score = 0
        for row in board.grid:
            for cell in row:
                if cell is None: continue
                score += self.cell_eval(board, cell, color, FEN, b_king_sqs, w_king_sqs) * self.SQUARE_FACTOR
        return max(-500, min(500, (score/64)*10))

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
        eval_=self.count_material(board,FEN)
        eval_ += self.hanging_penalty(board, board.turn,FEN=FEN) * self.HANGING_FACTOR
        eval_+=self.square_eval(board,board.turn,FEN=FEN)
        return eval_

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
#  output:

