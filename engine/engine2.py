""" thisng to be uptaded in this cersion:
- material count is more important in endgame
- hanging penalty is more important
"""
# Minor typo: "thisng" → "things", "cersion" → "version"
import tools.GUI as gui
# ok, imports GUI module, fine

# Material table
material_table = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
    'e': 0
}
# fine, basic material values, though 'e' is unused usually

class Engine:
    def __init__(self):
        self.INITIAL_MATERIAL_FACTOR = 1.2
        # ok, initial factor for weighting material
        self.ATTACK_FACTOR = 1.1
        self.DEFENCE_FACTOR=0.9
        # ok, but swapping these dynamically later is dangerous
        self.SQUARE_FACTOR = 0.25
        # scaling control of squares
        self.KING_SAFETY_FACTOR = 1.2
        # fine, weighting king protection
        self.CENTER_FACTOR = 0.7
        # fine, will scale down in endgame
        self.HANGING_FACTOR = 1.0  
        # fine, but later re-assignment is messy

    def count_material(self, board: gui.Board, FEN: str):
        """Calculate total material on the board.min:120,max:1800
        In endgame, material factor increases to give more weight to material."""
        MATERIAL_FACTOR=self.INITIAL_MATERIAL_FACTOR + (20 - 2*board.get_move_count(FEN=FEN))*0.015 if 2*board.get_move_count(FEN=FEN) <20 else 1.5
        # Mistake: calling get_move_count twice, calculation is confusing
        # Better: store move_count in variable
        total = 0
        for row in board.grid:
            for cell in row:
                total += material_table[cell.piece] * MATERIAL_FACTOR
                # ok, sums material, but will crash if cell.piece invalid
        return total

    def hanging_penalty(self, board: gui.Board, color: str,FEN: str):
        """ calculate hanging piece penalty."""
        MATERIAL_FACTOR=self.INITIAL_MATERIAL_FACTOR + (20 - board.get_move_count(FEN=FEN))*0.015 if board.get_move_count(FEN=FEN) <20 else 1.5
        # same problem: repeated call, messy ternary
        self.HANGING_FACTOR=MATERIAL_FACTOR/1.125
        if color=='b':
            self.HANGING_FACTOR,MATERIAL_FACTOR=MATERIAL_FACTOR,self.HANGING_FACTOR
            # Mistake: swapping instance variable and local var → confusing
            self.ATTACK_FACTOR,self.DEFENCE_FACTOR=self.DEFENCE_FACTOR,self.ATTACK_FACTOR
            # Mistake: mutating class state dynamically → bugs later

        penalty=0
        for row in board.grid:
            for cell in row:
                attack=board.get_attackers(board,cell,'w')
                defence=board.get_attackers(board,cell,'b')
                # Mistake: ignores color argument; always white attackers vs black defenders
                eval=sum([(500-material_table[c.piece] )*self.ATTACK_FACTOR for c in attack])-sum([(-500+material_table[c.piece])*self.DEFENCE_FACTOR for c in defence])
                # Mistake: 'eval' shadows builtin eval(), confusing
                if abs(eval)>=100:
                    penalty+=eval*self.HANGING_FACTOR
        return penalty/64
        # fine division to normalize

    def king_squares(self,board:gui.Board,color:str):
        return board.show_moves(board.position_table['K'if color=='w' else 'k'])
        # Mistake: assumes position_table['K'] exists, can crash if king captured

    def square_eval(self, board: gui.Board,color: str,FEN):
        """evaluate control of center squares."""
        self.b_king_squares=self.king_squares(board,'b')
        self.w_king_squares=self.king_squares(board,'w')
        # Mistake: storing king squares as instance variable, will persist across moves → dangerous
        score=0
        for row in board.grid:
            for cell in row:
                if cell!=None:
                    eval=self.cell_eval(board,cell,color,FEN)
                    score+=eval*self.SQUARE_FACTOR
        return (score/64)
        # ok, normalized score

    def cell_eval(self, board: gui.Board,cell:gui.cell,color:str,FEN:str):
        # Mistake: swapping ATTACK_FACTOR/DEFENCE_FACTOR inside each call → state mutation
        CENTER_FACTOR=self.CENTER_FACTOR - (20 - board.get_move_count(FEN))*0.0075 if board.get_move_count(FEN=FEN) <20 else 0.1
        square_constant=100
        attackers=board.get_attackers(board,cell,'w')
        defenders=board.get_attackers(board,cell,'b')
        eval=0                     
        bkings_constant=1
        wkings_constant=1
        if cell in self.b_king_squares:
            bkings_constant=self.KING_SAFETY_FACTOR
        if cell in self.w_king_squares:
            wkings_constant=self.KING_SAFETY_FACTOR
        for ap in attackers:
            square_constant=self.get_square_value(cell,board)* CENTER_FACTOR
            eval+= self.ATTACK_FACTOR*(square_constant / (material_table[ap.piece]/100))*bkings_constant
            # Mistake: dividing by piece value → can be zero division if 'e'
        for dp in defenders:
            eval-= self.DEFENCE_FACTOR*(square_constant * (material_table[dp.piece]/100))*wkings_constant
        return eval*bkings_constant*wkings_constant
        # eval logic is very unstable

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
        # fine, basic square weighting

    def eval(self, board: gui.Board=gui.Board(), FEN: str="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        eval=self.count_material(board,FEN)
        eval += self.hanging_penalty(board, board.turn,FEN=FEN) * self.HANGING_FACTOR
        eval+=self.square_eval(board,board.turn,FEN=FEN)
        return eval
        # Mistake: calling board.turn but passed board may not sync with FEN
        # Mistake: using eval variable shadows builtin

    def move(self,FEN:str="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        board = gui.Board(FEN=FEN)
        best_move = None
        best_eval = -float('inf') if board.turn=='w' else float('inf')
        for r in board.grid:
            for sq in r:
                if (board.turn=='w' and sq.piece.isupper()) or (board.turn=='b' and sq.piece.islower()):
                    moves = board.show_moves(sq)
                    for move in moves:
                        g = gui.Board(FEN=FEN)
                        g.move(sq.pos, move.pos)
                        eval_score = self.eval(board,g.to_FEN())
                        # Mistake: self.eval(board,g.to_FEN()) → eval expects 1 or 2 args differently
                        if (board.turn=='w' and eval_score > best_eval) or (board.turn=='b' and eval_score < best_eval):
                            best_eval = eval_score
                            best_move = (sq.pos, move.pos)
        return best_move
        # ok, greedy move selection, but no depth search

    def play(self):
        b=gui.Board()
        FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 
        while True:
            bot=self.move(FEN)
            print(f'bot plays from {bot[0]}--->{bot[1]}')
            b.move(bot[0],bot[1])
            from_=(input('enter from file a,b,c'),int(input('enter from file 1,2,3')))
            to_=(input('enter to file a,b,c'),int(input('enter to file 1,2,3')))
            b.move(from_,to_)
            FEN=b.to_FEN(b.grid)
            # Mistake: no input validation, can crash
            # Mistake: assumes move() never returns None
            # Mistake: infinite loop with no exit condition

if __name__ == "__main__":
    e=Engine()
    e.play()
