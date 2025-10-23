import tools.GUI as gui

# Material table
material_table = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000,
    'e': 0
}

# Multipliers
MATERIAL_FACTOR = 1.2
ATTACK_FACTOR = 0.4
SQUARE_FACTOR = 0.25
KING_SAFETY_FACTOR = 1.2
TRADE_FACTOR = 0.8
CENTER_FACTOR = 0.7
HANGING_FACTOR = 1.0  # multiply penalty inside hanging_penalty




class Engine:
    def count_material(self, board: gui.Board):
        total = 0
        for row in board.grid:
            for cell in row:
                total += material_table[cell.piece] * MATERIAL_FACTOR
        return total
    def hanging_penalty(self, board: gui.Board, color: str):
        penalty = 0
        for row in board.grid:
            for cell in row:
                if cell.piece == 'e': continue
                is_mine = cell.piece.isupper() if color=='w' else cell.piece.islower()
                if is_mine:
                    attacked = False
                    for r in board.grid:
                        for sq in r:
                            if (sq.piece != 'e' and (sq.piece.isupper() != cell.piece.isupper())):
                                if cell in board.show_moves(sq):
                                    attacked = True
                                    break
                        if attacked: break
                    if attacked:
                        penalty -= abs(material_table[cell.piece])  # big negative
        return penalty

    def king(self, board: gui.Board, color: str):
        for r in board.grid:
            for cell in r:
                if cell.piece == 'K' and color=='w':
                    return cell
                if cell.piece == 'k' and color=='b':
                    return cell
        return None

    def king_safety(self, board: gui.Board, color: str):
        king_cell = self.king(board, color)
        if king_cell is None:
            return -1000000 if color=='w' else 1000000 # King missing, extreme penalty
        moves = board.show_moves(king_cell)
        safety_scores = [0] 
        for cell in moves:
            safety_scores.append(self.square_eval(board, cell))
        return max(safety_scores) if color=='w' else min(safety_scores)

    def square_eval(self, board: gui.Board, cell: gui.cell):
        col, row = cell.pos
        square_val = 5

        if col in ['d','e'] and str(row) in ['4','5']:
            square_val = 30
        elif col in ['c','f'] and str(row) in ['3','6']:
            square_val = 20
        elif col in ['b','g'] and str(row) in ['2','7']:
            square_val = 10
        square_val*= CENTER_FACTOR
        square_val *= material_table[cell.piece]/100 if cell.piece!='e' else 1

        attackers = 0
        for r in board.grid:
            for sq in r:
                if cell in board.show_moves(sq):
                    attackers += TRADE_FACTOR*material_table[sq.piece]

        return ATTACK_FACTOR*attackers + SQUARE_FACTOR*square_val

    def eval(self, FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        board = gui.Board(FEN=FEN)
        value = self.count_material(board)
        for row in board.grid:
            for cell in row:
                value += self.square_eval(board, cell)/64
        value += KING_SAFETY_FACTOR*(self.king_safety(board,'w') + self.king_safety(board,'b'))
        return value+ self.hanging_penalty(board,'w') + self.hanging_penalty(board,'b')

    def next_move(self, FEN, turn):
        board = gui.Board(FEN=FEN)
        best_move = None
        best_eval = -float('inf') if turn=='w' else float('inf')

        for r in board.grid:
            for sq in r:
                if (turn=='w' and sq.piece.isupper()) or (turn=='b' and sq.piece.islower()):
                    moves = board.show_moves(sq)
                    for move in moves:
                        g = gui.Board(FEN=FEN)
                        g.move(sq.pos, move.pos)
                        eval_score = self.eval(g.to_FEN())
                        if (turn=='w' and eval_score > best_eval) or (turn=='b' and eval_score < best_eval):
                            best_eval = eval_score
                            best_move = (sq.pos, move.pos)
        return best_move


    def play(self, color='wb', FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", max_moves=50):
        import time
        f = FEN
        move_count = 0

        while move_count < max_moves:
            turn = f.split()[1]

            if turn in color:  # engine plays this turn
                move = self.next_move(f, turn)
                if move is None:
                    print(f"No legal moves for {turn}. Game over!")
                    break

                g = gui.Board(FEN=f)
                g.move(move[0], move[1])
                f = g.to_FEN()

                move_str = f"{move[0][0]}{move[0][1]}-{move[1][0]}{move[1][1]}"
                turn_name = "White" if turn == 'w' else "Black"
                print(f"{turn_name} plays: {move_str}")
                move_count += 1


                # check for game end
                legal_moves_exist = False
                for r in g.grid:
                    for c in r:
                        if (turn=='w' and c.piece.isupper()) or (turn=='b' and c.piece.islower()):
                            if g.show_moves(c):
                                legal_moves_exist = True
                                break
                    if legal_moves_exist:
                        break
                if not legal_moves_exist:
                    if g.is_in_check(turn):
                        print(f"Checkmate! {'Black' if turn=='w' else 'White'} wins")
                    else:
                        print("Stalemate!")
                    break

            # switch turn automatically
            f = f.replace(f" {turn} ", f" {'b' if turn=='w' else 'w'} ", 1)

        return f  # return final FEN


if __name__ == "__main__":
    engine = Engine()
    engine.play(color='w')
