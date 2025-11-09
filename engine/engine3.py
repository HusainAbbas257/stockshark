import chess
import chess.pgn
import random
import json

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100
}
PIECE_SQUARE_TABLES = {
    'P': [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 5, 5, 5,
        1, 1, 2, 3, 3, 2, 1, 1,
        0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5,
        0, 0, 0, 2, 2, 0, 0, 0,
        0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5,
        0.5, 1, 1, -2, -2, 1, 1, 0.5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    'N': [
        -5, -4, -3, -3, -3, -3, -4, -5,
        -4, -2, 0, 0, 0, 0, -2, -4,
        -3, 0, 1, 1.5, 1.5, 1, 0, -3,
        -3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3,
        -3, 0, 1.5, 2, 2, 1.5, 0, -3,
        -3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3,
        -4, -2, 0, 0.5, 0.5, 0, -2, -4,
        -5, -4, -3, -3, -3, -3, -4, -5
    ],
    'B': [
        -2, -1, -1, -1, -1, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 1, 1, 0.5, 0, -1,
        -1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1,
        -1, 0, 1, 1, 1, 1, 0, -1,
        -1, 1, 1, 1, 1, 1, 1, -1,
        -1, 2, 0, 0, 0, 0, 2, -1,
        -2, -1, -1, -1, -1, -1, -1, -2
    ],
    'R': [
        0, 0, 0, 0, 0, 0, 0, 0,
        0.5, 1, 1, 1, 1, 1, 1, 0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        0, 0, 0, 0.5, 0.5, 0, 0, 0
    ],
    'Q': [
        -2, -1, -1, -0.5, -0.5, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
        0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0,
        -1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -2, -1, -1, -0.5, -0.5, -1, -1, -2
    ],
    'K': [
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -3, -4, -4, -5, -5, -4, -4, -3,
        -2, -3, -3, -4, -4, -3, -3, -2,
        -1, -2, -2, -2, -2, -2, -2, -1,
        2, 2, 0, 0, 0, 0, 2, 2,
        2, 3, 1, 0, 0, 1, 3, 2
    ]
}



def ordered_moves(board):
    moves = list(board.legal_moves)
    scores = {}

    for move in moves:
        score = 0
        piece = board.piece_at(move.from_square)
        target = board.piece_at(move.to_square)

        # MVV-LVA
        if target:
            score += 1000 + 10 * PIECE_VALUES[target.piece_type] - PIECE_VALUES[piece.piece_type]

        # Promotions
        if move.promotion:
            score += 800 + PIECE_VALUES[move.promotion] * 10

        # Castling
        if board.is_castling(move):
            score += 500

        # Positional heuristic (lightweight)
        score += get_positional_value(piece, move.to_square) * 0.5

        scores[move] = score

    return sorted(moves, key=lambda m: scores[m], reverse=True)


# Add these constants near the top of your file
POSITIONAL_CONSTANT = 0.20
MOBILITY_CONSTANT = 0.05
KING_SAFETY_CONSTANT = 0.2
PASSED_PAWN_BONUS = 0.4
TRAPPED_PENALTY = 0.5
KING_ATTACK_WEIGHT = 0.4  
MATE_THREAT_BONUS = 1.0



# ---------- Fixed functions ----------
def get_positional_value(piece, square):
    """Return positional bonus for a piece on square (white positive, black negative)."""
    if not piece:
        return 0.0
    table = PIECE_SQUARE_TABLES.get(piece.symbol().upper(), [0]*64)
    idx = square if piece.color == chess.WHITE else 63 - square
    # table values already in pawn/centipawn-like scale; apply positional constant here
    return table[idx] * POSITIONAL_CONSTANT

def king_safety(board: 'chess.Board'):
    """
    Evaluates king safety: more attackers near king = worse.
    Positive = white safer, negative = black safer.
    """
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is None:
            continue

        # squares in 1-square radius around king
        danger_zone = [sq for sq in chess.SQUARES if chess.square_distance(sq, king_sq) <= 1]

        # count enemy attacks on those squares
        danger = sum(1 for sq in danger_zone if board.is_attacked_by(not color, sq))

        # each attack near king slightly hurts that color
        penalty = danger * KING_SAFETY_CONSTANT
        score += -penalty if color == chess.WHITE else penalty  # white exposed → negative

    return score


def evaluate_board(board: 'chess.Board'):
    """
    Material + positional + mobility + king safety + dynamic heuristics.
    Positive = white advantage.
    """
    value = 0.0

    # --- Material + positional ---
    for piece_type, val in PIECE_VALUES.items():
        whites = board.pieces(piece_type, chess.WHITE)
        blacks = board.pieces(piece_type, chess.BLACK)

        value += len(whites) * val - len(blacks) * val

        for sq in whites:
            value += get_positional_value(board.piece_at(sq), sq)
        for sq in blacks:
            value -= get_positional_value(board.piece_at(sq), sq)

    # --- Mobility ---
    tmp = board.copy()
    tmp.turn = chess.WHITE
    mobility_white = len(list(tmp.legal_moves))
    tmp.turn = chess.BLACK
    mobility_black = len(list(tmp.legal_moves))
    value += MOBILITY_CONSTANT * (mobility_white - mobility_black)

    # --- King Safety ---
    value += king_safety(board)

    # --- Passed Pawns ---
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.pieces(chess.PAWN, color):
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            enemy_pawns = board.pieces(chess.PAWN, not color)
            same_file_enemy = any(chess.square_file(p) == file for p in enemy_pawns)
            if not same_file_enemy:
                bonus = PASSED_PAWN_BONUS * (rank if color == chess.WHITE else 7 - rank)
                value += bonus if color == chess.WHITE else -bonus

    # --- Trapped pieces ---
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        if len(list(board.attacks(sq))) == 0:  # piece has no moves
            trap_val = TRAPPED_PENALTY
            value += -trap_val if piece.color == chess.WHITE else trap_val

    # --- Mate threat ---
    if board.is_check():
        value += MATE_THREAT_BONUS * (1 if board.turn == chess.BLACK else -1)

    # --- King attacks proximity ---
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(not color)
        if king_sq is None:
            continue
        attackers = sum(
            1 for sq in board.attacks(king_sq)
            if board.piece_at(sq) and board.piece_at(sq).color == color
        )
        value += KING_ATTACK_WEIGHT * attackers * (1 if color == chess.WHITE else -1)

    return round(value, 2)


class Engine:
    def __init__(self, opening_file=None):
        self.opening_book = {}
        if opening_file:
            with open(opening_file, 'r') as f:
                self.opening_book = json.load(f)
        self.previous_positions = {}  # track board positions

    def reset_position_history(self):
        self.previous_positions = {}

    def record_position(self, board):
        fen = board.board_fen()
        turn = "w" if board.turn == chess.WHITE else "b"
        key = f"{fen} {turn}"
        self.previous_positions[key] = self.previous_positions.get(key, 0) + 1
        return self.previous_positions[key]

    def book_move(self, board:'chess.Board'):
        fen = board.board_fen()
        turn = "w" if board.turn == chess.WHITE else "b"
        key = f"{fen} {turn} KQkq - 0 1"
        moves = self.opening_book.get(key)
        if moves:
            move_uci = random.choice(moves)
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                return move
        return None

    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return evaluate_board(board)
        if maximizing:
            max_eval = float('-inf')
            for move in ordered_moves(board):
                board.push(move)
                repetition_count = self.record_position(board)
                if repetition_count >= 3:
                    board.pop()
                    continue
                eval = self.minimax(board, depth-1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves(board):
                board.push(move)
                repetition_count = self.record_position(board)
                if repetition_count >= 3:
                    board.pop()
                    continue
                eval = self.minimax(board, depth-1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

    def find_best_move(self, board, depth=3):
        # first check opening book
        move = self.book_move(board)
        if move:
            return move

        move = None
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        legal_moves = ordered_moves(board)
        for m in legal_moves:
            board.push(m)
            repetition_count = self.record_position(board)
            if repetition_count >= 3:
                board.pop()
                continue
            value = self.minimax(board, depth-1, float('-inf'), float('inf'), not board.turn)
            board.pop()
            if (board.turn == chess.WHITE and value > best_value) or (board.turn == chess.BLACK and value < best_value):
                best_value = value
                best_move = m
        # if all moves blocked by repetition, pick random legal
        if best_move is None and legal_moves:
            best_move = random.choice(list(legal_moves))
        return best_move

    def self_play(self, depth=3, max_moves=150):
        board = chess.Board()
        move_num = 1
        self.reset_position_history()
        while not board.is_game_over() and move_num <= max_moves:
            move = self.find_best_move(board, depth)
            if move is None:
                print("No legal move found even randomly.")
                break
            print(f"Move {move_num}: {'White' if board.turn else 'Black'} plays {move}")
            board.push(move)
            self.record_position(board)
            move_num += 1
        print("Game over:", board.result())
    def play_against_human(self, depth=3, human_color=chess.WHITE):
        board = chess.Board()
        self.reset_position_history()
        game = chess.pgn.Game()
        node = game

        while not board.is_game_over():
            if board.turn == human_color:
                # print(board)
                move_input = input("Enter your move in UCI format (e.g., e2e4): ")
                move = chess.Move.from_uci(move_input)
                if move not in board.legal_moves:
                    print("Illegal move! Try again.")
                    continue
            else:
                move = self.find_best_move(board, depth)
                if move is None:
                    print("No legal move found for engine!")
                    break
                print(f"Engine plays: {move}")

            board.push(move)
            self.record_position(board)
            node = node.add_variation(move)

        print("Game over:", board.result())
        print("\nPGN of the game:\n")
        print(game)


if __name__ == "__main__":
    e=Engine('data/opening.json')
    e.self_play(3)
