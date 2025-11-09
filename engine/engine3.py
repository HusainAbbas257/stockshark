import chess
import chess.pgn

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



def get_positional_value(piece, square):
    if not piece:
        return 0
    table = PIECE_SQUARE_TABLES.get(piece.symbol().upper(), [0]*64)
    idx = square if piece.color == chess.WHITE else 63 - square
    val = table[idx]
    return val if piece.color == chess.WHITE else -val

def evaluate_board(board: 'chess.Board'):
    """Material + positional + mobility evaluation."""
    value = 0

    # Material evaluation
    for piece_type, val in PIECE_VALUES.items():
        white_squares = board.pieces(piece_type, chess.WHITE)
        black_squares = board.pieces(piece_type, chess.BLACK)

        value += len(white_squares) * val
        value -= len(black_squares) * val

        # Positional evaluation
        for sq in white_squares:
            piece = board.piece_at(sq)
            value += get_positional_value(piece, sq)
        for sq in black_squares:
            piece = board.piece_at(sq)
            value += get_positional_value(piece, sq)  # will return negative automatically for black

    # Mobility evaluation
    mobility = len(list(board.legal_moves))
    value += 0.1 * mobility if board.turn == chess.WHITE else -0.1 * mobility

    return value


def ordered_moves(board:'chess.Board'):
    moves = list(board.legal_moves)
    def score(move):
        if board.is_capture(move): return 4
        if board.gives_check(move): return 3
        if board.can_claim_draw():return 1
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            return 0
        return 1
    return sorted(moves, key=score, reverse=True)

def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    if maximizing:
        max_eval = float('-inf')
        for move in ordered_moves(board):
            piece = board.piece_at(move.from_square)
            if depth > 2 and piece and piece.piece_type == chess.PAWN:
                continue
            board.push(move)
            eval = minimax(board, depth-1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves(board):
            piece = board.piece_at(move.from_square)
            if depth > 2 and piece and piece.piece_type == chess.PAWN:
                continue
            board.push(move)
            eval = minimax(board, depth-1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth=3):
    best_move = None
    if board.turn == chess.WHITE:
        best_value = float('-inf')
        for move in ordered_moves(board):
            board.push(move)
            value = minimax(board, depth-1, float('-inf'), float('inf'), False)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
    else:
        best_value = float('inf')
        for move in ordered_moves(board):
            board.push(move)
            value = minimax(board, depth-1, float('-inf'), float('inf'), True)
            board.pop()
            if value < best_value:
                best_value = value
                best_move = move
    return best_move



def self_play(depth=3):
    board = chess.Board()
    move_num = 1
    while not board.is_game_over() and move_num <= 150:
        move = find_best_move(board, depth)
        print(f"Move {move_num}: {'White' if board.turn else 'Black'} plays {move}")
        board.push(move)
        move_num += 1
    print("Game over:", board.result())

def play_against_human(depth=3, human_color=chess.BLACK):
    board = chess.Board()
    move_num = 1
    print("You are playing", "White" if human_color == chess.WHITE else "Black")
    while not board.is_game_over() and move_num <= 150:
        if board.turn == human_color:
            print(board)
            move_uci = input("Your move in UCI (e2e4): ")
            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    print("Illegal move, try again.")
                    continue
            except:
                print("Invalid format, try again.")
                continue
        else:
            move = find_best_move(board, depth)
            print(f"Engine plays: {move}")
        board.push(move)
        move_num += 1
    print(board)
    print("Game over:", board.result())

if __name__ == "__main__":
    # self_play(depth=4) 
    # # for engine vs engine
    play_against_human(depth=3, human_color=chess.WHITE)