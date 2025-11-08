import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def evaluate_board(board):
    """Simple material + mobility evaluation."""
    value = 0
    for piece_type in PIECE_VALUES:
        value += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]

    # Add slight bonus for mobility (active positions)
    mobility = len(list(board.legal_moves))
    if board.turn == chess.WHITE:
        value += 0.1 * mobility
    else:
        value -= 0.1 * mobility

    return value

def minimax(board, depth, is_maximizing):
    """Basic minimax without alpha-beta pruning."""
    if depth == 0 or board.is_game_over() or board.is_repetition():
        return evaluate_board(board)

    if is_maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval

def find_best_move(board, depth=2):
    """Finds the best move for the current player."""
    best_move = None
    if board.turn == chess.WHITE:
        best_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, False)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
    else:
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, True)
            board.pop()
            if value < best_value:
                best_value = value
                best_move = move
    return best_move

def self_play(depth=2):
    """Engine plays both sides automatically."""
    board = chess.Board()
    move_num = 1

    while not board.is_game_over() and move_num<=150:
        move = find_best_move(board, depth)
        print(f"Move {move_num}: {'White' if board.turn else 'Black'} plays {move}")
        board.push(move)
        # print(board, "\n")
        move_num += 1

    print("Game over:", board.result())

if __name__ == "__main__":
    self_play(depth=3)
