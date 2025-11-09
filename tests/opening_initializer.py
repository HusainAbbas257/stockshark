import chess
import chess.engine
import json

STOCKFISH_PATH = r"C:\Users\dell\Desktop\stockshark\tests\stockfish\stockfish-windows-x86-64.exe"
OUTPUT_JSON = "data/opening.json"
PLY_DEPTH = 8  # number of half-moves (plies)
TOP_N_MOVES = 2  # keep only top N moves per position for performance

def generate_opening_book(board, depth, engine, book):
    if depth == 0 or board.is_game_over():
        return

    fen = board.board_fen()
    turn = "w" if board.turn == chess.WHITE else "b"
    key = f"{fen} {turn} KQkq - 0 1"

    if key in book:
        return  # already processed

    # Use Stockfish to get top moves
    info = engine.analyse(board, chess.engine.Limit(depth=1))
    legal_moves = list(board.legal_moves)
    
    # Score moves using Stockfish
    scored_moves = []
    for move in legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=2))
        score = info["score"].white().score(mate_score=10000)  # +ve good for white
        board.pop()
        if score is None:  # in case of mate
            score = 0
        scored_moves.append((move.uci(), score))

    # Sort moves by score (descending if white, ascending if black)
    scored_moves.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
    best_moves = [m[0] for m in scored_moves[:TOP_N_MOVES]]

    book[key] = best_moves

    # Recursively generate for each best move
    for move_uci in best_moves:
        board.push(chess.Move.from_uci(move_uci))
        generate_opening_book(board, depth-1, engine, book)
        board.pop()

if __name__ == "__main__":
    board = chess.Board()
    book = {}
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        generate_opening_book(board, PLY_DEPTH, engine, book)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(book, f, indent=2)

    print(f"Opening book up to {PLY_DEPTH} ply saved to {OUTPUT_JSON}")
