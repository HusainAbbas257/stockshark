
import chess
import chess.engine
from engine import engine2 as en 

STOCKFISH_PATH = r"C:\Users\dell\Desktop\stockshark\tests\stockfish\stockfish-windows-x86-64.exe"

def main():
    e = en.Engine()
    board = chess.Board()
    
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            FEN = board.fen()
            move, eval_score = e.next_move(FEN, depth=5, lines=5)
            if move is None:
                print("Your engine has no moves. Game over.")
                break

            from_pos, to_pos = move
            uci_move = f"{from_pos[0]}{from_pos[1]}{to_pos[0]}{to_pos[1]}".lower()

            try:
                chess_move = chess.Move.from_uci(uci_move)
                if chess_move in board.legal_moves:
                    board.push(chess_move)
                    print(f"Your Engine plays: {uci_move}")
                else:
                    print(f"Illegal move attempted by engine: {uci_move}")
                    # fallback: play random legal move
                    board.push(random.choice(list(board.legal_moves)))
                    print(f"Fallback move played: {board.peek()}")
            except Exception as ex:
                print(f"Error parsing move: {uci_move}, {ex}")
                board.push(random.choice(list(board.legal_moves)))
                print(f"Fallback move played: {board.peek()}")

        else:
            # Stockfish plays black
            result = sf.play(board, chess.engine.Limit(time=0.01))
            board.push(result.move)
            print(f"Stockfish plays: {result.move}")

    print("Game Over")
    print("Result:", board.result())
    sf.quit()

if __name__ == "__main__":
    import random
    main()
