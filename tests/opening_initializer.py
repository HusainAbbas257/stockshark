import chess
import chess.pgn
import json

INPUT_PGN = r"C:\Users\dell\Desktop\stockshark\data\master databse.pgn"
OUTPUT_JSON = r"C:\Users\dell\Desktop\stockshark\data\opening.json"

book = {}

def add_move(fen, move):
    if fen not in book:
        book[fen] = []
    uci = move.uci()
    if uci not in book[fen]:
        book[fen].append(uci)
count=0
with open(INPUT_PGN, encoding="utf-8", errors="ignore") as pgn:
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        board = game.board()
        ply_count = 0  # 1 ply = 1 move by either side
        count+=1
        if count%100==0:
            print(count)
        for move in game.mainline_moves():
            if ply_count >= 40:  # 40 plies = 20 full moves
                break

            fen = board.fen()
            add_move(fen, move)
            board.push(move)
            ply_count += 1

with open(OUTPUT_JSON, "w") as f:
    json.dump(book, f, indent=2)

print("Opening.json generated successfully!")
