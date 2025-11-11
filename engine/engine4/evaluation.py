import chess
import chess.pgn
import random
import json



# values set according to stockfish
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 280,   
    chess.BISHOP: 320,  
    chess.ROOK: 479,      
    chess.QUEEN: 929,     
    chess.KING: 60000      
}


#Big thanks to : https://github.com/thomasahle/sunfish.git
piece_square_table= {
     chess.PAWN: (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    chess.KNIGHT: ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    chess.BISHOP: ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    chess.ROOK: (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    chess.QUEEN: (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    chess.KING: (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
def evaluate(board: 'chess.Board'):
    value = 0
    for piece_type, val in PIECE_VALUES.items():
        whites = board.pieces(piece_type, chess.WHITE)
        blacks = board.pieces(piece_type, chess.BLACK)

        # Material count
        value += len(whites) * val - len(blacks) * val

        # Positional count (without adding material again)
        for sq in whites:
            value += piece_square_table[piece_type][sq]
        for sq in blacks:
            value -= piece_square_table[piece_type][63 - sq]
    return value

def ordered_moves(board: 'chess.Board'):
    moves = board.legal_moves
    move_scores = []

    for move in moves:
        score = 0
        target = board.piece_at(move.to_square)

        # promotions first
        if move.promotion:
            score += 10_000
        elif target:
            # MVV-LVA simplified
            score += 1000 * PIECE_VALUES[target.piece_type]
        elif board.is_castling(move):
            score += 300

        move_scores.append((score, move))

    # sort only once — O(n log n)
    move_scores.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in move_scores]

if __name__ == "__main__":
    tests = {
    # --- Opening Positions ---
    "Start Position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Ruy Lopez": "rnbqkbnr/pppp1ppp/8/4p3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    "Italian Game": "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2",
    "Scotch Game": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Sicilian Defense": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "French Defense": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "Caro-Kann": "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Pirc Defense": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "Kings Indian": "rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2",
    "English Opening": "rnbqkbnr/pppppppp/8/8/4P3/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",

    # --- Midgame Situations ---
    "White +Pawn": "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    "Black +Rook": "rnbqkbnr/pppppppp/8/8/8/8/PPP5/RNBQKBNR w KQkq - 0 1",
    "Center Locked": "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 2 3",
    "Open Center": "rnbqkb1r/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3",
    "Exposed Kings": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "Trapped Bishop": "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 2 2",
    "Weak Pawns": "rnbqkb1r/pp1ppppp/2p5/8/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "Passed Pawn": "8/8/3k4/8/3P4/8/8/3K4 w - - 0 1",
    "Isolated Pawn": "rnbqkb1r/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 4",
    "Doubled Pawns": "rnbqkbnr/ppp2ppp/8/3pp3/8/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 4",

    # --- Tactical Patterns ---
    "Mate Threat": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 3",
    "King Attack": "rnb1kbnr/pppp1ppp/8/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 4 4",
    "Pinned Knight": "rnbqkbnr/pppp1ppp/8/4p3/2B5/5N2/PPPPPPPP/RNBQK2R b KQkq - 3 3",
    "Fork Threat": "rnbqkbnr/pppp1ppp/8/4p3/2B5/8/PPPPPPPP/RNBQK1NR w KQkq - 2 3",
    "Skewer Threat": "r3k2r/pppqppbp/2np1np1/8/2BPP3/2N2N2/PPP2PPP/R1BQ1RK1 w kq - 0 7",
    "Discovered Attack": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 2 4",
    "Hanging Piece": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPPBPPP/R1BQK2R w KQkq - 2 4",
    "Back Rank Weakness": "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1",
    "Overextended Pawns": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    "King Open File": "rnbqkbnr/pppp1ppp/8/4p3/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 2",

    # --- Endgames ---
    "King and Pawn": "8/8/3k4/8/3P4/8/8/3K4 w - - 0 1",
    "Rook Endgame": "8/8/3k4/8/3R4/8/8/3K4 w - - 0 1",
    "Knight Endgame": "8/8/3k4/8/3N4/8/8/3K4 w - - 0 1",
    "Bishop Endgame": "8/8/3k4/8/3B4/8/8/3K4 w - - 0 1",
    "Opposite Bishops": "8/8/3k4/8/3B4/8/8/3K2b1 w - - 0 1",
    "Rook vs Bishop": "8/8/3k4/8/3R4/8/8/3K2b1 w - - 0 1",
    "Knight vs Bishop": "8/8/3k4/8/3N4/8/8/3K2b1 w - - 0 1",
    "Pawn Race": "8/3k4/8/3P4/8/8/4p3/3K4 w - - 0 1",
    "Two Queens": "8/8/8/3k4/3Q4/8/3Q4/3K4 w - - 0 1",
    "Underpromotion": "8/3k4/8/8/3P4/8/8/3K4 w - - 0 1",

    # --- Imbalanced Positions ---
    "Queen Sacrifice": "rnb1kbnr/ppppqppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 3",
    "Material Imbalance": "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPP2PPP/RNBQKBNR w KQkq - 0 2",
    "Two Bishops Advantage": "rnbqkbnr/pppppppp/8/8/8/8/PPP2PPP/RNBQKBBR w KQkq - 0 1",
    "Rook Lift": "rnbqkbnr/pppppppp/8/8/3R4/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Center Break": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "Advanced Pawn Chain": "rnbqkbnr/pppppppp/8/8/3PPP2/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    "King Walk": "8/8/3k4/8/4K3/8/8/8 w - - 0 1",
    "Knight Outpost": "rnbqkbnr/pppppppp/8/8/3N4/8/PPP2PPP/R1BQKBNR b KQkq - 0 3",
    "Pawn Storm": "rnbq1rk1/ppppppbp/6p1/8/3PP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 6",
}
    b=chess.Board()
    for name in tests:
        b.set_fen(tests[name])
        print(f'{name}-->{evaluate(b)}')
