import chess
import math
import chess.engine
STOCKFISH_PATH=r'C:\Users\dell\Desktop\stockshark\tests\stockfish\stockfish-windows-x86-64.exe'
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
           -26,   3,  10,   20,   20,   1,   0, -23,
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
            19,  30,  11,   6,   7,   6,  30,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    chess.ROOK: (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -60, -18,   20,  20, 10, -60, -32),
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
            -4,   3, -14, -50, -57, -30,  13,   4,
            17,  30,  45, -30,   6,  -30,  40,  18),
}
def get_current_material(board):
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        if piece_type == chess.KING:
            continue
        
        total += len(board.pieces(piece_type, chess.WHITE)) * value
        total += len(board.pieces(piece_type, chess.BLACK)) * value

    return total
def king_trop(board: chess.Board):
    """
    Full Stockfish-like king danger evaluation.
    Includes:
    - attack units
    - attack weights
    - king zone control
    - attacker count
    - king tropism (enemy king distance)
    - pawn shield & pawn storm
    Final score: whiteSafety - blackSafety
    """

    # -----------------------
    # Stockfish-style tables
    # -----------------------

    PIECE_ATTACK_WEIGHT = {
        chess.PAWN: 2,
        chess.KNIGHT: 5,
        chess.BISHOP: 5,
        chess.ROOK: 7,
        chess.QUEEN: 10,
        chess.KING: 0
    }

    # penalty table based on total attack units
    KING_DANGER_TABLE = [
        0, 0, 10, 20, 35, 60, 90, 130, 180, 240, 310, 390, 480, 580,
        690, 810, 940, 1080, 1230, 1390, 1560
    ]

    # ------------------------
    # Helper: king tropism
    # ------------------------
    def king_tropism(attacker_square, king_square):
        # Manhattan distance works best for chess engines
        f1, r1 = chess.square_file(attacker_square), chess.square_rank(attacker_square)
        f2, r2 = chess.square_file(king_square), chess.square_rank(king_square)
        return 14 - (abs(f1 - f2) + abs(r1 - r2))

    # ------------------------
    # Evaluate one side's king
    # ------------------------
    def eval_side(color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0

        enemy = not color

        king_zone = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]

        attack_units = 0
        tropism_score = 0
        attackers_count = 0

        # scan enemy pieces attacking king zone
        for sq in board.piece_map():
            piece = board.piece_at(sq)
            if piece.color != enemy:
                continue

            if any(board.attacks(sq) & king_zone):
                attackers_count += 1
                attack_units += PIECE_ATTACK_WEIGHT[piece.piece_type]
                tropism_score += king_tropism(sq, king_sq)

        # get danger index (clamped)
        danger_index = min(attack_units, len(KING_DANGER_TABLE) - 1)
        base_danger = KING_DANGER_TABLE[danger_index]

        # include tropism
        base_danger += tropism_score * 4

        # pawn shield penalty
        def pawn_shield():
            rank = chess.square_rank(king_sq)
            file = chess.square_file(king_sq)
            score = 0
            direction = 1 if color == chess.WHITE else -1

            for df in [-1, 0, 1]:
                f = file + df
                r = rank + direction
                if 0 <= f < 8 and 0 <= r < 8:
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    if piece is None:
                        score += 15  # missing pawn = danger
                    elif piece.color != color or piece.piece_type != chess.PAWN:
                        score += 25  # wrong pawn = bigger danger
            return score

        base_danger += pawn_shield()

        return base_danger

    # ------------------------
    # Final score
    # ------------------------
    white_danger = eval_side(chess.WHITE)
    black_danger = eval_side(chess.BLACK)

    # engine returns GOOD = positive, BAD = negative
    # danger means negative
    return (black_danger - white_danger) / 10  # convert to centipawns

    
    
def evaluate(board: 'chess.Board') -> float:
    """
    Static evaluation function for chess positions.
    
    Combines multiple heuristics to estimate the advantage of one side over another
    without searching deeper. Higher positive scores indicate White's advantage,
    negative scores indicate Black's advantage.
    
    Args:
        board: Current chess board state to evaluate
    
    Returns:
        float: Evaluation score in centipawns (100 = 1 pawn advantage)
            - Positive: White is better
            - Negative: Black is better
            - Zero: Equal position
    
    Evaluation components:
        1. Material: Piece values (Queen=900, Rook=500, etc.)
        2. Positional: Piece-square tables reward good piece placement
        3. King Safety: Penalties for being in check
        4. Mobility: Bonus for having more legal moves (activity)
    """
    
    # =============================================================================
    # CONFIGURATION - Tunable weights for different evaluation factors
    # =============================================================================
    MAX_MATERIAL = 3887 * 2   # both sides
    current_material = get_current_material(board)
    phase = current_material / MAX_MATERIAL

    KING_SAFETY_WEIGHT = 10  # Penalty for being in check (in centipawns)
    MOBILITY_WEIGHT =9.6825# Weight per legal move available
    MATE_VALUE = 32000          #constant for expressing mate value
    BISHOP_PAIR_BASE=40
    POSITIONAL_WEIGHT=0.125
    
    # =============================================================================
    # SCORE ACCUMULATORS - Track different aspects of position
    # =============================================================================
    
    material_score = 0      # Total material difference (White - Black)
    positional_score = 0    # Positional bonuses from piece placement
    king_safety_score = 0   # King safety penalties
    mobility_score = 0      # Advantage from having more moves available
    bishop_pair = 0
    king_tropism=0
    # =============================================================================
    # MATERIAL & POSITIONAL EVALUATION
    # =============================================================================
    
    # returning infinity breaks the alpha beta pruning
    if board.is_checkmate():
        return -MATE_VALUE if board.turn else MATE_VALUE

    # Iterate through each piece type (pawn, knight, bishop, rook, queen, king)
    for piece_type, value in PIECE_VALUES.items():
        # Get all pieces of this type for both colors
        white_pieces = board.pieces(piece_type, chess.WHITE)
        black_pieces = board.pieces(piece_type, chess.BLACK)
        
        # Calculate material difference
        # Positive if White has more/better pieces, negative if Black does
        material_score += (len(white_pieces) - len(black_pieces)) * value
        
        # Evaluate White's piece placement using piece-square tables
        # PST gives bonuses for pieces on good squares (e.g., knights in center)
        # we want the positional advantage to be least considerable in endgame
        for square in white_pieces:
            positional_score += (piece_square_table[piece_type][square])*phase*POSITIONAL_WEIGHT
        
        # Evaluate Black's piece placement
        # Mirror the board (flip vertically) since PST is designed for White's perspective
        for square in black_pieces:
            mirrored_square = chess.square_mirror(square)
            positional_score -=( piece_square_table[piece_type][mirrored_square])*phase*POSITIONAL_WEIGHT
    
    # =============================================================================
    # KING SAFETY EVALUATION
    # =============================================================================
    
    # Being in check is dangerous and should be penalized
    if board.is_check():
        # If White is in check, subtract penalty (bad for White)
        # If Black is in check, add bonus (good for White)
        # this decreases in endgames but mustn't reach 0
        if board.turn == chess.WHITE:
            king_safety_score = KING_SAFETY_WEIGHT * (0.3 + 0.7 * phase)


        else:
            king_safety_score =-KING_SAFETY_WEIGHT * (0.3 + 0.7 * phase)

    
    # =============================================================================
    # MOBILITY EVALUATION
    # =============================================================================
    
    # More legal moves = more tactical options and flexibility
    # Count legal moves for the side to move
    white_board = board.copy()
    white_board.turn = chess.WHITE
    white_moves = len(list(white_board.legal_moves))

    black_board = board.copy()
    black_board.turn = chess.BLACK
    black_moves = len(list(black_board.legal_moves))


    # it decreases in endgame
    mobility_score = (white_moves - black_moves) * MOBILITY_WEIGHT*phase

    #bishop pair advantage calculation
    if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
        bishop_pair += BISHOP_PAIR_BASE
    if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
        bishop_pair -=BISHOP_PAIR_BASE


    # =============================================================================
    # KING TROPISM  EVALUATION
    # =============================================================================
    king_tropism+=king_trop(board)
    # =============================================================================
    # COMBINE ALL FACTORS
    # =============================================================================
    
    # Sum all evaluation components
    # Material is the dominant factor, others provide fine-tuning
    total_evaluation = (
        material_score +      # Dominant factor: piece count and value
        positional_score +    # Piece placement quality
        king_safety_score +   # King safety considerations
        mobility_score+        # Tactical flexibility
        bishop_pair             #bishop pair advantage
    )
    
    return total_evaluation
def ordered_moves(board: 'chess.Board', killer_moves={}, depth=0,history = [[0]*64 for _ in range(64)]
) -> list:
        """
        Orders legal moves using heuristic scoring to optimize alpha-beta pruning.
        Includes killer move heuristic.
        """


        # scoring constants
        PROMOTION_BONUS = 10_000
        KILLER_BONUS     = 9_000      # below promotion, above captures
        CAPTURE_BASE     = 1_000
        CASTLING_BONUS   = 500
        CENTER_BONUS     = 100

        CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}

        move_scores = []

        # retrieve killer moves for this depth
        killers = killer_moves.get(depth, [])

        for move in board.legal_moves:
            score = 0
            target_piece = board.piece_at(move.to_square)

            # -----------------------------------------------------------
            # PRIORITY 0 — KILLER MOVES (quiet only)
            # -----------------------------------------------------------
            if move in killers:
                # only quiet moves should get killer bonus
                if not target_piece and not move.promotion and not board.is_castling(move):
                    score += KILLER_BONUS

            # -----------------------------------------------------------
            # PRIORITY 1 — PROMOTIONS
            # -----------------------------------------------------------
            if move.promotion:
                score += PROMOTION_BONUS

            # -----------------------------------------------------------
            # PRIORITY 2 — CAPTURES (MVV-LVA)
            # -----------------------------------------------------------
            elif target_piece:
                attacker_piece = board.piece_at(move.from_square)
                victim = PIECE_VALUES.get(target_piece.piece_type, 0)
                attacker = PIECE_VALUES.get(attacker_piece.piece_type, 1) if attacker_piece else 1
                score += CAPTURE_BASE * victim - attacker

            # -----------------------------------------------------------
            # PRIORITY 3 — CASTLING
            # -----------------------------------------------------------
            elif board.is_castling(move):
                score += CASTLING_BONUS

            # -----------------------------------------------------------
            # PRIORITY 4 — CENTER CONTROL
            # -----------------------------------------------------------
            elif move.to_square in CENTER_SQUARES:
                score += CENTER_BONUS

            
            # ========================================================
            # Priority 5 - other moves will use history heurestic
            # ========================================================
            else:
                score+=history[move.from_square][move.to_square]
            
            move_scores.append((score, move))

        # sort best first
        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [m for s, m in move_scores]

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
        
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    engine.configure({
        "Threads": 1,
        "Hash": 32
    })

    differences = []

    for name, fen in tests.items():
        board = chess.Board(fen)

        # your eval
        my_eval = evaluate(board)

        # skip invalid positions (prevents SF crash)
        if not board.is_valid():
            print(name, "INVALID FEN, skipping...")
            continue

        # stockfish eval depth-1
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        sf_score = info["score"].white().score(mate_score=32000)

        # difference
        diff = abs(my_eval - sf_score)
        differences.append(diff)

        print(f"{name}:")
        print(f"  FEN:     {fen}")
        print(f"  Yours:   {my_eval}")
        print(f"  SF:      {sf_score}")
        print(f"  Diff:    {diff}")
        print()

        engine.ping()  # prevents engine from dying

    engine.quit()

    avg = sum(differences) / len(differences)
    print(f"\nAVERAGE DIFFERENCE: {avg}")