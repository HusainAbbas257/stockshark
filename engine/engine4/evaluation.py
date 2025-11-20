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
opening_values= {
    "KING_SAFETY_WEIGHT": 4.21875,
    "MOBILITY_WEIGHT": 0.421875,
    "MATE_VALUE": 6000.0,
    "BISHOP_PAIR_BASE": 2.8125,
    "POSITIONAL_WEIGHT": 0.0087890625,
    "KING_TROP_WEIGHT": 0.0703125,
    "DEFENCE_FACTOR": 0.0703125,
    "ATTACK_FACTOR": 0.0665625,
    "PAWN_CHAIN": 6.0,
    "PASSED_PAWN": 20.0,
    "PAWN_FACTOR": 0.5,
}
middlegame_values = {
    "KING_SAFETY_WEIGHT": 3.75,
    "MOBILITY_WEIGHT": 2.25,
    "MATE_VALUE": 6000.0,
    "BISHOP_PAIR_BASE": 9.375,
    "POSITIONAL_WEIGHT": 0.028125,
    "KING_TROP_WEIGHT": 0.375,
    "DEFENCE_FACTOR": 0.1875,
    "ATTACK_FACTOR": 0.375,
    "PAWN_CHAIN": 6.0,
    "PASSED_PAWN": 20.0,
    "PAWN_FACTOR": 0.5,
}

endgame_values = {
    "KING_SAFETY_WEIGHT": 0.5,
    "MOBILITY_WEIGHT": 3.75,
    "MATE_VALUE": 8000.0,
    "BISHOP_PAIR_BASE": 5.0,
    "POSITIONAL_WEIGHT": 0.025,
    "KING_TROP_WEIGHT": 0.125,
    "DEFENCE_FACTOR": 0.25,
    "ATTACK_FACTOR": 0.25,
    "PAWN_CHAIN": 6.0,
    "PASSED_PAWN": 20.0,
    "PAWN_FACTOR": 0.5
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
# evaluation.py
def calculate_mobility(board: chess.Board) -> float:
    """
    Calculate mobility difference (white_moves - black_moves) without mutating the input board.
    Uses a cheap copy to flip turn and count legal moves.
    """
    # Count mobility for current side quickly
    # Use shallow copies so original board is not mutated
    white_board = board.copy()
    white_board.turn = chess.WHITE
    white_mobility = sum(1 for _ in white_board.legal_moves)

    black_board = board.copy()
    black_board.turn = chess.BLACK
    black_mobility = sum(1 for _ in black_board.legal_moves)

    return white_mobility - black_mobility

def get_current_material(board):
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        if piece_type == chess.KING:
            continue
        
        total += len(board.pieces(piece_type, chess.WHITE)) * value
        total += len(board.pieces(piece_type, chess.BLACK)) * value

    return total

def pawn_factor(board: chess.Board, values: dict[str, int]) -> float:
    """
    Evaluate pawn structure:
    - pawn chains
    - passed pawns
    """
    score = 0.0

    pawn_chain_bonus = values.get("PAWN_CHAIN", 10)
    passed_pawn_bonus = values.get("PASSED_PAWN", 30)

    # loop over all pawns
    for square in board.pieces(chess.PAWN, chess.WHITE):
        score += evaluate_pawn(board, square, chess.WHITE,
                               pawn_chain_bonus, passed_pawn_bonus)

    for square in board.pieces(chess.PAWN, chess.BLACK):
        score -= evaluate_pawn(board, square, chess.BLACK,
                               pawn_chain_bonus, passed_pawn_bonus)

    return score


def evaluate_pawn(board, sq, color, chain_bonus, passed_bonus):
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)

    bonus = 0

    # --- Pawn Chain ---
    # friendly pawn on left or right diagonal behind?
    for df in (-1, 1):
        nf = file + df
        nr = rank - 1 if color == chess.WHITE else rank + 1
        if 0 <= nf <= 7 and 0 <= nr <= 7:
            if board.piece_at(chess.square(nf, nr)) == chess.Piece(chess.PAWN, color):
                bonus += chain_bonus

    # --- Passed Pawn ---
    # no enemy pawns ahead on same file or adjacent?
    passed = True
    direction = 1 if color == chess.WHITE else -1

    for df in (-1, 0, 1):
        nf = file + df
        if nf < 0 or nf > 7:
            continue

        # check all ranks ahead
        current_rank = rank + direction
        while 0 <= current_rank <= 7:
            sq_check = chess.square(nf, current_rank)
            piece = board.piece_at(sq_check)
            if piece and piece.piece_type == chess.PAWN and piece.color != color:
                passed = False
                break
            current_rank += direction

    if passed:
        bonus += passed_bonus

    return bonus
def get_neighbour(board: chess.Board, piece: chess.Piece) -> list:
    # find piece square
    square = None
    for sq in chess.SQUARES:
        if board.piece_at(sq) == piece:
            square = sq
            break
    
    if square is None:
        return []

    neighbours = []
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    # 8 surrounding directions
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr = rank + dr
            nf = file + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                neighbours.append(chess.square(nf, nr))

    return neighbours


def king_trop(board: chess.Board,weight):
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    trop_white = 0
    trop_black = 0

    # WHITE KING
    for nb in get_neighbour(board, board.piece_at(wk)):
        piece = board.piece_at(nb)

        # defence: friendly pieces near king
        if piece and piece.color == chess.WHITE:
            trop_white += weight['DEFENCE_FACTOR']

        # attack: enemy pieces attacking that square
        attackers = board.attackers(chess.BLACK, nb)
        if attackers:
            trop_white -= weight['ATTACK_FACTOR'] * len(attackers)

    # BLACK KING
    for nb in get_neighbour(board, board.piece_at(bk)):
        piece = board.piece_at(nb)

        if piece and piece.color == chess.BLACK:
            trop_black += weight['DEFENCE_FACTOR']

        attackers = board.attackers(chess.WHITE, nb)
        if attackers:
            trop_black -= weight['ATTACK_FACTOR'] * len(attackers)

    # return a score (white positive → black king worse)
    return trop_white - trop_black
    
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
    MAX_MATERIAL = 3887 * 2
    current_material = get_current_material(board)
    phase = current_material / MAX_MATERIAL   # 1 = opening, 0 = endgame

    # ---- Pick phase weights ----
    if phase > 0.66:
        # Opening phase
        weights = opening_values.copy()

    elif phase > 0.33:
        # Middlegame phase
        weights = middlegame_values.copy()

    else:
        # Endgame phase
        weights = endgame_values.copy()

    # =============================================================
    # SCORE ACCUMULATORS - Track different aspects of position
    # =============================================================================
    
    material_score = 0      # Total material difference (White - Black)
    positional_score = 0    # Positional bonuses from piece placement
    king_safety_score = 0   # King safety penalties
    mobility_score = 0      # Advantage from having more moves available
    bishop_pair = 0
    king_tropism=0
    pawn_eval=0
    # =============================================================================
    # MATERIAL & POSITIONAL EVALUATION
    # =============================================================================
    
    # returning infinity breaks the alpha beta pruning
    if board.is_checkmate():
        return -weights['MATE_VALUE'] if board.turn else weights['MATE_VALUE']

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
            positional_score += (piece_square_table[piece_type][square])*weights['POSITIONAL_WEIGHT']
        
        # Evaluate Black's piece placement
        # Mirror the board (flip vertically) since PST is designed for White's perspective
        for square in black_pieces:
            mirrored_square = chess.square_mirror(square)
            positional_score -=( piece_square_table[piece_type][mirrored_square])*weights['POSITIONAL_WEIGHT']
    
    # =============================================================================
    # KING SAFETY EVALUATION
    # =============================================================================
    
    # Being in check is dangerous and should be penalized
    if board.is_check():
        # If White is in check, subtract penalty (bad for White)
        # If Black is in check, add bonus (good for White)
        # this decreases in endgames but mustn't reach 0
        if board.turn == chess.WHITE:
            king_safety_score = weights["KING_SAFETY_WEIGHT"]* (0.3 + 0.7 * phase)


        else:
            king_safety_score =-weights["KING_SAFETY_WEIGHT"] * (0.3 + 0.7 * phase)

    
    # =============================================================================
    # MOBILITY EVALUATION
    # =============================================================================
    # new simplified mobility calculation 
    mobility_score = calculate_mobility(board) * weights['MOBILITY_WEIGHT'] * phase

    #bishop pair advantage calculation
    if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2:
        bishop_pair += weights['BISHOP_PAIR_BASE']
    if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2:
        bishop_pair -=weights['BISHOP_PAIR_BASE']


    # =============================================================================
    # KING TROPISM  EVALUATION
    # =============================================================================
    king_tropism+=king_trop(board,weights)*weights['KING_TROP_WEIGHT']
    
    pawn_eval=pawn_factor(board,weights)*weights['PAWN_FACTOR']
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
        bishop_pair+             #bishop pair advantage
        pawn_eval
    )
    
    return total_evaluation
# evaluation.py
def ordered_moves(board: 'chess.Board', killer_moves=None, depth=0, history=None) -> list:
    """
    Orders legal moves using heuristic scoring to optimize alpha-beta pruning.
    Includes killer move heuristic.
    """
    # Defensive defaults (avoid mutable default args)
    if killer_moves is None:
        killer_moves = {}
    if history is None:
        history = [[0] * 64 for _ in range(64)]

    # scoring constants
    PROMOTION_BONUS = 10_000
    KILLER_BONUS     = 9_000      # below promotion, above captures
    CAPTURE_BASE     = 1_000
    CASTLING_BONUS   = 500
    CENTER_BONUS     = 100

    CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}

    move_scores = []

    # retrieve killer moves for this depth (ensure list-like)
    killers = killer_moves.get(depth, [])
    if killers is None:
        killers = []

    for move in board.legal_moves:
        score = 0
        target_piece = board.piece_at(move.to_square)

        # PRIORITY 0 — KILLER MOVES (quiet only)
        if move in killers:
            if not target_piece and not move.promotion and not board.is_castling(move):
                score += KILLER_BONUS

        # PRIORITY 1 — PROMOTIONS
        if move.promotion:
            score += PROMOTION_BONUS

        # PRIORITY 2 — CAPTURES (MVV-LVA)
        elif target_piece:
            attacker_piece = board.piece_at(move.from_square)
            victim = PIECE_VALUES.get(target_piece.piece_type, 0)
            attacker = PIECE_VALUES.get(attacker_piece.piece_type, 1) if attacker_piece else 1
            # MVV-LVA style: prefer capturing high value victims with low-value attackers
            score += CAPTURE_BASE * victim - attacker

        # PRIORITY 3 — CASTLING
        elif board.is_castling(move):
            score += CASTLING_BONUS

        # PRIORITY 4 — CENTER CONTROL
        elif move.to_square in CENTER_SQUARES:
            score += CENTER_BONUS

        # Priority 5 - history heuristic
        else:
            score += history[move.from_square][move.to_square]

        move_scores.append((score, move))

    # sort best first
    move_scores.sort(reverse=True, key=lambda x: x[0])
    return [m for s, m in move_scores]









def auto_tune(phase: str, param: str, tests: dict, weight_range=None):

    phase_maps = {
        "opening": opening_values,
        "middlegame": middlegame_values,
        "endgame": endgame_values
    }

    if phase not in phase_maps:
        raise ValueError("Invalid phase. Choose: opening/middlegame/endgame")

    WEIGHTS = phase_maps[phase]

    if param not in WEIGHTS:
        raise ValueError(f"Invalid param {param}. Available: {list(WEIGHTS.keys())}")

    # -----------------------------
    # RANGE OF VALUES TO TEST
    # -----------------------------
    if weight_range is None:
        # automatic reasonable tuning range
        base = WEIGHTS[param]
        weight_range = [base * 0.5, base * 0.75, base, base * 1.25, base * 1.5]

    print("\n=========================================")
    print(f"   AUTO TUNING PARAMETER: {param}")
    print(f"   PHASE: {phase}")
    print("=========================================\n")

    # -----------------------------
    # ENGINE SETUP
    # -----------------------------
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 1, "Hash": 32})

    results = {}

    for value in weight_range:
        print(f"\n------ TESTING {param} = {value} ------\n")

        WEIGHTS[param] = value
        diffs = []

        for name, fen in tests.items():
            board = chess.Board(fen)

            if not board.is_valid():
                # print(f"[SKIP] {name} -> invalid FEN.")
                continue

            my_eval = evaluate(board)  # your eval

            info = engine.analyse(board, chess.engine.Limit(depth=2))
            sf_eval = info["score"].white().score(mate_score=WEIGHTS["MATE_VALUE"])

            diff = abs(my_eval - sf_eval)
            diffs.append(diff)

            # print(f"{name}:  diff = {diff}")
            # print(f"  Yours = {my_eval}")
            # print(f"  SF    = {sf_eval}")
            # print("")

            engine.ping()

        avg = sum(diffs) / len(diffs)
        results[value] = avg

        print(f"==> AVERAGE DIFF FOR {param}={value}: {avg}\n")

    engine.quit()

    # -----------------------------
    # BEST VALUE
    # -----------------------------
    best_value = min(results, key=results.get)

    print("\n=========================================")
    print(f"BEST VALUE FOR {param}: {best_value}")
    print(f"AVERAGE DIFF: {results[best_value]}")
    print("=========================================\n")

    return best_value, results



if __name__ == "__main__":
    tests = {
    # --- Opening Positions ---
  "Start Position": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",

    "Ruy Lopez": "rnbqkbnr/pppp1ppp/8/4p3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    "Italian Game": "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2",
    "Scotch Game": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Sicilian Defense": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "French Defense": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "Caro-Kann Defense": "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Pirc Defense": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "King's Indian Defense": "rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2",
    "English Opening": "rnbqkbnr/pppppppp/8/8/4P3/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",

    "Vienna Game": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "Four Knights Game": "r1bqkbnr/pppp1ppp/2n5/4p3/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 3",
    "Petrov Defense": "rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 3",
    "Philidor Defense": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Alekhine Defense": "rnbqkbnr/pppppppp/8/8/4P3/5n2/PPPP1PPP/RNBQKBNR w KQkq - 2 2",
    "Modern Defense": "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",

    "Queen's Gambit": "rnbqkbnr/ppp1pppp/8/3p4/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Slav Defense": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "Semi-Slav Defense": "rnbqkb1r/ppp1pppp/4pn2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",
    "King's Gambit": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
    "Queen's Indian Defense": "rnbqkb1r/pppp1ppp/5n2/4p3/2BP4/8/PPP1PPPP/RNBQK1NR b KQkq - 2 3",
    "Nimzo-Indian": "rnbqkb1r/pppp1ppp/4pn2/8/2BP4/5N2/PPP1PPPP/RNBQK2R b KQkq - 2 3",
    "Bogo-Indian": "rnbqkb1r/pppp1ppp/5n2/8/3P4/4PN2/PPP2PPP/R1BQKB1R b KQkq - 1 3",
    "Benoni Defense": "rnbqkbnr/ppp1pppp/8/3p4/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 2",
    "Benko Gambit": "rnbqkbnr/ppp1pppp/8/q2p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3",

    "Dutch Defense": "rnbqkbnr/pppppppp/8/8/3P1p2/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "Grunfeld Defense": "rnbqkb1r/ppp1pppp/5n2/3p4/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 3",
    "Catalan Opening": "rnbqkb1r/ppp2ppp/4pn2/3p4/2BPP3/5N2/PPP2PPP/RNBQK2R b KQkq - 1 3",
    "Trompowsky Attack": "rnbqkbnr/pppppppp/8/8/3P3B/8/PPP1PPPP/RNBQK1NR b KQkq - 1 1",
    "London System": "rnbqkbnr/pp1ppppp/2p5/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 2 2",
    "Colle System": "rnbqkbnr/pppppppp/8/8/3P4/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 1 2",
    "Tarrasch Defense": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "Budapest Gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 2",

    "Scandinavian Defense": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2",
    "Owen Defense": "rnbqkbnr/pppppppp/8/8/3p1b2/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "Horwitz Defense": "rnbqkbnr/pppp1ppp/8/4p3/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 2",

    "Reti Opening": "rnbqkbnr/pppppppp/8/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 1",
    "King's Fianchetto": "rnbqkbnr/pppppppp/8/8/4P3/6NP/PPPP1PP1/RNBQKB1R b KQkq - 0 2",
    "Bird Opening": "rnbqkbnr/pppppppp/8/8/3P1p2/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "Larsen Opening": "rnbqkbnr/pppppppp/8/8/2B5/8/PPPPPPPP/RNBQK1NR b KQkq - 1 1",

    "Evans Gambit": "rnbqkbnr/pppp1ppp/8/4p3/2BPP3/8/PPP2PPP/RNBQK1NR b KQkq - 0 3",
    "Fried Liver Attack": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/3NP3/8/PPP2PPP/R1BQK2R b KQkq - 6 5",
    "Max Lange Attack": "r1bqkb1r/pppp1ppp/2n5/1B2p3/3NP3/8/PPP2PPP/R1BQK2R b KQkq - 4 5",
    "Two Knights Defense": "r1bqkb1r/pppp1ppp/2n5/4p3/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 2 3",
    "Giuoco Pianissimo": "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 2",

    "Scotch Gambit": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 1 2",
    "Danish Gambit": "rnbqkbnr/ppp2ppp/3p4/4p3/2PP4/8/PP3PPP/RNBQKBNR b KQkq - 0 3",
    "Smith-Morra Gambit": "rnbqkbnr/pp1ppppp/8/2p5/4P3/2P5/PP1P1PPP/RNBQKBNR b KQkq - 0 2",
    "Wing Gambit (Sicilian)": "rnbqkbnr/pp1ppppp/8/2p5/2P1P3/8/PP1P1PPP/RNBQKBNR b KQkq - 0 2",

    # --- Midgame Situations ---
    # "White +Pawn": "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    # "Black +Rook": "rnbqkbnr/pppppppp/8/8/8/8/PPP5/RNBQKBNR w KQkq - 0 1",
    # "Center Locked": "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 2 3",
    # "Open Center": "rnbqkb1r/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3",
    # "Exposed Kings": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    # "Trapped Bishop": "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 2 2",
    # "Weak Pawns": "rnbqkb1r/pp1ppppp/2p5/8/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    # "Passed Pawn": "8/8/3k4/8/3P4/8/8/3K4 w - - 0 1",
    # "Isolated Pawn": "rnbqkb1r/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 4",
    # "Doubled Pawns": "rnbqkbnr/ppp2ppp/8/3pp3/8/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 4",
    # "Backward Pawn": "rnbqkbnr/ppp2ppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3",
    # "Pawn Chain": "rnbqkbnr/ppp1pppp/8/3p4/3P4/2P5/PP2PPPP/RNBQKBNR b KQkq - 0 3",
    
    # # --- Tactical Patterns ---
    # "Mate Threat": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 3",
    # "King Attack": "rnb1kbnr/pppp1ppp/8/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R b KQkq - 4 4",
    # "Pinned Knight": "rnbqkbnr/pppp1ppp/8/4p3/2B5/5N2/PPPPPPPP/RNBQK2R b KQkq - 3 3",
    # "Fork Threat": "rnbqkbnr/pppp1ppp/8/4p3/2B5/8/PPPPPPPP/RNBQK1NR w KQkq - 2 3",
    # "Skewer Threat": "r3k2r/pppqppbp/2np1np1/8/2BPP3/2N2N2/PPP2PPP/R1BQ1RK1 w kq - 0 7",
    # "Discovered Attack": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 2 4",
    # "Hanging Piece": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPPBPPP/R1BQK2R w KQkq - 2 4",
    # "Back Rank Weakness": "r3k2r/ppp2ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1",
    # "Overextended Pawns": "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    # "King Open File": "rnbqkbnr/pppp1ppp/8/4p3/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 2",
    # "Double Attack": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/2N5/PPPP1PPP/R1BQK2R w KQkq - 2 4",
    # "Discovered Check": "r1bqk1nr/pppp1ppp/2n5/4p3/2B1P3/2N2Q2/PPPP1PPP/R1B1K2R b KQkq - 3 4",
    # "Skewer + Pin": "r3k2r/pppqppbp/2np1np1/8/2BPP3/2N2Q2/PPP2PPP/R1B2RK1 w kq - 0 7",
    
#     # --- Endgame Patterns ---
#     "King Opposition": "8/8/8/3k4/3K4/8/8/8 w - - 0 1",
#     "Lucena Position": "8/8/8/8/8/2K5/5P2/6k1 w - - 0 1",
#     "Philidor Position": "8/8/8/8/1k6/8/2K5/8 w - - 0 1",
#     "Rook Behind Passed Pawn": "8/8/8/8/4P3/8/3K4/5k2 w - - 0 1",
#     "Opposite Colored Bishops": "8/8/8/8/3b4/4B3/3K4/5k2 w - - 0 1",
#     "King and Pawn": "8/8/3k4/8/3P4/8/8/3K4 w - - 0 1",
#     "Rook Endgame": "8/8/3k4/8/3R4/8/8/3K4 w - - 0 1",
#     "Knight Endgame": "8/8/3k4/8/3N4/8/8/3K4 w - - 0 1",
#     "Bishop Endgame": "8/8/3k4/8/3B4/8/8/3K4 w - - 0 1",
#     "Opposite Bishops": "8/8/3k4/8/3B4/8/8/3K2b1 w - - 0 1",
#     "Rook vs Bishop": "8/8/3k4/8/3R4/8/8/3K2b1 w - - 0 1",
#     "Knight vs Bishop": "8/8/3k4/8/3N4/8/8/3K2b1 w - - 0 1",
#     "Pawn Race": "8/3k4/8/3P4/8/8/4p3/3K4 w - - 0 1",
#     "Two Queens": "8/8/8/3k4/3Q4/8/3Q4/3K4 w - - 0 1",
#     "Underpromotion": "8/3k4/8/8/3P4/8/8/3K4 w - - 0 1",

#     # --- Imbalanced Positions ---
#     "Queen Sacrifice": "rnb1kbnr/ppppqppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 3",
#     "Material Imbalance": "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPP2PPP/RNBQKBNR w KQkq - 0 2",
#     "Two Bishops Advantage": "rnbqkbnr/pppppppp/8/8/8/8/PPP2PPP/RNBQKBBR w KQkq - 0 1",
#     "Rook Lift": "rnbqkbnr/pppppppp/8/8/3R4/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
#     "Center Break": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
#     "Advanced Pawn Chain": "rnbqkbnr/pppppppp/8/8/3PPP2/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
#     "King Walk": "8/8/3k4/8/4K3/8/8/8 w - - 0 1",
#     "Knight Outpost": "rnbqkbnr/pppppppp/8/8/3N4/8/PPP2PPP/R1BQKBNR b KQkq - 0 3",
#     "Pawn Storm": "rnbq1rk1/ppppppbp/6p1/8/3PP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 6",
}
    
    for factor in middlegame_values:

        print(auto_tune("middlegame", factor, tests))

