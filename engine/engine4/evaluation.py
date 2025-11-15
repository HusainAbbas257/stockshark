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
    
    KING_SAFETY_WEIGHT = 10    # Penalty for being in check (in centipawns)
    MOBILITY_WEIGHT = 0.1      # Weight per legal move available
    MATE_VALUE = 999999     #constant for expressing mate value
    
    # =============================================================================
    # SCORE ACCUMULATORS - Track different aspects of position
    # =============================================================================
    
    material_score = 0      # Total material difference (White - Black)
    positional_score = 0    # Positional bonuses from piece placement
    king_safety_score = 0   # King safety penalties
    mobility_score = 0      # Advantage from having more moves available
    
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
        for square in white_pieces:
            positional_score += piece_square_table[piece_type][square]
        
        # Evaluate Black's piece placement
        # Mirror the board (flip vertically) since PST is designed for White's perspective
        for square in black_pieces:
            mirrored_square = chess.square_mirror(square)
            positional_score -= piece_square_table[piece_type][mirrored_square]
    
    # =============================================================================
    # KING SAFETY EVALUATION
    # =============================================================================
    
    # Being in check is dangerous and should be penalized
    if board.is_check():
        # If White is in check, subtract penalty (bad for White)
        # If Black is in check, add bonus (good for White)
        if board.turn == chess.WHITE:
            king_safety_score = -KING_SAFETY_WEIGHT
        else:
            king_safety_score = KING_SAFETY_WEIGHT
    
    # =============================================================================
    # MOBILITY EVALUATION
    # =============================================================================
    
    # More legal moves = more tactical options and flexibility
    # Count legal moves for the side to move
    white_moves = len(list(board.legal_moves))
    board.push(chess.Move.null())
    black_moves = len(list(board.legal_moves))
    board.pop()

    mobility_score = (white_moves - black_moves) * MOBILITY_WEIGHT

    # =============================================================================
    # COMBINE ALL FACTORS
    # =============================================================================
    
    # Sum all evaluation components
    # Material is the dominant factor, others provide fine-tuning
    total_evaluation = (
        material_score +      # Dominant factor: piece count and value
        positional_score +    # Piece placement quality
        king_safety_score +   # King safety considerations
        mobility_score        # Tactical flexibility
    )
    
    return total_evaluation
def ordered_moves(board: 'chess.Board', killer_moves={}, depth=0) -> list:
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
    b=chess.Board()
    for name in tests:
        b.set_fen(tests[name])
        print(f'{name}:{tests[name]}-->{evaluate(b)}')
