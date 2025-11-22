import chess
from evaluation import evaluate as evaluate_board
from evaluation import ordered_moves, PIECE_VALUES, piece_square_table
import random
import json
from collections import defaultdict
# import os

# FIXED: Use relative path instead of hardcoded absolute 
# FIX 2: softcoded code is not working properly so for now we currently use hardcoded
# opening_file_path = os.path.join(os.path.dirname(__file__), 'data', 'opening.json')
opening_file_path=r'C:\Users\dell\Desktop\stockshark\data\opening.json'

# Zobrist Table (Global)
ZOBRIST_PIECES = {}
ZOBRIST_CASTLING = {}
ZOBRIST_ENPASSANT = {}
ZOBRIST_SIDE = random.getrandbits(64)

class Engine:
    def __init__(self):
        self.tt = {}
        self.init_zobrist()
        self.killer_moves = defaultdict(list)
        self.history = [[0] * 64 for _ in range(64)]
        self.history_max = 10000  # Cap for history values
        
        # incremental eval caches
        self.material_score = 0   # White - Black material
        self.pst_score = 0        # White - Black PST total
        self._eval_stack = []     # stack of (delta_mat, delta_pst) for undoing
        
        try:
            with open(opening_file_path, "r") as f:
                self.opening = json.load(f)
        except FileNotFoundError:
            self.opening = {}
            
    def clear_tt(self):
        """Clear transposition table to free memory."""
        self.tt.clear()
    def init_zobrist(self):
        """Initialize Zobrist hashing tables with deterministic random values."""
        global ZOBRIST_PIECES, ZOBRIST_CASTLING, ZOBRIST_ENPASSANT, ZOBRIST_SIDE
        
        # Seed fixed for consistency across runs
        random.seed(1337)
        
        # Clear existing tables first
        ZOBRIST_PIECES.clear()
        ZOBRIST_CASTLING.clear()
        ZOBRIST_ENPASSANT.clear()
        
        # Generate piece-square zobrist keys
        for piece_type in range(1, 7):
            for color in [chess.WHITE, chess.BLACK]:
                for square in range(64):
                    ZOBRIST_PIECES[(piece_type, color, square)] = random.getrandbits(64)
        
        # Generate castling rights zobrist keys
        for right in ["K", "Q", "k", "q"]:
            ZOBRIST_CASTLING[right] = random.getrandbits(64)
        
        # Generate en passant file zobrist keys
        for file in range(8):
            ZOBRIST_ENPASSANT[file] = random.getrandbits(64)
        
        # Generate side to move zobrist key
        ZOBRIST_SIDE = random.getrandbits(64) 

    def compute_hash(self, board):
        """Compute Zobrist hash - safe for en-passant and uses globals."""
        global ZOBRIST_PIECES, ZOBRIST_CASTLING, ZOBRIST_ENPASSANT, ZOBRIST_SIDE

        h = 0
        # piece-square
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                key = (piece.piece_type, piece.color, square)
                # safe fallback if table missing (shouldn't happen)
                if key in ZOBRIST_PIECES:
                    h ^= ZOBRIST_PIECES[key]

        # side to move
        if board.turn == chess.BLACK:
            h ^= ZOBRIST_SIDE

        # castling rights (use boolean checks)
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= ZOBRIST_CASTLING.get("K", 0)
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= ZOBRIST_CASTLING.get("Q", 0)
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= ZOBRIST_CASTLING.get("k", 0)
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= ZOBRIST_CASTLING.get("q", 0)

        # en-passant: XOR file key if ep_square exists (standard practice).
        ep = board.ep_square
        if ep is not None:
            file_idx = chess.square_file(ep)
            # only XOR if file key exists
            h ^= ZOBRIST_ENPASSANT.get(file_idx, 0)

        return h


    # --- Transposition table helpers ---#
    def tt_lookup(self, key: int, depth: int, alpha: float, beta: float):
        """
        Return a usable value from TT or None.
        Entry format: (value, stored_depth, flag)
        Flags: "EXACT", "LOWERBOUND", "UPPERBOUND"
        """
        entry = self.tt.get(key)
        if entry is None:
            return None

        value, stored_depth, flag = entry
        # only use if stored depth >= required depth
        if stored_depth < depth:
            return None

        if flag == "EXACT":
            return value
        if flag == "LOWERBOUND" and value >= beta:
            return value
        if flag == "UPPERBOUND" and value <= alpha:
            return value

        return None

    def tt_store(self, key: int, depth: int, value: float, flag: str):
        """
        Store into TT. Overwrite only if new depth >= stored depth (simple policy).
        """
        entry = self.tt.get(key)
        if entry is None or depth >= entry[1]:
            self.tt[key] = (value, depth, flag)

    def is_good_capture(self, board, move):
        piece_from = board.piece_at(move.from_square)
        piece_to = board.piece_at(move.to_square)

        if not piece_from or not piece_to:
            return False

        # Only search captures where you take equal or higher value
        return piece_to.piece_type >= piece_from.piece_type

    # ------------------------------
    # Incremental evaluation helpers
    # ------------------------------
    def evaluate_fast(self):
        """Ultra-fast eval using cached incremental values - NO BOARD SCAN"""
        return self.material_score + (self.pst_score * 0.028125)
    def init_eval_cache(self, board: 'chess.Board'):
        """Compute initial material_score and pst_score from board."""
        mat = 0
        pst = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p:
                continue
            val = PIECE_VALUES.get(p.piece_type, 0)
            pst_val = piece_square_table.get(p.piece_type, (0,)*64)[sq]
            if p.color == chess.WHITE:
                mat += val
                pst += pst_val
            else:
                mat -= val
                # BLACK PIECES: USE MIRRORED PST (FIXED)
                mirrored = chess.square_mirror(sq)
                pst -= piece_square_table.get(p.piece_type, (0,)*64)[mirrored]
        self.material_score = mat
        self.pst_score = pst
        self._eval_stack = []
    def push_move_with_eval(self, board: 'chess.Board', move: 'chess.Move'):
        """
        Replace board.push(move) with this.
        Compute deltas and push move while keeping eval caches consistent.
        """
        moving = board.piece_at(move.from_square)
        # Normal captured piece at to_square (may be None)
        captured = board.piece_at(move.to_square)
        promo = move.promotion  # None or piece_type int

        delta_mat = 0
        delta_pst = 0

        # Moving piece PST/material deltas
        if moving:
            from_pst = piece_square_table.get(moving.piece_type, (0,)*64)[move.from_square]
            dest_piece_type = promo if promo else moving.piece_type
            to_pst = piece_square_table.get(dest_piece_type, (0,)*64)[move.to_square]

            if moving.color == chess.WHITE:
                # remove from from_square
                delta_pst -= from_pst
                # add at to_square (promotion accounted)
                delta_pst += to_pst
                if promo:
                    # white promoted -> gained material = promo_val - pawn_val (or old piece val)
                    old_val = PIECE_VALUES.get(moving.piece_type, 0)
                    new_val = PIECE_VALUES.get(promo, old_val)
                    delta_mat += (new_val - old_val)
                # otherwise material unchanged for white move (unless capture handled below)
            else:
                # BLACK PIECES: USE MIRRORED PST (FIXED)
                from_mirror = chess.square_mirror(move.from_square)
                to_mirror = chess.square_mirror(move.to_square)
                from_pst_mirror = piece_square_table.get(moving.piece_type, (0,)*64)[from_mirror]
                to_pst_mirror = piece_square_table.get(dest_piece_type, (0,)*64)[to_mirror]
                
                # black moves decrease white advantage when piece leaves from_square
                delta_pst += from_pst_mirror
                delta_pst -= to_pst_mirror
                if promo:
                    old_val = PIECE_VALUES.get(moving.piece_type, 0)
                    new_val = PIECE_VALUES.get(promo, old_val)
                    delta_mat -= (new_val - old_val)

        # Handle capture (including en-passant)
        if board.is_en_passant(move):
            # captured pawn is not on to_square
            ep_sq = move.to_square + (-8 if moving and moving.color == chess.WHITE else 8)
            cap = board.piece_at(ep_sq)
            if cap:
                cap_val = PIECE_VALUES.get(cap.piece_type, 0)
                if cap.color == chess.WHITE:
                    delta_mat -= cap_val
                    cap_pst = piece_square_table.get(cap.piece_type, (0,)*64)[ep_sq]
                    delta_pst -= cap_pst
                else:
                    delta_mat += cap_val
                    # BLACK CAPTURED: USE MIRRORED PST
                    ep_mirror = chess.square_mirror(ep_sq)
                    cap_pst = piece_square_table.get(cap.piece_type, (0,)*64)[ep_mirror]
                    delta_pst += cap_pst
        else:
            if captured:
                cap_val = PIECE_VALUES.get(captured.piece_type, 0)
                if captured.color == chess.WHITE:
                    delta_mat -= cap_val
                    cap_pst = piece_square_table.get(captured.piece_type, (0,)*64)[move.to_square]
                    delta_pst -= cap_pst
                else:
                    delta_mat += cap_val
                    # BLACK CAPTURED: USE MIRRORED PST
                    to_mirror = chess.square_mirror(move.to_square)
                    cap_pst = piece_square_table.get(captured.piece_type, (0,)*64)[to_mirror]
                    delta_pst += cap_pst

        # Handle castling: rook also moves; include rook PST delta
        if board.is_castling(move):
            # Determine rook from/to squares depending on color & side
            if moving and moving.color == chess.WHITE:
                # white king: e1->g1 (kingside) rook h1->f1 ; e1->c1 (queenside) rook a1->d1
                if move.to_square == chess.G1:
                    rook_from, rook_to = chess.H1, chess.F1
                elif move.to_square == chess.C1:
                    rook_from, rook_to = chess.A1, chess.D1
                else:
                    rook_from, rook_to = None, None
            elif moving and moving.color == chess.BLACK:
                if move.to_square == chess.G8:
                    rook_from, rook_to = chess.H8, chess.F8
                elif move.to_square == chess.C8:
                    rook_from, rook_to = chess.A8, chess.D8
                else:
                    rook_from, rook_to = None, None
            else:
                rook_from, rook_to = None, None

            if rook_from is not None:
                rook_piece = board.piece_at(rook_from)
                if rook_piece:
                    # rook PST change
                    rf_pst = piece_square_table.get(rook_piece.piece_type, (0,)*64)[rook_from]
                    rt_pst = piece_square_table.get(rook_piece.piece_type, (0,)*64)[rook_to]
                    if rook_piece.color == chess.WHITE:
                        delta_pst -= rf_pst
                        delta_pst += rt_pst
                    else:
                        # BLACK ROOK: USE MIRRORED PST
                        rf_mirror = chess.square_mirror(rook_from)
                        rt_mirror = chess.square_mirror(rook_to)
                        rf_pst_m = piece_square_table.get(rook_piece.piece_type, (0,)*64)[rf_mirror]
                        rt_pst_m = piece_square_table.get(rook_piece.piece_type, (0,)*64)[rt_mirror]
                        delta_pst += rf_pst_m
                        delta_pst -= rt_pst_m

        # Apply deltas to caches
        self.material_score += delta_mat
        self.pst_score += delta_pst

        # push undo record
        self._eval_stack.append((delta_mat, delta_pst))
        # finally push the move
        board.push(move)

    def pop_move_with_eval(self, board: 'chess.Board'):
        """
        Replace board.pop() with this. Pops board and reverts incremental cache using stack.
        """
        # pop board first to restore position
        board.pop()
        if not self._eval_stack:
            # safety fallback: recompute whole cache (slow)
            self.init_eval_cache(board)
            return
        delta_mat, delta_pst = self._eval_stack.pop()
        # revert deltas
        self.material_score -= delta_mat
        self.pst_score -= delta_pst

    # ------------------------------
    # End incremental helpers
    # ------------------------------

    def quiescence(self, board, alpha, beta, is_maximizing, depth=0, max_q=4):
        """
        Quiescence search to avoid horizon effect by searching only tactical moves.
        Continues searching captures until a quiet position is reached.
        """
        if depth >= max_q:
            # USE CACHED EVAL INSTEAD OF FULL EVALUATION
            return self.material_score + (self.pst_score * 0.028125)
        
        # USE CACHED EVAL - MASSIVE SPEEDUP
        stand_pat = self.material_score + (self.pst_score * 0.028125)
        
        # Beta cutoff - position too good
        if is_maximizing:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat
        
        # Generate only capture moves efficiently
        capture_moves = [m for m in board.legal_moves if board.is_capture(m)]
        
        # Delta pruning: skip if even best capture can't improve position
        if is_maximizing:
            # If even capturing a queen won't help, skip
            if stand_pat + 1000 < alpha:
                return alpha
        else:
            if stand_pat - 1000 > beta:
                return beta
        
        for move in capture_moves:
            # use incremental push/pop
            self.push_move_with_eval(board, move)
            score = self.quiescence(board, alpha, beta, not is_maximizing, depth + 1, max_q)
            self.pop_move_with_eval(board)
            
            if is_maximizing:
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            else:
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
        
        return alpha if is_maximizing else beta
        
    def minimax(self, board: 'chess.Board', depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf'),depth_count=1) -> float:
        """
        Minimax algorithm with alpha-beta pruning for chess position evaluation.
        
        Recursively evaluates positions by simulating optimal play from both sides.
        The algorithm alternates between maximizing (White's perspective) and minimizing 
        (Black's perspective) at each depth level.
        
        Args:
            board: Current chess board state
            depth: Remaining search depth (decrements each recursion)
            is_maximizing: True if maximizing player's turn (White), False if minimizing (Black)
            alpha: Best score maximizer can guarantee (for pruning)
            beta: Best score minimizer can guarantee (for pruning)
        
        Returns:
            float: Evaluation score of the position
                - Positive values favor White
                - Negative values favor Black
                - Magnitude indicates strength of advantage
        """
        
        # =============================================================================
        # TERMINAL POSITION CHECKS (Base cases)
        # =============================================================================
        
        # Checkmate: Game over, return depth-adjusted score
        # - Faster checkmates are valued higher (for winning) or lower (for losing)
        # - Adding depth for loser, subtracting for winner ensures proper ordering
        MATE_SCORE =  32000 
        if board.is_checkmate():
            # side to move is checkmated = bad for them
            return -MATE_SCORE + depth if board.turn == chess.WHITE else MATE_SCORE - depth

        # Explicit stalemate check (draw condition)
        if board.is_stalemate():
            return 0.0
        
        # Draw scenarios - always return neutral evaluation
        if (board.is_fivefold_repetition() or 
            board.is_insufficient_material() or 
            board.is_repetition()):
            return 0.0
        
        # =============================================================================
        # DEPTH LIMIT - Leaf node evaluation
        # =============================================================================
        
        # When depth reaches 0, evaluate the position statically
        # This is the base case for the recursion
        if depth == 0:
            return self.quiescence(board, alpha, beta, is_maximizing)


        
        
        
        # ==================================================================
        # TRANSPOSITION TABLE: checking for already scanned transposition
        # ==================================================================
        key = self.compute_hash(board)
        tt_val = self.tt_lookup(key, depth, alpha, beta)
        if tt_val is not None:
            return tt_val

        
        
        
        # =============================================================================
        # MOVE GENERATION - Order moves for better pruning
        # =============================================================================
        
        # Get moves sorted by heuristic priority (captures, checks, etc. first)
        # Better move ordering = more alpha-beta cutoffs = faster search
        moves = ordered_moves(board,self.killer_moves,depth,self.history)
        
        # if there is no legal move then its a draw buddy:
        if not moves:
            return 0.0
        
        # =============================================================================
        # MAXIMIZING PLAYER - White's turn (or whoever is maximizing)
        # =============================================================================
        if is_maximizing:
            alpha_orig = alpha
            max_eval = float('-inf')

            # In minimax(), maximizing player section:
            for idx, move in enumerate(moves):
                # Only evaluate expensive checks if we're considering LMR
                is_capture = board.is_capture(move)
                is_killer = move in self.killer_moves.get(depth, [])
                
                # Check if this move qualifies for LMR
                can_reduce = (
                    depth >= 3
                    and idx >= 4  # Late move (after first 4)
                    and not is_capture
                    and not is_killer
                )
                
                # Only call gives_check if we're reducing (expensive call)
                if can_reduce:
                    gives_check = board.gives_check(move)
                    can_reduce = not gives_check
                
                # use incremental push/pop
                self.push_move_with_eval(board, move)
                
                if can_reduce:
                    # Reduced depth search
                    eval_score = self.minimax(board, depth - 2, False, alpha, beta, depth_count + 1)
                    
                    # Re-search if promising
                    if eval_score > alpha:
                        eval_score = self.minimax(board, depth - 1, False, alpha, beta, depth_count + 1)
                else:
                    # Normal full-depth search
                    eval_score = self.minimax(board, depth - 1, False, alpha, beta, depth_count + 1)
                
                self.pop_move_with_eval(board)


                if eval_score > max_eval:
                    max_eval = eval_score
                if eval_score > alpha:
                    alpha = eval_score

                if alpha >= beta:
                    # storing this move as a good move:
                    self.history[move.from_square][move.to_square] += depth_count * depth_count
                    # Cap the value to prevent overflow
                    if self.history[move.from_square][move.to_square] > self.history_max:
                        # Age all history values when one exceeds max
                        for i in range(64):
                            for j in range(64):
                                self.history[i][j] //= 2
                    # storing killer moves 
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []

                    # Store killer moves (max 2 per depth)
                    killers = self.killer_moves[depth]

                    # Check if move already in killer list
                    if len(killers) == 0:
                        killers.append(move)
                    elif len(killers) == 1:
                        if killers[0] != move:
                            killers.insert(0, move)
                    else:  # len == 2
                        if killers[0] != move and killers[1] != move:
                            killers.pop()  # Remove oldest
                            killers.insert(0, move)  # Add newest at front

                    # Beta cutoff at maximizing node -> LOWERBOUND
                    self.tt_store(key, depth, alpha, "LOWERBOUND")
                    return alpha

            # no cutoff: set flag based on original window 
            if max_eval <= alpha_orig:
                flag = "UPPERBOUND"
            elif max_eval >= beta:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"

            self.tt_store(key, depth, max_eval, flag)
            return max_eval
        # =============================================================================
        # MINIMIZING PLAYER - Black's turn (or whoever is minimizing)
        # =============================================================================
        else:
            beta_orig = beta
            alpha_orig = alpha
            min_eval = float('inf')

            for idx, move in enumerate(moves):
                # Evaluate flags BEFORE pushing
                is_capture = board.is_capture(move)
                gives_check = board.gives_check(move)
                is_killer = move in self.killer_moves.get(depth, [])

                # use incremental push/pop
                self.push_move_with_eval(board, move)

                # --- LMR CONDITIONS ---
                # For minimizing side: only reduce quiet/non-check/non-killer late moves
                if (
                    depth >= 3
                    and idx >= 3
                    and not is_capture
                    and not gives_check
                    and not is_killer
                ):
                    eval_score = self.minimax(board, depth-2, True, alpha, beta, depth_count+1)

                    # re-search condition for minimizing side: if reduced search gave a value that improves beta, re-search full depth
                    if eval_score < beta:
                        eval_score = self.minimax(board, depth-1, True, alpha, beta, depth_count+1)
                else:
                    eval_score = self.minimax(board, depth-1, True, alpha, beta, depth_count+1)


                self.pop_move_with_eval(board)


                if eval_score < min_eval:
                    min_eval = eval_score
                if eval_score < beta:
                    beta = eval_score

                if beta <= alpha:
                    # history update on cutoff
                    self.history[move.from_square][move.to_square] += depth_count * depth_count
                    # Cap the value to prevent overflow
                    if self.history[move.from_square][move.to_square] > self.history_max:
                        # Age all history values when one exceeds max
                        for i in range(64):
                            for j in range(64):
                                self.history[i][j] //= 2

                    # killer moves
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []

                    # Store killer moves (max 2 per depth)
                    killers = self.killer_moves[depth]

                    # Check if move already in killer list
                    if len(killers) == 0:
                        killers.append(move)
                    elif len(killers) == 1:
                        if killers[0] != move:
                            killers.insert(0, move)
                    else:  # len == 2
                        if killers[0] != move and killers[1] != move:
                            killers.pop()  # Remove oldest
                            killers.insert(0, move)  # Add newest at front

                    # Alpha cutoff at minimizing node -> UPPERBOUND
                    self.tt_store(key, depth, beta, "UPPERBOUND")
                    return beta


            # set flag based on original window (use original alpha/beta)
            if min_eval <= alpha_orig:
                flag = "UPPERBOUND"
            elif min_eval >= beta_orig:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"

            self.tt_store(key, depth, min_eval, flag)
            return min_eval

    def best_move(self, board: 'chess.Board', depth: int) -> tuple:
        """
        Determines the optimal move for the current player using minimax with alpha-beta pruning.
        
        This is the entry point for the chess engine's move selection. It evaluates all legal
        moves at the root of the search tree and returns the move with the best evaluation score
        according to the minimax algorithm.
        
        The function iterates through all legal moves (ordered by heuristic priority), recursively
        evaluates each resulting position to the specified depth, and selects the move that leads
        to the best outcome assuming both players play optimally.
        
        Args:
            board: Current chess board state to analyze
            depth: Search depth in plies (half-moves)
                - Higher depth = stronger play but exponentially slower
                - Typical values: 3-6 for reasonable performance
                - Each additional ply roughly multiplies search time by ~35 (branching factor)
        
        Returns:
            tuple: (best_move, evaluation_score)
                - best_move (chess.Move): The optimal move to play, or None if no legal moves
                - evaluation_score (float): The evaluation of the position after making best_move
                    * Positive: Favorable for White
                    * Negative: Favorable for Black
                    * Magnitude indicates strength of advantage (in centipawns typically)
        
        Example:
            >>> engine = ChessEngine()
            >>> board = chess.Board()
            >>> move, score = engine.best_move(board, depth=4)
            >>> print(f"Best move: {move}, Evaluation: {score}")
            Best move: e2e4, Evaluation: 25.0
        
        Note:
            - If the position is already terminal (checkmate/stalemate), returns (None, evaluation)
            - Uses ordered_moves() to improve alpha-beta pruning efficiency
            - The root node always uses full alpha-beta window (-inf, +inf)
        """
        # dont forget to clear the transopstion table or code may take too much memortry:
        if len(self.tt) > 1000000:  # 1M entries
            self.tt.clear()
        
        
        fen = board.fen()
        moves_list = self.opening.get(fen)
        if moves_list is not None:
            print('picking from from opening book...')
            import random as rd
            rd.seed()
            uci = rd.choice(moves_list)# pick random from list
            move = chess.Move.from_uci(uci)  # convert to move object
            return (move, 0)
        
        # INITIALIZE EVAL CACHE HERE - CRITICAL FIX
        self.init_eval_cache(board)
        
        # resett killers and history:
        self.killer_moves = defaultdict(list)
        self.history = [[0]*64 for _ in range(64)]

        # =============================================================================
        # DETERMINE PERSPECTIVE - Which player is making the move
        # =============================================================================
        
        # Store which color is currently moving (chess.WHITE or chess.BLACK)
        # At root level, we maximize for White, minimize for Black
        is_maximizing = board.turn == chess.WHITE
        
        # =============================================================================
        # INITIALIZE BEST MOVE TRACKING
        # =============================================================================
        
        # Initialize best evaluation found so far
        # Start with worst possible value for the current player:
        #   - If maximizing (White): start at -infinity (any move is better)
        #   - If minimizing (Black): start at +infinity (any move is better)
        best_evaluation = float('-inf') if is_maximizing else float('inf')
        
        # Track the move that produces the best evaluation
        best_move_found = None
        
        # =============================================================================
        # MOVE ORDERING - Get moves sorted by heuristic priority
        # =============================================================================
        
        # ordered_moves() returns legal moves sorted by likely strength
        # This improves alpha-beta pruning efficiency at the root level
        # Good moves (captures, promotions) are examined first
        moves = ordered_moves(board, self.killer_moves, depth, self.history)
        
        # =============================================================================
        # EDGE CASE - No legal moves available
        # =============================================================================
        
        # If there are no legal moves, the game is over (checkmate or stalemate)
        # Return None for move and current evaluation
        if not moves:
            return None, self.evaluate_fast()
        
        # =============================================================================
        # ROOT MOVE ITERATION - Evaluate each legal move
        # =============================================================================
        
        # Alpha-beta window for root node
        # Always start with full window since we need to evaluate all moves at root
        alpha = float('-inf')  # Best score maximizer can guarantee
        beta = float('inf')    # Best score minimizer can guarantee
        
        # Track original alpha for TT flag determination
        alpha_orig = alpha
        beta_orig = beta
        
        for move in moves:
            # ---------------------------------------------------------------------
            # Make the move temporarily (use incremental push)
            # ---------------------------------------------------------------------
            self.push_move_with_eval(board, move)
            
            # ---------------------------------------------------------------------
            # Recursively evaluate the resulting position
            # ---------------------------------------------------------------------
            # After making the move, it's the opponent's turn
            # So we pass the OPPOSITE of is_maximizing to reflect the perspective switch
            eval_score = self.minimax(
                board, 
                depth - 1, 
                not is_maximizing,  # Switch perspective: opponent's turn now
                alpha, 
                beta
            )
            
            # ---------------------------------------------------------------------
            # Restore board state (use incremental pop)
            # ---------------------------------------------------------------------
            self.pop_move_with_eval(board)
            
            # ---------------------------------------------------------------------
            # Update best move if this is an improvement
            # ---------------------------------------------------------------------
            
            if is_maximizing:
                # MAXIMIZING PLAYER (White)
                # Looking for highest evaluation score
                if eval_score > best_evaluation:
                    best_evaluation = eval_score
                    best_move_found = move
                    
                    # Update alpha (best score we can guarantee)
                    alpha = max(alpha, eval_score)
            
            else:
                # MINIMIZING PLAYER (Black)
                # Looking for lowest evaluation score
                if eval_score < best_evaluation:
                    best_evaluation = eval_score
                    best_move_found = move
                    
                    # Update beta (best score we can guarantee)
                    beta = min(beta, eval_score)
            
            # Optional early exit at root level for alpha-beta pruning
            # (Less common at root but technically valid)
            if beta <= alpha:
                break
        
        # =============================================================================
        # Store root position in transposition table
        # =============================================================================
        key = self.compute_hash(board)
        
        # Determine TT flag based on final evaluation vs original window
        if is_maximizing:
            if best_evaluation <= alpha_orig:
                flag = "UPPERBOUND"
            elif best_evaluation >= beta_orig:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"
        else:
            if best_evaluation <= alpha_orig:
                flag = "UPPERBOUND"
            elif best_evaluation >= beta_orig:
                flag = "LOWERBOUND"
            else:
                flag = "EXACT"
        
        self.tt_store(key, depth, best_evaluation, flag)
        
        # =============================================================================
        # RETURN RESULTS
        # =============================================================================
        
        return best_move_found, best_evaluation
    def self_play(self, depth: int = 3, max_moves: int = 150):
        print("\n" + "="*50)
        print("       SELF-PLAY GAME STARTING")
        print("="*50)
        print(f"Search Depth: {depth} plies")
        print(f"Max Moves: {max_moves}")
        print("="*50 + "\n")

        board = chess.Board()
        move_history = []
        move_number = 1

        while not board.is_game_over() and move_number <= max_moves:
            current_player = "White" if board.turn == chess.WHITE else "Black"
            print(f"\n{'__'*25}")
            print(f"Move {move_number} -> {current_player} to move")
            print(f"{'__'*25}")

            best_move_found, evaluation = self.best_move(board, depth)

            if best_move_found is None:
                print("ERROR: No legal move found")
                break

            eval_display = f"{evaluation:+.1f}" if abs(evaluation) < 1000 else \
                        ("+inf" if evaluation > 0 else "-inf")

            print(f"Move: {best_move_found.uci():6s} \t Eval: {eval_display:>6s}")

            board.push(best_move_found)
            move_history.append(best_move_found.uci())
            move_number += 1

        print("\nGame Over!")
        print("Result:", board.result())
        print("Moves Played:", len(move_history))
        print("Move List:", move_history)


    def play_against_human(self, color=chess.WHITE, depth=3):
        print("\n=== HUMAN VS AI ===\n")
        board = chess.Board()

        while not board.is_game_over():
            print(board)

            # Human move
            if board.turn == color:
                while True:
                    user_move = input("Your move (UCI): ").strip()
                    try:
                        move = chess.Move.from_uci(user_move)
                        if move in board.legal_moves:
                            board.push(move)
                            break
                        else:
                            print("Illegal move.")
                    except:
                        print("Invalid format.")
            else:
                ai_move, eval_ = self.best_move(board, depth)
                print(f"AI plays: {ai_move.uci()}  Eval: {eval_}")
                board.push(ai_move)

        print("\nGame Over!")
        print("Result:", board.result())
        print("Final position:")
        print(board)

    def compare(self, depth1: int = 3, depth2: int = 3, max_moves: int = 150) -> dict:
        """
        Play engine against itself with different depths for White and Black.
        
        Useful for:
            - Testing if deeper search actually plays better
            - Evaluating strength difference between depths
            - Finding bugs that only appear at certain depths
        
        Args:
            depth1: Search depth for White (plies)
            depth2: Search depth for Black (plies)
            max_moves: Maximum half-moves before declaring draw
        
        Returns:
            dict: Game result with statistics
        """
        
        print("\n" + "="*60)
        print(f"  ENGINE COMPARISON: White(depth={depth1}) vs Black(depth={depth2})")
        print("="*60 + "\n")
        
        board = chess.Board()
        move_history = []
        move_number = 1
        
        while not board.is_game_over() and move_number <= max_moves:
            
            # Determine current depth based on whose turn it is
            current_depth = depth1 if board.turn == chess.WHITE else depth2
            current_player = "White" if board.turn == chess.WHITE else "Black"
            
            print(f"\n{current_player} (depth={current_depth}) thinking...")
            
            # Get best move
            best_move_found, evaluation = self.best_move(board, current_depth)
            
            # CRITICAL: Check if move is None BEFORE trying to use it
            if best_move_found is None:
                print(f"{current_player} has no legal moves")
                print(f"Position FEN: {board.fen()}")
                
                # This should only happen if game is over
                # If not, there's a bug in your best_move function
                if not board.is_game_over():
                    print("BUG: No move found but game not over!")
                break
            
            # Display move info
            eval_str = f"{evaluation:+.1f}"
            
            print(f"{current_player} plays {best_move_found.uci()} with eval={eval_str}")
            
            # Store and execute move
            move_history.append(best_move_found.uci())
            board.push(best_move_found)
            
            # Check draw conditions
            if board.can_claim_threefold_repetition():
                print("\nDraw by threefold repetition")
                break
            
            if board.can_claim_fifty_moves():
                print("\nDraw by fifty-move rule")
                break
            
            move_number += 1
        
        # Game over - display results
        print("\n" + "="*60)
        print("GAME OVER")
        print("="*60)
        
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            winner_depth = depth1 if winner == "White" else depth2
            loser_depth = depth2 if winner == "White" else depth1
            print(f"{winner} (depth={winner_depth}) wins by checkmate!")
            print(f"   Black (depth={loser_depth}) was defeated")
        elif board.is_stalemate():
            print(" Draw by stalemate")
        elif board.is_insufficient_material():
            print(" Draw by insufficient material")
        elif move_number > max_moves:
            print(f" Draw by move limit ({max_moves} moves)")
        else:
            print(f"Game ended: {board.result()}")
        
        print(f"\nResult: {board.result()}")
        print(f"Total moves: {len(move_history)}")
        print(f"\nMove sequence: {move_history}")
        print("="*60 + "\n")
        
        return {
            'result': board.result(),
            'moves': move_history,
            'move_count': len(move_history),
            'white_depth': depth1,
            'black_depth': depth2,
            'final_fen': board.fen()
        }
if __name__ == "__main__":
    e = Engine()
    # e.self_play(4, 100)
    # # b=chess.Board('rn4k1/ppp1rpbp/4N1p1/3q3P/3pN3/7P/PPP2P2/R2QKB1R b KQ - 0 13')
    # print(e.best_move(b,5))
    e.compare(depth1=4,depth2=5,max_moves=150)
    # e.play_against_human(chess.WHITE,5)
    
    '''legendry game against @chess.com zamanatop:
    [Event "?"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "?"]
[Black "?"]
[Result "*"]
[Link "https://www.chess.com/classroom/smoggy-classic-turret"]

1. e4 d6 
2. Nf3 Nf6 
3. Nc3 g6 
4. d4 Bg7 
5. Bg5 O-O 
6. e5 Nh5 
7. h3 Be6 
8. g4 dxe5 
9. gxh5 exd4 
10. Ne4 Qd5 
11. Bxe7 Re8 
12. Nfg5 Rxe7 
13.Nxe6 Qxe4+ 

zamanatop resigns realising he is losing.
And so stockshark defeeated its first human opponent on 14/11/2025
    '''