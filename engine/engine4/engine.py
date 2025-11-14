import chess
from evaluation import evaluate as evaluate_board
from evaluation import ordered_moves


class Engine:
    def minimax(self, board: 'chess.Board', depth: int, is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> float:
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
        if board.is_checkmate():
            # If it's maximizer's turn and they're checkmated, they lost
            if is_maximizing:
                return float('-inf') + depth  # Losing score (prefer slower losses)
            else:
                return float('inf') - depth  # Winning score (prefer faster wins)
        
        # Draw scenarios - always return neutral evaluation
        if (board.is_stalemate() or 
            board.is_fivefold_repetition() or 
            board.is_insufficient_material() or 
            board.is_repetition()):
            return 0.0
        
        # =============================================================================
        # DEPTH LIMIT - Leaf node evaluation
        # =============================================================================
        
        # When depth reaches 0, evaluate the position statically
        # This is the base case for the recursion
        if depth == 0:
            return evaluate_board(board)
        
        # =============================================================================
        # MOVE GENERATION - Order moves for better pruning
        # =============================================================================
        
        # Get moves sorted by heuristic priority (captures, checks, etc. first)
        # Better move ordering = more alpha-beta cutoffs = faster search
        moves = ordered_moves(board)
        
        # =============================================================================
        # MAXIMIZING PLAYER - White's turn (or whoever is maximizing)
        # =============================================================================
        
        if is_maximizing:
            max_eval = float('-inf')  # Start with worst possible score
            
            for move in moves:
                # Make move
                board.push(move)
                
                # Recursively evaluate from opponent's perspective (minimizing)
                # Key fix: pass 'not is_maximizing' to alternate perspectives
                eval_score = self.minimax(board, depth - 1, not is_maximizing, alpha, beta)
                
                # Undo move
                board.pop()
                
                # Update best score found
                max_eval = max(max_eval, eval_score)
                
                # Update alpha (best score maximizer can guarantee so far)
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning: if alpha >= beta, minimizer won't allow this path
                if beta <= alpha:
                    break  # Beta cutoff - remaining moves can't be better
            
            return max_eval
        
        # =============================================================================
        # MINIMIZING PLAYER - Black's turn (or whoever is minimizing)
        # =============================================================================
        
        else:
            min_eval = float('inf')  # Start with worst possible score for minimizer
            
            for move in moves:
                # Make move
                board.push(move)
                
                # Recursively evaluate from opponent's perspective (maximizing)
                # Key fix: pass 'not is_maximizing' to alternate perspectives
                eval_score = self.minimax(board, depth - 1, not is_maximizing, alpha, beta)
                
                # Undo move
                board.pop()
                
                # Update best score found
                min_eval = min(min_eval, eval_score)
                
                # Update beta (best score minimizer can guarantee so far)
                beta = min(beta, eval_score)
                
                # Alpha-beta pruning: if beta <= alpha, maximizer won't allow this path
                if beta <= alpha:
                    break  # Alpha cutoff - remaining moves can't be better
            
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
        moves = ordered_moves(board)
        
        # =============================================================================
        # EDGE CASE - No legal moves available
        # =============================================================================
        
        # If there are no legal moves, the game is over (checkmate or stalemate)
        # Return None for move and current evaluation
        if not moves:
            return None, self.evaluate(board)
        
        # =============================================================================
        # ROOT MOVE ITERATION - Evaluate each legal move
        # =============================================================================
        
        # Alpha-beta window for root node
        # Always start with full window since we need to evaluate all moves at root
        alpha = float('-inf')  # Best score maximizer can guarantee
        beta = float('inf')    # Best score minimizer can guarantee
        
        for move in moves:
            # ---------------------------------------------------------------------
            # Make the move temporarily
            # ---------------------------------------------------------------------
            board.push(move)
            
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
            # Restore board state
            # ---------------------------------------------------------------------
            board.pop()
            
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
                print("ERROR: No legal move found (unexpected)")
                print(f"Position: {board.fen()}")
                break

            eval_display = f"{evaluation:+.1f}" if abs(evaluation) < 1000 else \
                        ("+inf" if evaluation > 0 else "-inf")
            print(f"Move: {best_move_found.uci():6s} \t Eval: {eval_display:>6s}")

            # push move to update the board
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
        print(board)

        while not board.is_game_over():
            if board.turn == color:
                print("AI thinking...")
                move,_ = self.best_move(board, depth)
                print(f"AI plays: {move}")
                if move!=None:
                    board.push(move)
            else:
                print("Your turn!")
                user_move = input("Enter move (e.g. e2e4): ")
                try:
                    board.push_uci(user_move)
                except:
                    print("Invalid move, try again.")
                    continue
            print(board)

        print("\n=== GAME OVER ===")
        print("Result:", board.result())
        print()
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
            eval_str = f"{evaluation:+.1f}" if abs(evaluation) < 10000 else \
                    ("+inf" if evaluation > 0 else "-inf")
            
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
        print(f"\nMove sequence: {' '.join(move_history)}")
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
    e.self_play(4, 50)
    # b=chess.Board('7Q/6B1/8/3N4/1p6/kP6/2K5/8 w - - 6 7')
    # print(e.best_move(b,5))
    # e.compare(depth1=3,depth2=3,max_moves=25)
    # e.play_against_human(chess.WHITE,4)