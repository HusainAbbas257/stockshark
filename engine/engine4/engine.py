import chess
from evaluation import evaluate as evaluate_board
from evaluation import ordered_moves


class Engine:
    def minimax(self, board: 'chess.Board', depth, maximizing_color, alpha=float('-inf'), beta=float('inf')):
        """
        Minimax with alpha-beta pruning. Evaluates chess positions recursively.
        Returns best evaluation score for current position.
        """
        
        # Terminal conditions check karo pehle - base cases
        
        # Checkmate: depth-adjusted score (faster mate = better/worse)
        if board.is_checkmate():
            # Agar maximizer ki turn pe checkmate hua = maximizer haar gaya
            return (float('-inf') - depth) if board.turn == maximizing_color else (float('inf') + depth)
        
        # Draw scenarios - neutral evaluation
        if (board.is_stalemate() or board.is_fivefold_repetition() or 
            board.is_insufficient_material() or board.is_repetition()):
            return 0.0
        
        # Depth limit hit = leaf node evaluation
        if depth == 0:
            return evaluate_board(board)
        
        # Move ordering for better pruning efficiency
        moves = ordered_moves(board)
        
        # MAXIMIZER: maximize evaluation
        if board.turn == maximizing_color:
            max_eval = float('-inf')
            for move in moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, maximizing_color, alpha, beta)
                board.pop()
                
                max_eval = max(max_eval, eval_score)  # Cleaner update
                alpha = max(alpha, eval_score)  # Update alpha window
                
                if beta <= alpha:  # Pruning condition
                    break  # Beta cutoff - baaki moves useless hain
            
            return max_eval
        
        # MINIMIZER: minimize evaluation
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, maximizing_color, alpha, beta)
                board.pop()
                
                min_eval = min(min_eval, eval_score)  # Cleaner update
                beta = min(beta, eval_score)  # Update beta window
                
                if beta <= alpha:  # Pruning condition
                    break  # Alpha cutoff - aage ka no point
            
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
        # This determines whether we're maximizing (White) or minimizing (Black) the score
        maximizing_color = board.turn
        
        # =============================================================================
        # INITIALIZE BEST MOVE TRACKING
        # =============================================================================
        
        # Initialize best evaluation found so far
        # Start with worst possible value for the current player:
        #   - If maximizing (White): start at -infinity (any move is better)
        #   - If minimizing (Black): start at +infinity (any move is better)
        best_evaluation = float('-inf') if maximizing_color else float('inf')
        
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
            return None, self.minimax(board, 0, maximizing_color)
        
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
            # Search one ply deeper (depth - 1) from opponent's perspective
            # The opponent will try to respond optimally
            # Note: We pass maximizing_color (not board.turn) to maintain perspective
            eval_score = self.minimax(
                board, 
                depth - 1, 
                maximizing_color,  # Keep original perspective (who we're optimizing for)
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
            
            if maximizing_color:
                # MAXIMIZING PLAYER (typically White)
                # Looking for highest evaluation score
                if eval_score > best_evaluation:
                    best_evaluation = eval_score
                    best_move_found = move
                    
                    # Update alpha (best score we can guarantee)
                    alpha = max(alpha, eval_score)
            
            else:
                # MINIMIZING PLAYER (typically Black)
                # Looking for lowest evaluation score
                if eval_score < best_evaluation:
                    best_evaluation = eval_score
                    best_move_found = move
                    
                    # Update beta (best score we can guarantee)
                    beta = min(beta, eval_score)
            
            # Optional early exit: If we've proven this branch is worse than others
            # (This is less useful at root since we want to evaluate all moves,
            # but included for completeness)
            # if beta <= alpha:
            #     break
        
        # =============================================================================
        # RETURN RESULTS
        # =============================================================================
        
        # Return the best move found and its evaluation score
        # best_move_found could theoretically be None if moves list was empty
        # but we handle that edge case earlier
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
    e.self_play(3, 25)
    # b=chess.Board('7Q/6B1/8/3N4/1p6/kP6/2K5/8 w - - 6 7')
    # print(e.best_move(b,5))
    # e.compare(depth1=3,depth2=3,max_moves=25)
    # e.play_against_human(chess.WHITE,4)