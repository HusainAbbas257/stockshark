import chess
from evaluation import evaluate as evaluate_board
from evaluation import ordered_moves


class Engine:
    def minimax(self, board: 'chess.Board', depth, maximizing_color, alpha=float('-inf'), beta=float('inf')):
        """
        Minimax function with alpha-beta pruning and ordered moves to evaluate chess positions.
        """

        # Checkmate (depth-sensitive)
        if board.is_checkmate():
            # Depth-sensitive value to prioritize faster checkmates or slower losses
            return (float('inf') + depth) if board.turn == maximizing_color else (float('-inf') - depth)

        # Draw conditions
        if board.is_stalemate() or board.is_fivefold_repetition() or board.is_insufficient_material():
            return 0.0

        # Leaf node evaluation
        if depth == 0:
            return evaluate_board(board)  # Ensure evaluate_board is properly implemented

        # Retrieve ordered moves
        moves = ordered_moves(board)  # Use the `ordered_moves(board)` function to sort legal moves

        # MAXIMIZING
        if board.turn == maximizing_color:
            max_eval = float('-inf')
            for move in moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, maximizing_color, alpha, beta)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:  # Beta cutoff
                    break
            return max_eval

        # MINIMIZING
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, maximizing_color, alpha, beta)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:  # Alpha cutoff
                    break
            return min_eval
    def best_move(self, board: 'chess.Board', depth):
        """
        Finds the best move for the current player using the minimax algorithm with alpha-beta pruning.
        
        Args:
            board (chess.Board): Current chess board state.
            depth (int): Depth for the minimax search.

        Returns:
            tuple: Best move and its evaluation score.
        """
        maximizing_color = board.turn  # Determine if the current player is maximizing or minimizing
        best_val = float('-inf') if maximizing_color else float('inf')
        best_move = None

        # Use ordered_moves(board) to prioritize move evaluation (e.g., captures, promotions)
        moves = ordered_moves(board)

        for move in moves:
            board.push(move)  # Make the move
            eval_val = self.minimax(board, depth - 1, maximizing_color, alpha=float('-inf'), beta=float('inf'))
            board.pop()  # Undo the move

            if maximizing_color and eval_val > best_val:  # Maximizing player
                best_val = eval_val
                best_move = move
            elif not maximizing_color and eval_val < best_val:  # Minimizing player
                best_val = eval_val
                best_move = move

        return best_move, best_val


    def self_play(self, depth=3, max_moves=150):
        print("\n=== SELF PLAY START ===\n")
        moves = []
        board = chess.Board()
        move_num = 1

        while not board.is_game_over() and move_num <= max_moves:
            print(f"\n--- Move {move_num} ({'White' if board.turn else 'Black'}) ---")
            move,val = self.best_move(board, depth)
            if move is None:
                print("No legal move found.")
                break

            moves.append(move.uci())
            print(f"Playing: {move} wit eval={val}")
            board.push(move)
            # print(board)

            if board.is_fivefold_repetition() or board.is_stalemate():
                print("Draw by repetition or stalemate.")
                break

            move_num += 1

        print("\n=== GAME OVER ===")
        print("Result:", board.result())
        print("Move sequence:", moves)


    def play_against_human(self, color=chess.WHITE, depth=3):
        print("\n=== HUMAN VS AI ===\n")
        board = chess.Board()
        print(board)

        while not board.is_game_over():
            if board.turn == color:
                print("AI thinking...")
                move = self.best_move(board, depth)
                print(f"AI plays: {move}")
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


if __name__ == "__main__":
    e = Engine()
    e.self_play(3, 50)
    # b=chess.Board('7Q/6B1/8/3N4/1p6/kP6/2K5/8 w - - 6 7')
    # print(e.best_move(b,5))
