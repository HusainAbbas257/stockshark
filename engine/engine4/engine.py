import chess
from evaluation import evaluate as evaluate_board
from evaluation import ordered_moves



class Engine:
    def minimax(self, board: 'chess.Board', depth, maximizingPlayer, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or board.is_game_over():
            val = evaluate_board(board)
            return val if maximizingPlayer else -val

        for move in ordered_moves(board):
            b = board.copy()
            b.push(move)

            val = self.minimax(b, depth - 1, not maximizingPlayer, alpha, beta)

            if maximizingPlayer:
                alpha = max(alpha, val)
                if alpha >= beta:
                    break  # beta cut-off
            else:
                beta = min(beta, val)
                if beta <= alpha:
                    break  # alpha cut-off

        return alpha if maximizingPlayer else beta


    def best_move(self, board: chess.Board, depth=3):
        best_val = -float('inf') if board.turn == chess.WHITE else float('inf')
        best_move = None

        for move in board.legal_moves:
            b = board.copy()
            b.push(move)
            value = self.minimax(b, depth - 1, not board.turn)  # switch player

            if board.turn == chess.WHITE and value > best_val:
                best_val = value
                best_move = move
            elif board.turn == chess.BLACK and value < best_val:
                best_val = value
                best_move = move

        return best_move  

    def self_play(self, depth=3, max_moves=150):
        moves=[]
        board = chess.Board()
        move_num = 1
        while not board.is_game_over() and move_num <= max_moves:
            move = self.best_move(board, depth)
            if move is None:
                print("No legal move found even randomly.")
                break
            moves.append(move)
            print(f"Move {move_num}: {'White' if board.turn else 'Black'} plays {move}")
            board.push(move)
            move_num += 1
        print("Game over:", board.result())
        print(moves)

if __name__ == "__main__":
    e=Engine()
    e.self_play(4,50)