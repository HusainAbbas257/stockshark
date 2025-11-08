''' This module provides chess-related utilities.it is different from chess this file is named chesss  . I will try my best yo optimise it as much as possible. '''

from collections import Counter
pos_to_index = {
            'a8': 0, 'b8': 1, 'c8': 2, 'd8': 3, 'e8': 4, 'f8': 5, 'g8': 6, 'h8': 7,
            'a7': 8, 'b7': 9, 'c7': 10, 'd7': 11, 'e7': 12, 'f7': 13, 'g7': 14, 'h7': 15,
            'a6': 16, 'b6': 17, 'c6': 18, 'd6': 19, 'e6': 20, 'f6': 21, 'g6': 22, 'h6': 23,
            'a5': 24, 'b5': 25, 'c5': 26, 'd5': 27, 'e5': 28, 'f5': 29, 'g5': 30, 'h5': 31,
            'a4': 32, 'b4': 33, 'c4': 34, 'd4': 35, 'e4': 36, 'f4': 37, 'g4': 38, 'h4': 39,
            'a3': 40, 'b3': 41, 'c3': 42, 'd3': 43, 'e3': 44, 'f3': 45, 'g3': 46, 'h3': 47,
            'a2': 48, 'b2': 49, 'c2': 50, 'd2': 51, 'e2': 52, 'f2': 53, 'g2': 54, 'h2': 55,
            'a1': 56, 'b1': 57, 'c1': 58, 'd1': 59, 'e1': 60, 'f1': 61, 'g1': 62, 'h1': 63
        }

PIECE_TABLES = {
    'wp': [  # White pawns
        0,0,0,0,0,0,0,0,
        5,5,5,5,5,5,5,5,
        1,1,2,3,3,2,1,1,
        0.5,0.5,1,2.5,2.5,1,0.5,0.5,
        0,0,0,2,2,0,0,0,
        0.5,-0.5,-1,0,0,-1,-0.5,0.5,
        0.5,1,1,-2,-2,1,1,0.5,
        0,0,0,0,0,0,0,0
    ],
    'bp': [  # Black pawns (manual mirrored)
        0,0,0,0,0,0,0,0,
        0.5,1,1,-2,-2,1,1,0.5,
        0.5,-0.5,-1,0,0,-1,-0.5,0.5,
        0,0,0,2,2,0,0,0,
        0.5,0.5,1,2.5,2.5,1,0.5,0.5,
        1,1,2,3,3,2,1,1,
        5,5,5,5,5,5,5,5,
        0,0,0,0,0,0,0,0
    ],
    'wn': [
        -5,-4,-3,-3,-3,-3,-4,-5,
        -4,-2,0,0,0,0,-2,-4,
        -3,0,1,1.5,1.5,1,0,-3,
        -3,0,1.5,2,2,1.5,0,-3,
        -3,0,1.5,2,2,1.5,0,-3,
        -3,0,1,1.5,1.5,1,0,-3,
        -4,-2,0,0,0,0,-2,-4,
        -5,-4,-3,-3,-3,-3,-4,-5
    ],
    'bn': [
        -5,-4,-3,-3,-3,-3,-4,-5,
        -4,-2,0,0,0,0,-2,-4,
        -3,0,1,1.5,1.5,1,0,-3,
        -3,0,1.5,2,2,1.5,0,-3,
        -3,0,1.5,2,2,1.5,0,-3,
        -3,0,1,1.5,1.5,1,0,-3,
        -4,-2,0,0,0,0,-2,-4,
        -5,-4,-3,-3,-3,-3,-4,-5
    ],
    'wb': [
        -2,-1,-1,-1,-1,-1,-1,-2,
        -1,0,0,0,0,0,0,-1,
        -1,0,0.5,1,1,0.5,0,-1,
        -1,0.5,0.5,1,1,0.5,0.5,-1,
        -1,0,1,1,1,1,0,-1,
        -1,1,1,1,1,1,1,-1,
        -1,0.5,0,0,0,0,0.5,-1,
        -2,-1,-1,-1,-1,-1,-1,-2
    ],
    'bb': [
        -2,-1,-1,-1,-1,-1,-1,-2,
        -1,0.5,0,0,0,0,0.5,-1,
        -1,1,1,1,1,1,1,-1,
        -1,0,1,1,1,1,0,-1,
        -1,0.5,0.5,1,1,0.5,0.5,-1,
        -1,0,0.5,1,1,0.5,0,-1,
        -1,0,0,0,0,0,0,-1,
        -2,-1,-1,-1,-1,-1,-1,-2
    ],
    'wr': [
        0,0,0,0,0,0,0,0,
        0.5,1,1,1,1,1,1,0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        0,0,0,0.5,0.5,0,0,0
    ],
    'br': [
        0,0,0,0.5,0.5,0,0,0,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        -0.5,0,0,0,0,0,0,-0.5,
        0.5,1,1,1,1,1,1,0.5,
        0,0,0,0,0,0,0,0
    ],
    'wq': [
        -2,-1,-1,-0.5,-0.5,-1,-1,-2,
        -1,0,0,0,0,0,0,-1,
        -1,0,0.5,0.5,0.5,0.5,0,-1,
        -0.5,0,0.5,0.5,0.5,0.5,0,-0.5,
        0,0,0.5,0.5,0.5,0.5,0,-0.5,
        -1,0.5,0.5,0.5,0.5,0.5,0,-1,
        -1,0,0.5,0,0,0,0,-1,
        -2,-1,-1,-0.5,-0.5,-1,-1,-2
    ],
    'bq': [
        -2,-1,-1,-0.5,-0.5,-1,-1,-2,
        -1,0,0.5,0,0,0.5,0,-1,
        -1,0.5,0.5,0.5,0.5,0.5,0,-1,
        0,0,0.5,0.5,0.5,0.5,0,-0.5,
        -0.5,0,0.5,0.5,0.5,0.5,0,-0.5,
        -1,0,0.5,0.5,0.5,0.5,0,-1,
        -1,0,0,0,0,0,0,-1,
        -2,-1,-1,-0.5,-0.5,-1,-1,-2
    ],
    'wk': [
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -2,-3,-3,-4,-4,-3,-3,-2,
        -1,-2,-2,-2,-2,-2,-2,-1,
        2,2,0,0,0,0,2,2,
        2,3,1,0,0,1,3,2
    ],
    'bk': [
        2,3,1,0,0,1,3,2,
        2,2,0,0,0,0,2,2,
        -1,-2,-2,-2,-2,-2,-2,-1,
        -2,-3,-3,-4,-4,-3,-3,-2,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3,
        -3,-4,-4,-5,-5,-4,-4,-3
    ],
    '': []  # Empty cell
    }

class cell:
    def __init__(self,piece:str, pos:str):
        self.pos= pos
        # pos is in format 'e4', 'a1', etc.
        
        self.piece= piece
        # is in the form 'wp' for white pawn, 'bq' for black queen, etc. for empty cell it is ''
    
    def get_eval(self)->float:
        piece_value = {
            'p': 1,
            'n': 3.2,
            'b': 3.3,
            'r': 5,
            'q': 9,
            'k': 100,
            '': 0
        }
        if not self.piece:
            return 0
        
        value= piece_value[self.piece[1]] 
        return value if self.piece[0] == 'w' else -value
    def get_value(self)->float:
        # Evaluate the board position based on piece-square tables
        if not self.piece:
            return 0
        
        
        piece_table = PIECE_TABLES.get(self.piece)
        if piece_table:
            index = pos_to_index.get(self.pos)
            if index is not None:
                return piece_table[index]
        return 0
    
    

class Move:
    def __init__(self,board:'Board' ,from_:cell, to_:cell,promote_to:str=''):
        # from_ and to_ are cell objects and promote_to is piece type for promotion like 'q','r','b','n'
        self.board= board
        self.f=from_
        self.t=to_
        self.promote_to= promote_to
        
    @staticmethod
    def from_uci(board:'Board',uci_str:str)->'Move':
        # Convert UCI string to Move object
        if not isinstance(uci_str, str) or len(uci_str) < 4:
            raise ValueError("Invalid UCI string")
        from_pos = uci_str[0:2]
        to_pos = uci_str[2:4]
        from_pos = from_pos.strip()
        to_pos = to_pos.strip()
        board_piece_from = board.piece_at(from_pos)
        board_piece_to = board.piece_at(to_pos)
        promote_to=uci_str[4:] if len(uci_str)>4 else ''
        if board_piece_from is None or board_piece_to is None:
            raise ValueError("Invalid UCI string")
        # debug print removed
        return Move(board,board_piece_from, board_piece_to,promote_to)
    def __str__(self):
        # Return uci representation of the move
        uci_str = f"{self.f.pos}{self.t.pos}"
        if self.promote_to:
            uci_str += self.promote_to
        return uci_str

    


class Board:
    def __init__(self, FEN: str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.update_from_fen(FEN)
    def copy(self) -> 'Board':
        new_board = Board(self.FEN)
        return new_board
    def pop(self):
        """Reverts the last move made on the board."""
        if not hasattr(self, 'move_history') or not self.move_history:
            return  # No moves to revert

        # remove last move and last fen
        self.move_history.pop()
        self.fENs_history.pop()
        # restore board to previous FEN (last in list)
        last_fen = self.fENs_history[-1]
        # update fields from fen without wiping history
        self._load_fen_without_history(last_fen)
        
    def _load_fen_without_history(self, FEN: str):
        # internal helper: load board fields from FEN but keep histories intact
        self.FEN = FEN
        self.grid = self.get_grid(FEN)
        parts = FEN.split(' ')
        self.turn = parts[1]
        self.castling_rights = '' if parts[2] == '-' else parts[2]
        self.en_passant_target = None if parts[3] == '-' else parts[3]
        self.halfmove_clock = int(parts[4])
        self.fullmove_number = int(parts[5])
        self.state = 'ongoing'
        self.non_empty_squares = [c for r in self.grid for c in r if c.piece]

    def get_grid(self, FEN: str) -> list[list['cell']]:
        rows = FEN.split(' ')[0].split('/')
        grid = []
        for r, row in enumerate(rows):
            grid_row, col = [], 0
            for char in row:
                if char.isdigit():
                    for _ in range(int(char)):
                        grid_row.append(cell('', f"{chr(col + ord('a'))}{8 - r}"))
                        col += 1
                else:
                    color = 'w' if char.isupper() else 'b'
                    grid_row.append(cell(f"{color}{char.lower()}", f"{chr(col + ord('a'))}{8 - r}"))
                    col += 1
            grid.append(grid_row)
        return grid

    def update_from_fen(self, FEN: str):
        self.FEN = FEN
        self.grid = self.get_grid(FEN)
        parts = FEN.split(' ')
        self.turn = parts[1]
        self.castling_rights = '' if parts[2] == '-' else parts[2]
        self.en_passant_target = None if parts[3] == '-' else parts[3]
        self.halfmove_clock = int(parts[4])
        self.fullmove_number = int(parts[5])
        self.state = 'ongoing'
        self.non_empty_squares = [c for r in self.grid for c in r if c.piece]
        # initialize histories only if not present (so pop doesn't wipe them)
        if not hasattr(self, 'fENs_history'):
            self.fENs_history = [FEN]
        if not hasattr(self, 'move_history'):
            self.move_history = []

    def is_square_attacked(self, pos: str, by_color: str) -> bool:
        for piece in self.non_empty_squares:
            if piece.piece[0] == by_color:
                for move in self.legal_move(piece):
                    if move.t.pos == pos:
                        return True
        return False
    
    def piece_at(self, pos: str) -> 'cell':
        # return a valid cell object for out-of-board positions (prevents attribute errors)
        if not pos or len(pos) < 2:
            return cell('', pos)
        file, rank = pos[0], int(pos[1])
        row = 8 - rank
        col = ord(file) - ord('a')
        if 0 <= row < 8 and 0 <= col < 8:
            return self.grid[row][col]
        # return an empty cell for invalid squares (caller should check if needed)
        return cell('', pos)

    def legal_moves(self) -> list['Move']:
        return [m for p in self.non_empty_squares if p.piece and p.piece[0] == self.turn for m in self.legal_move(p)]

    def is_checkmate(self) -> bool:
        king = self.find_king(self.turn)
        if king is None:
            # no king on board -> treat as checkmate (invalid state)
            return True
        king_pos = king.pos
        return not self.legal_moves() and self.is_square_attacked(king_pos, 'b' if self.turn=='w' else 'w')

    def is_stalemate(self) -> bool:
        king = self.find_king(self.turn)
        if king is None:
            return False
        king_pos = king.pos
        if not self.legal_moves() and not self.is_square_attacked(king_pos, 'b' if self.turn=='w' else 'w'):
            return True

        pieces = [c.piece for c in self.non_empty_squares]
        minor = [p for p in pieces if p and len(p) > 1 and p[1] in 'bn']

        # Insufficient material
        if len(pieces) <= 3 and (len(pieces) == 2 or (len(pieces) == 3 and len(minor) == 1)):
            return True
        # Fifty-move rule
        if self.halfmove_clock >= 100:
            return True
        return False
    def find_king(self,color):
        for r in self.grid:
            for c in r:
                if c.piece == f"{color}k":
                    return c
        return None
    def update_state(self):
        if self.is_checkmate():
            self.state = '0-1' if self.turn=='w' else '1-0'
        elif self.is_stalemate():
            self.state = '1/2-1/2'

    def fen(self) -> str:
        rows = []
        for r in self.grid:
            row_str, empty = '', 0
            for c in r:
                if not c.piece:
                    empty += 1
                else:
                    if empty: row_str += str(empty); empty = 0
                    row_str += c.piece[1].upper() if c.piece[0]=='w' else c.piece[1]
            if empty: row_str += str(empty)
            rows.append(row_str)
        return f"{'/'.join(rows)} {self.turn} {self.castling_rights or '-'} {self.en_passant_target or '-'} {self.halfmove_clock} {self.fullmove_number}"
    def legal_move(self, piece: 'cell') -> list['Move']:
        moves = []
        if not piece.piece:
            return moves

        enemy = 'b' if piece.piece[0]=='w' else 'w'
        file, rank = piece.pos[0], int(piece.pos[1])

        def on_board(f,r):
            return 'a'<=f<='h' and 1<=r<=8

        def add(dest_pos, promote=None):
            target = self.piece_at(dest_pos)
            if not target.piece or target.piece[0]!=piece.piece[0]:
                moves.append(Move(self, piece, target, promote_to=promote))

        # Pawn moves
        if piece.piece[1]=='p':
            direction = 1 if piece.piece[0]=='w' else -1
            start = 2 if piece.piece[0]=='w' else 7
            promotion_rank = 8 if piece.piece[0]=='w' else 1

            # Forward 1
            forward1 = f"{file}{rank+direction}"
            if self.piece_at(forward1).piece=='':
                if rank+direction==promotion_rank:
                    for promo in ['q','r','b','n']:
                        add(forward1, promo)
                else:
                    add(forward1)
                # Forward 2
                if rank==start:
                    forward2 = f"{file}{rank+2*direction}"
                    if self.piece_at(forward2).piece=='':
                        add(forward2)
            # Captures
            for df in [-1,1]:
                f_capture = chr(ord(file)+df)
                r_capture = rank+direction
                if on_board(f_capture,r_capture):
                    dest = f"{f_capture}{r_capture}"
                    target = self.piece_at(dest)
                    if target.piece and target.piece[0]==enemy:
                        if r_capture==promotion_rank:
                            for promo in ['q','r','b','n']:
                                add(dest, promo)
                        else:
                            add(dest)
            # En passant
            if self.en_passant_target:
                ep_rank = int(self.en_passant_target[1])
                if rank==ep_rank-direction and abs(ord(file)-ord(self.en_passant_target[0]))==1:
                    add(self.en_passant_target)
            return moves

        # Knight moves
        if piece.piece[1]=='n':
            for df,dr in [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]:
                f2 = chr(ord(file)+df)
                r2 = rank+dr
                if on_board(f2,r2):
                    add(f"{f2}{r2}")
            return moves

        # Sliding pieces
        def slide(dirs):
            for df,dr in dirs:
                for step in range(1,8):
                    f2 = chr(ord(file)+df*step)
                    r2 = rank+dr*step
                    if not on_board(f2,r2): break
                    target = self.piece_at(f"{f2}{r2}")
                    if not target.piece:
                        add(f"{f2}{r2}")
                    elif target.piece[0]!=piece.piece[0]:
                        add(f"{f2}{r2}")
                        break
                    else:
                        break

        if piece.piece[1]=='b': slide([(1,1),(1,-1),(-1,1),(-1,-1)]); return moves
        if piece.piece[1]=='r': slide([(1,0),(-1,0),(0,1),(0,-1)]); return moves
        if piece.piece[1]=='q': slide([(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]); return moves

        # King
        if piece.piece[1]=='k':
            for df,dr in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                f2 = chr(ord(file)+df)
                r2 = rank+dr
                if on_board(f2,r2):
                    add(f"{f2}{r2}")
            # Castling
            rank_str = '1' if piece.piece[0]=='w' else '8'
            # Kingside
            if ('K' in self.castling_rights and piece.piece[0]=='w') or ('k' in self.castling_rights and piece.piece[0]=='b'):
                if self.piece_at(f"f{rank_str}").piece=='' and self.piece_at(f"g{rank_str}").piece=='':
                    add(f"g{rank_str}")
            # Queenside
            if ('Q' in self.castling_rights and piece.piece[0]=='w') or ('q' in self.castling_rights and piece.piece[0]=='b'):
                if self.piece_at(f"d{rank_str}").piece=='' and self.piece_at(f"c{rank_str}").piece=='' and self.piece_at(f"b{rank_str}").piece=='':
                    add(f"c{rank_str}")
            return moves

        return moves
    def push(self, move: 'Move'):
        """Executes the given move on the board, updating all relevant state."""
        # Move piece
        from_cell = move.f
        to_cell = move.t
        piece = from_cell.piece
        captured = to_cell.piece

        # Handle promotion
        if move.promote_to:
            to_cell.piece = piece[0] + move.promote_to
        else:
            to_cell.piece = piece

        from_cell.piece = ''

        # Update en passant
        self.en_passant_target = None
        if piece[1] == 'p' and abs(int(from_cell.pos[1]) - int(to_cell.pos[1])) == 2:
            ep_rank = (int(from_cell.pos[1]) + int(to_cell.pos[1])) // 2
            self.en_passant_target = f"{from_cell.pos[0]}{ep_rank}"

        # Handle en passant capture
        if piece[1]=='p' and to_cell.pos == self.en_passant_target:
            # captured pawn is behind the target square
            ep_capture_rank = int(to_cell.pos[1]) - (1 if piece[0]=='w' else -1)
            ep_captured_cell = self.piece_at(f"{to_cell.pos[0]}{ep_capture_rank}")
            if ep_captured_cell and ep_captured_cell.piece and ep_captured_cell.piece[0]!=piece[0]:
                ep_captured_cell.piece = ''

        # Update castling rights if rook or king moved
        if piece[1]=='k':
            if piece[0]=='w': self.castling_rights = self.castling_rights.replace('K','').replace('Q','')
            else: self.castling_rights = self.castling_rights.replace('k','').replace('q','')
        if piece[1]=='r':
            if from_cell.pos=='a1': self.castling_rights = self.castling_rights.replace('Q','')
            if from_cell.pos=='h1': self.castling_rights = self.castling_rights.replace('K','')
            if from_cell.pos=='a8': self.castling_rights = self.castling_rights.replace('q','')
            if from_cell.pos=='h8': self.castling_rights = self.castling_rights.replace('k','')

        # Update halfmove clock
        if piece[1]=='p' or captured:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Update fullmove number
        if self.turn=='b':
            self.fullmove_number += 1

        # Switch turn
        self.turn = 'b' if self.turn=='w' else 'w'

        # Update non-empty squares
        self.non_empty_squares = [c for r in self.grid for c in r if c.piece]

        # Update game state
        self.update_state()
        # append history
        if not hasattr(self, 'fENs_history'):
            self.fENs_history = [self.fen()]
        else:
            self.fENs_history.append(self.fen())
        if not hasattr(self, 'move_history'):
            self.move_history = [move]
        else:
            self.move_history.append(move)

    def display(self):
        for r in self.grid:
            for c in r:
                if not c.piece:
                    print("\033[30m·\033[0m", end=' ')
                elif c.piece[0] == 'w':
                    print(f"\033[31m{c.piece[1]}\033[0m", end=' ')
                elif c.piece[0] == 'b':
                    print(f"\033[33m{c.piece[1]}\033[0m", end=' ')
            print()
        print('\n\n')
    


if __name__ == "__main__":
    import random
    board = Board()
    board.display()
    
    while not (board.is_checkmate() or board.is_stalemate()):
        moves=board.legal_moves()
        if not moves: break
        move=random.choice(moves)
        board.push(move)
        board.display()

