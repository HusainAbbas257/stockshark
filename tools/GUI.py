import tkinter as tk
from tkinter import Button as btn, messagebox as msg, simpledialog
from PIL import Image, ImageTk, Image

class cell:
    def __init__(self, color:str, pos:tuple, piece:str="e"):
        self.colour = color
        self.pos = pos
        self.piece = piece
    def __str__(self):
        return self.pos[0] + str(self.pos[1])

class Board:
    def __init__(self, FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.turn = 'w'
        self.grid = [[None for _ in range(8)] for _ in range(8)]
        # Castling rights & en-passant square storage
        self.castling_rights = {'w_k': True, 'w_q': True, 'b_k': True, 'b_q': True}
        self.en_passant_target = None  # e.g. ('e',6) or None
        self.FENtoGrid(FEN)

    def FENtoGrid(self, FEN):
        rows = FEN.split()[0].split('/')
        for i, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    for _ in range(int(char)):
                        self.grid[i][col_idx] = cell(
                            color="w" if (i+col_idx)%2==0 else "b",
                            pos=(chr(col_idx+97), 8-i),
                            piece="e"
                        )
                        col_idx += 1
                else:
                    self.grid[i][col_idx] = cell(
                        color="w" if (i+col_idx)%2==0 else "b",
                        pos=(chr(col_idx+97), 8-i),
                        piece=char
                    )
                    col_idx += 1

    def move(self, from_pos:tuple, to_pos:tuple, promote_to: str = None):
        """
        Performs move and returns list of positions (like ('e',2)) that changed visually.
        This helps GUI decide what canvases to refresh (minimal refresh).
        """
        changed = []
        from_col = ord(from_pos[0]) - 97
        from_row = 8 - from_pos[1]
        to_col = ord(to_pos[0]) - 97
        to_row = 8 - to_pos[1]

        piece = self.grid[from_row][from_col].piece
        if piece == "e":
            return changed

        # We'll record squares that change
        changed.append(from_pos)
        changed.append(to_pos)

        # reset en-passant target by default
        self.en_passant_target = None

        # Pawn double push -> set en_passant_target
        if piece.lower() == 'p' and abs(from_row - to_row) == 2:
            ep_row = (from_row + to_row) // 2
            self.en_passant_target = (chr(from_col + 97), 8 - ep_row)

        # En-passant capture detection: moving pawn diagonally to empty square that is en-passant target
        if piece.lower() == 'p' and from_col != to_col and self.grid[to_row][to_col].piece == "e":
            # captured pawn sits behind target square
            captured_row = to_row + (1 if piece.isupper() else -1)
            if 0 <= captured_row < 8:
                # clear captured pawn
                self.grid[captured_row][to_col].piece = "e"
                # include captured pawn square in changed list
                cap_pos = (chr(to_col + 97), 8 - captured_row)
                if cap_pos not in changed:
                    changed.append(cap_pos)

        # Castling: if king moves two squares, move rook accordingly
        castling_rook_from = None
        castling_rook_to = None
        if piece.lower() == 'k' and abs(from_col - to_col) == 2:
            # kingside
            if to_col == 6:
                # move rook from h to f (7->5)
                self.grid[to_row][5].piece = self.grid[to_row][7].piece
                self.grid[to_row][7].piece = "e"
                castling_rook_from = ( 'h', 8 - to_row )
                castling_rook_to   = ( 'f', 8 - to_row )
            else:
                # queenside: move rook from a to d (0->3)
                self.grid[to_row][3].piece = self.grid[to_row][0].piece
                self.grid[to_row][0].piece = "e"
                castling_rook_from = ( 'a', 8 - to_row )
                castling_rook_to   = ( 'd', 8 - to_row )
            # record rook squares changed
            if castling_rook_from and castling_rook_from not in changed:
                changed.append(castling_rook_from)
            if castling_rook_to and castling_rook_to not in changed:
                changed.append(castling_rook_to)

        # Basic move (king/rook moves handled above for rook movement)
        self.grid[to_row][to_col].piece = piece
        self.grid[from_row][from_col].piece = "e"

        # Promotion handling
        if piece.lower() == 'p' and (to_row == 0 or to_row == 7):
            if promote_to is None:
                promote_to = 'Q' if piece.isupper() else 'q'
            self.grid[to_row][to_col].piece = promote_to

        # Update castling rights (basic)
        if piece == 'K':
            self.castling_rights['w_k'] = False
            self.castling_rights['w_q'] = False
        if piece == 'k':
            self.castling_rights['b_k'] = False
            self.castling_rights['b_q'] = False
        # If rook moved from corner, disable appropriate right
        if from_pos == ('h',1): self.castling_rights['w_k'] = False
        if from_pos == ('a',1): self.castling_rights['w_q'] = False
        if from_pos == ('h',8): self.castling_rights['b_k'] = False
        if from_pos == ('a',8): self.castling_rights['b_q'] = False
        # If a rook was captured on corner - basic disable (scan corners)
        corners = {('a',1):'w_q', ('h',1):'w_k', ('a',8):'b_q', ('h',8):'b_k'}
        for pos, key in corners.items():
            col = ord(pos[0]) - 97
            row = 8 - int(pos[1])
            if self.grid[row][col].piece.lower() != 'r':
                self.castling_rights[key] = False

        return changed
    def cell_At(self, pos:str):
        col = ord(pos[0]) - 97
        row = 8 - int(pos[1])
        return self.grid[row][col]
    def is_in_check(self, color):
        # Find king position
        king_pos = None
        for rowi, row in enumerate(self.grid):
            for coli, cell in enumerate(row):
                if (color == 'w' and cell.piece == 'K') or (color == 'b' and cell.piece == 'k'):
                    king_pos = (rowi, coli)  # store as int position
                    break
            if king_pos:
                break

        if not king_pos:
            return False



    def show_moves(self, cell:cell):
        reachable_cells = []
        if cell.piece == "e":
            return reachable_cells
        col = ord(cell.pos[0]) - 97
        row = 8 - cell.pos[1]

        # Pawn
        if cell.piece.lower() == 'p':
            direction = -1 if cell.piece.isupper() else 1
            start_row = 6 if cell.piece.isupper() else 1
            # forward one
            if 0 <= row + direction < 8 and self.grid[row + direction][col].piece == "e":
                reachable_cells.append(self.grid[row + direction][col])
                # double
                if row == start_row and self.grid[row + 2*direction][col].piece == "e":
                    reachable_cells.append(self.grid[row + 2*direction][col])
            # captures
            for dc in [-1, 1]:
                nc = col + dc
                if 0 <= nc < 8 and 0 <= row + direction < 8:
                    target = self.grid[row + direction][nc]
                    if target.piece != "e" and target.piece.islower() != cell.piece.islower():
                        reachable_cells.append(target)
            # en-passant capture: check stored target
            if self.en_passant_target:
                ep_col = ord(self.en_passant_target[0]) - 97
                ep_row = 8 - self.en_passant_target[1]
                # if en-passant square is one step forward diagonally
                if (ep_row == row + direction) and (abs(ep_col - col) == 1):
                    reachable_cells.append(self.grid[ep_row][ep_col])

        # Rook
        if cell.piece.lower() == 'r':
            directions = [(1,0),(-1,0),(0,1),(0,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    t = self.grid[r][c]
                    if t.piece == "e":
                        reachable_cells.append(t)
                    else:
                        if t.piece.islower() != cell.piece.islower():
                            reachable_cells.append(t)
                        break
                    r += dr; c += dc

        # Knight
        if cell.piece.lower() == 'n':
            knight_moves = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
            for dr, dc in knight_moves:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    t = self.grid[r][c]
                    if t.piece == "e" or t.piece.islower() != cell.piece.islower():
                        reachable_cells.append(t)

        # Bishop
        if cell.piece.lower() == 'b':
            directions = [(1,1),(1,-1),(-1,1),(-1,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    t = self.grid[r][c]
                    if t.piece == "e":
                        reachable_cells.append(t)
                    else:
                        if t.piece.islower() != cell.piece.islower():
                            reachable_cells.append(t)
                        break
                    r += dr; c += dc

        # Queen
        if cell.piece.lower() == 'q':
            directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    t = self.grid[r][c]
                    if t.piece == "e":
                        reachable_cells.append(t)
                    else:
                        if t.piece.islower() != cell.piece.islower():
                            reachable_cells.append(t)
                        break
                    r += dr; c += dc

        # King (basic moves + simple castling availability)
        if cell.piece.lower() == 'k':
            directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    t = self.grid[r][c]
                    if t.piece == "e" or t.piece.islower() != cell.piece.islower():
                        reachable_cells.append(t)
            # Castling (basic: squares empty and rook present; does NOT check checks)
            if cell.piece == 'K' and row == 7 and col == 4:
                # white
                if self.castling_rights.get('w_k') and self.grid[row][7].piece.lower() == 'r' and all(self.grid[row][c].piece == "e" for c in (5,6)):
                    reachable_cells.append(self.grid[row][6])
                if self.castling_rights.get('w_q') and self.grid[row][0].piece.lower() == 'r' and all(self.grid[row][c].piece == "e" for c in (1,2,3)):
                    reachable_cells.append(self.grid[row][2])
            if cell.piece == 'k' and row == 0 and col == 4:
                # black
                if self.castling_rights.get('b_k') and self.grid[row][7].piece.lower() == 'r' and all(self.grid[row][c].piece == "e" for c in (5,6)):
                    reachable_cells.append(self.grid[row][6])
                if self.castling_rights.get('b_q') and self.grid[row][0].piece.lower() == 'r' and all(self.grid[row][c].piece == "e" for c in (1,2,3)):
                    reachable_cells.append(self.grid[row][2])
            legal_moves = []
            for target in reachable_cells:
                # make a trial move
                temp = Board(self.to_FEN())
                temp.move(cell.pos, target.pos)
                # only keep if not in check
                if not temp.is_in_check('w' if cell.piece.isupper() else 'b'):
                    legal_moves.append(target)
            return legal_moves

        return reachable_cells

    def to_FEN(self,grid=None):
        if grid is None:
            grid = self.grid
        fen_rows = []
        for row in grid:
            cnt = 0
            s = ""
            for c in row:
                if c.piece == "e":
                    cnt += 1
                else:
                    if cnt>0: s += str(cnt); cnt=0
                    s += c.piece
            if cnt>0: s += str(cnt)
            fen_rows.append(s)
        return "/".join(fen_rows) + " w KQkq - 0 1"

class GUI:
    def __init__(self):
        self.selected_piece = None
        self.reachable_cells = []
        # store canvases so we can update individual squares
        self.cell_canvases = [[None for _ in range(8)] for _ in range(8)]
        self.current_board = Board()
        self.FEN = self.current_board.to_FEN()
        self.board_frame = None  # will be created in run()

    def run(self):
        self.root = tk.Tk()
        self.root.title("Stockshark - Board")
        # top controls (menu/buttons) should remain intact -> place them before board frame
        btn(self.root, text="Single Player", command=lambda: msg.showinfo("Info","Single Player under development")).pack(pady=6)
        btn(self.root, text="Multiplayer", command=self.multiplayer).pack(pady=6)
        # create a dedicated frame for the board so we do NOT destroy the menu
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack()
        self.display_board(self.board_frame)
        self.root.mainloop()

    def cell_button(self, cell:cell, parent_frame:tk.Frame):
        color = "#F0D9B5" if cell.colour=="w" else "#B67C4D"
        canvas = tk.Canvas(parent_frame,width=55,height=55,bg=color,highlightthickness=0)
        self.update_canvas(cell, canvas)
        canvas.bind("<Button-1>", lambda e,c=cell,cv=canvas: self.show_moves(c,cv))
        return canvas

    def update_canvas(self, cell_or_pos, canvas=None):
        """
        Accepts either a cell object or a position tuple ('e',2).
        If position tuple passed, fetch corresponding cell from board.
        """
        if isinstance(cell_or_pos, tuple):
            pos = cell_or_pos
            r = 8 - pos[1]; c = ord(pos[0]) - 97
            cell_obj = self.current_board.grid[r][c]
        else:
            cell_obj = cell_or_pos
            r = 8 - cell_obj.pos[1]; c = ord(cell_obj.pos[0]) - 97

        if canvas is None: canvas = self.cell_canvases[r][c]
        canvas.delete("all")
        color = "#F0D9B5" if cell_obj.colour=="w" else "#B67C4D"
        canvas.configure(bg=color)
        if cell_obj.piece != "e":
            path="data/images/"
            imgs = {
                'P':'white-pawn.png','R':'white-rook.png','N':'white-knight.png','B':'white-bishop.png',
                'Q':'white-queen.png','K':'white-king.png','p':'black-pawn.png','r':'black-rook.png',
                'n':'black-knight.png','b':'black-bishop.png','q':'black-queen.png','k':'black-king.png'
            }
            if cell_obj.piece in imgs:
                img = Image.open(path+imgs[cell_obj.piece]).convert("RGBA").resize((55,55), Image.Resampling.LANCZOS)
                # merge with square color to avoid black bg
                square_rgb = (240,217,181) if cell_obj.colour=="w" else (182,124,77)
                background = Image.new("RGBA", img.size, square_rgb+(255,))
                img = Image.alpha_composite(background, img)
                photo = ImageTk.PhotoImage(img)
                canvas.create_image(27,27,image=photo)
                canvas.image = photo

    def show_moves(self, clicked_cell:cell, clicked_canvas:tk.Canvas):
        # Clear highlights first (only on existing canvases)
        for r in range(8):
            for c in range(8):
                if self.cell_canvases[r][c]:
                    self.cell_canvases[r][c].delete("highlight")

        # If a source selected and clicked_cell is a reachable target (compare pos)
        if self.selected_piece and any(clicked_cell.pos == rc.pos for rc in self.reachable_cells):
            # turn-check
            if (self.selected_piece.piece.isupper() and self.current_board.turn=='w') or \
               (self.selected_piece.piece.islower() and self.current_board.turn=='b'):
                from_pos = self.selected_piece.pos
                to_pos = clicked_cell.pos
                # promotion prompt if needed
                piece = self.selected_piece.piece
                promote_to = None
                if piece.lower() == 'p':
                    to_row = 8 - to_pos[1]
                    if to_row == 0 or to_row == 7:
                        # ask for promotion piece: q/r/b/n
                        choice = simpledialog.askstring("Promotion", "Promote to (q/r/b/n). Default q:", parent=self.root)
                        if not choice: choice = 'q'
                        choice = choice.lower()[0]
                        mapping = {'q':'Q','r':'R','b':'B','n':'N'}
                        promote_to = mapping.get(choice, 'Q') if piece.isupper() else mapping.get(choice, 'q').lower()
                # perform move on board and get changed squares list
                changed_positions = self.current_board.move(from_pos, to_pos, promote_to=promote_to)
                # toggle turn
                self.current_board.turn = 'b' if self.current_board.turn == 'w' else 'w'
                self.FEN = self.current_board.to_FEN()
                # Update only changed squares (minimal refresh)
                for pos in changed_positions:
                    # pos is like ('e',2)
                    self.update_canvas(pos)
                self.selected_piece = None
                self.reachable_cells = []
            return

        # Select new piece if it's that side's turn
        if clicked_cell.piece != "e" and ((clicked_cell.piece.isupper() and self.current_board.turn=='w') or (clicked_cell.piece.islower() and self.current_board.turn=='b')):
            self.selected_piece = clicked_cell
            self.reachable_cells = self.current_board.show_moves(clicked_cell)
            # highlight selection and targets
            clicked_canvas.create_rectangle(2,2,53,53, outline="#A80303", width=3, tags="highlight")
            for rc in self.reachable_cells:
                r_idx = 8 - rc.pos[1]; c_idx = ord(rc.pos[0]) - 97
                cv = self.cell_canvases[r_idx][c_idx]
                cv.create_oval(22,22,32,32, fill="blue", tags="highlight")
        else:
            self.selected_piece = None
            self.reachable_cells = []

    def display_board(self, parent_frame=None):
        # Use the dedicated board_frame created in run()
        if parent_frame is None:
            parent_frame = self.board_frame
        # Clear only the board_frame children (do NOT destroy menu)
        for widget in parent_frame.winfo_children():
            widget.destroy()
        B = self.current_board
        for i in range(8):
            for j in range(8):
                canvas = self.cell_button(B.grid[i][j], parent_frame)
                canvas.grid(row=i, column=j)
                self.cell_canvases[i][j] = canvas

    def multiplayer(self):
        # simply refresh board frame (keeps menu)
        self.display_board(self.board_frame)

if __name__=="__main__":
    gui = GUI()
    gui.run()
