import tkinter as tk
from tkinter import Button as btn
from tkinter import messagebox as msg
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
        self.turn='w'
        self.grid = [[None for _ in range(8)] for _ in range(8)]
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

    def move(self, from_pos:tuple, to_pos:tuple):
        from_col = ord(from_pos[0]) - 97
        from_row = 8 - from_pos[1]
        to_col = ord(to_pos[0]) - 97
        to_row = 8 - to_pos[1]
        self.grid[to_row][to_col].piece = self.grid[from_row][from_col].piece
        self.grid[from_row][from_col].piece = "e"

    def show_moves(self, cell:cell):
        reachable_cells = []
        if cell.piece == "e":
            return reachable_cells
        col = ord(cell.pos[0]) - 97
        row = 8 - cell.pos[1]

        if cell.piece.lower() == 'p':
            direction = -1 if cell.piece.isupper() else 1
            start_row = 6 if cell.piece.isupper() else 1
            if 0 <= row + direction < 8 and self.grid[row + direction][col].piece == "e":
                reachable_cells.append(self.grid[row + direction][col])
                if row == start_row and self.grid[row + 2*direction][col].piece == "e":
                    reachable_cells.append(self.grid[row + 2*direction][col])
            for dc in [-1,1]:
                if 0 <= col+dc < 8 and 0 <= row+direction < 8:
                    target = self.grid[row+direction][col+dc]
                    if target.piece != "e" and target.piece.islower() != cell.piece.islower():
                        reachable_cells.append(target)

        # Rook
        if cell.piece.lower() == 'r':
            directions = [(1,0),(-1,0),(0,1),(0,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    target = self.grid[r][c]
                    if target.piece == "e":
                        reachable_cells.append(target)
                    else:
                        if target.piece.islower() != cell.piece.islower():
                            reachable_cells.append(target)
                        break
                    r += dr; c += dc

        # Knight
        if cell.piece.lower() == 'n':
            knight_moves = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
            for dr, dc in knight_moves:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    target = self.grid[r][c]
                    if target.piece == "e" or target.piece.islower() != cell.piece.islower():
                        reachable_cells.append(target)

        # Bishop
        if cell.piece.lower() == 'b':
            directions = [(1,1),(1,-1),(-1,1),(-1,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    target = self.grid[r][c]
                    if target.piece == "e":
                        reachable_cells.append(target)
                    else:
                        if target.piece.islower() != cell.piece.islower():
                            reachable_cells.append(target)
                        break
                    r += dr; c += dc

        # Queen
        if cell.piece.lower() == 'q':
            directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    target = self.grid[r][c]
                    if target.piece == "e":
                        reachable_cells.append(target)
                    else:
                        if target.piece.islower() != cell.piece.islower():
                            reachable_cells.append(target)
                        break
                    r += dr; c += dc

        # King
        if cell.piece.lower() == 'k':
            directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    target = self.grid[r][c]
                    if target.piece == "e" or target.piece.islower() != cell.piece.islower():
                        reachable_cells.append(target)
        return reachable_cells

    def to_FEN(self):
        fen_rows = []
        for row in self.grid:
            empty_count = 0
            fen_row = ""
            for c in row:
                if c.piece=="e": empty_count+=1
                else:
                    if empty_count>0: fen_row+=str(empty_count); empty_count=0
                    fen_row+=c.piece
            if empty_count>0: fen_row+=str(empty_count)
            fen_rows.append(fen_row)
        return "/".join(fen_rows) + " w KQkq - 0 1"

class GUI:
    def __init__(self):
        self.selected_piece = None
        self.reachable_cells = []
        self.cell_canvases = [[None for _ in range(8)] for _ in range(8)]
        self.current_board = Board()
        self.FEN = self.current_board.to_FEN()

    def run(self):
        self.root = tk.Tk()
        self.root.title("Chess Board")
        btn(self.root, text="Single Player", command=lambda: msg.showinfo("Info","Single Player under development")).pack(pady=10)
        btn(self.root, text="Multiplayer", command=self.multiplayer).pack(pady=10)
        self.display_board()
        self.root.mainloop()

    def cell_button(self, cell:cell, parent_frame:tk.Frame):
        color = "#F0D9B5" if cell.colour=="w" else "#B67C4D"
        canvas = tk.Canvas(parent_frame,width=55,height=55,bg=color,highlightthickness=0)
        self.update_canvas(cell, canvas)
        canvas.bind("<Button-1>", lambda e,c=cell,cv=canvas: self.show_moves(c,cv))
        return canvas

    def update_canvas(self, cell, canvas=None):
        r = 8 - cell.pos[1]; c = ord(cell.pos[0])-97
        if canvas is None: canvas = self.cell_canvases[r][c]
        canvas.delete("all")
        color = "#F0D9B5" if cell.colour=="w" else "#B67C4D"
        canvas.configure(bg=color)
        if cell.piece!="e":
            path="data/images/"
            imgs={
                'P':'white-pawn.png','R':'white-rook.png','N':'white-knight.png','B':'white-bishop.png',
                'Q':'white-queen.png','K':'white-king.png','p':'black-pawn.png','r':'black-rook.png',
                'n':'black-knight.png','b':'black-bishop.png','q':'black-queen.png','k':'black-king.png'
            }
            img = Image.open(path+imgs[cell.piece]).convert("RGBA").resize((55,55))
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(27,27,image=photo)
            canvas.image = photo

    def show_moves(self, clicked_cell:cell, clicked_canvas:tk.Canvas):
        # Clear highlights
        for r in range(8):
            for c in range(8):
                self.cell_canvases[r][c].delete("highlight")

        # Move if valid
        if self.selected_piece and any(clicked_cell.pos==c.pos for c in self.reachable_cells):
            if (self.selected_piece.piece.isupper() and self.current_board.turn=='w') or \
               (self.selected_piece.piece.islower() and self.current_board.turn=='b'):
                self.current_board.move(self.selected_piece.pos, clicked_cell.pos)
                self.current_board.turn = 'b' if self.current_board.turn=='w' else 'w'
                self.FEN = self.current_board.to_FEN()
                # Update only source & target squares
                self.update_canvas(self.selected_piece)
                self.update_canvas(clicked_cell)
                self.selected_piece = None
                self.reachable_cells = []
            return

        # Select new piece
        if (clicked_cell.piece.isupper() and self.current_board.turn=='w') or \
           (clicked_cell.piece.islower() and self.current_board.turn=='b'):
            self.selected_piece = clicked_cell
            self.reachable_cells = self.current_board.show_moves(clicked_cell)
            clicked_canvas.create_rectangle(2,2,53,53, outline="#A80303", width=3, tags="highlight")
            for c in self.reachable_cells:
                r_idx = 8 - c.pos[1]
                c_idx = ord(c.pos[0]) - ord('a')
                cv = self.cell_canvases[r_idx][c_idx]
                cv.create_oval(22,22,32,32, fill="blue", tags="highlight")
        else:
            self.selected_piece = None
            self.reachable_cells = []

    def display_board(self, parent_frame=None):
        if parent_frame is None: parent_frame = self.root
        self.board_frame = parent_frame
        for widget in parent_frame.winfo_children():
            widget.destroy()
        B = self.current_board
        for i in range(8):
            for j in range(8):
                canvas = self.cell_button(B.grid[i][j], parent_frame)
                canvas.grid(row=i,column=j)
                self.cell_canvases[i][j] = canvas

    def multiplayer(self):
        self.display_board()

if __name__=="__main__":
    gui = GUI()
    gui.run()
