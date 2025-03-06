import torch


class Board:
    # For printing board representations
    CHESS_PIECES = [
        '\u2659', # white pawn
        '\u2658', # white knight
        '\u2657', # white bishop
        '\u2656', # white rook
        '\u2655', # white queen
        '\u2654', # white king
        '\u265F', # black pawn
        '\u265E', # black knight
        '\u265D', # black bishop
        '\u265C', # black rook
        '\u265B', # black queen
        '\u265A', # black king
    ]



    '''
    Basic class functions
    '''



    def __init__(self, board_tensor=None, white_to_move=None, castling_rights=None, last_pawns=None):
        # 12 channels for 6 white piece types and 6 black piece types
        # (white) pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5,
        # (black) pawn=6, kngiht=7, bishop=8, rook=9, queen=10, king=11
        
        # board
        if board_tensor is None:
            self.board = torch.zeros((12, 8, 8)) # create empty board
            self.init_board() # give initial piece configuration
            self.white_to_move = True # white moves first
        else:
            self.board = board_tensor
            self.white_to_move = white_to_move if white_to_move is not None else True

        # castling rights
        if castling_rights is None:
            # Keep track of whether rooks/kings are in their initial positions (to determine castling rights).
            # in order: (white) king, queen rook, king rook, (black) king, queen rook, king rook
            self.castling_rights = {piece: True for piece in [(5, 0, 4), (3, 0, 0), (3, 0, 7), (11, 7, 4), (9, 7, 0), (9, 7, 7)]}
        else:
            # updating castling rights during moves is complicated, especially since rooks share
            # a move-generating function with bishops and queens, so we simply check and update
            # which castling rights are still intact upon each new board
            self.castling_rights = castling_rights
            for piece in self.castling_rights:
                if self.castling_rights[piece] and self.board[piece].item() == 0:
                    self.castling_rights[piece] = False

        # last pawn moves (for en passent)
        if last_pawns is None:
            # If either side's last move was a long pawn move, store the new position of that pawn (for en passent).
            self.last_pawns = {True: None, False: None}
        else:
            # whether the last move was a long pawn move is easy enough to keep track of during moves,
            # so it is updated and passed to the next board
            self.last_pawns = last_pawns if last_pawns is not None else {True: None, False: None}


        
    # initialized board with standard starting position
    def init_board(self):
        self.board[0, 1, :] = 1  # white pawns
        self.board[1, 0, [1, 6]] = 1  # white knights
        self.board[2, 0, [2, 5]] = 1  # white bishops
        self.board[3, 0, [0, 7]] = 1  # white rooks
        self.board[4, 0, 3] = 1  # white queen
        self.board[5, 0, 4] = 1  # white king
        self.board[6, 6, :] = 1  # black pawns
        self.board[7, 7, [1, 6]] = 1  # black knights
        self.board[8, 7, [2, 5]] = 1  # black bishops
        self.board[9, 7, [0, 7]] = 1  # black rooks
        self.board[10, 7, 3] = 1  # black queen
        self.board[11, 7, 4] = 1  # black king


    # string representation of board
    def __repr__(self):
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        # loop through channels and populate board
        for channel, symbol in enumerate(Board.CHESS_PIECES):
            positions = self.board[channel].nonzero(as_tuple=False) # indices of nonzero entires in channel
            for row, col in positions:
                board[row][col] = symbol
        
        # convert board to string
        return '\n'.join(' '.join(row) for row in board[::-1]) # reverse order rows so that white appears on bottom
    
    

    '''
    Functions giving information about the board
    '''



    # Return True if square is on the board, otherwise False
    @staticmethod
    def is_on_board(row, col):
        return 0 <= row and row <= 7 and 0 <= col and col <= 7


    # Return the channel/piece type occupying (row, col), returns None if no piece is present
    def is_occupied(self, row, col):
        channel = self.board[:, row, col].nonzero(as_tuple=True)[0]
        return channel.item() if channel.numel() > 0 else None


    # Return True if target channel corresponds to an enemy piece, otherwise False
    def is_enemy(self, target):
        if self.white_to_move:
            return 6 <= target and target <= 11
        else:
            return 0 <= target and target <= 5
    

    # Return True if a square is threatened by an enemy, otherwise False
    def is_threatened(self, row, col):
        if self.white_to_move:
            enemy_pawn   = 6  # black pawn
            enemy_knight = 7  # black knight
            enemy_bishop = 8  # black bishop
            enemy_rook   = 9  # black rook
            enemy_queen  = 10 # black queen
            enemy_king   = 11 # black king
            pawn_squares = [(1, -1), (1, 1)] # squares where enemy pawns are a threat
        else:
            enemy_pawn   = 0  # white pawn
            enemy_knight = 1  # white knight
            enemy_bishop = 2  # white bishop
            enemy_rook   = 3  # white rook
            enemy_queen  = 4  # white queen
            enemy_king   = 5  # white king
            pawn_squares = [(-1, -1), (-1, 1)] # squares where enemy pawns are a threat
        
        # check enemy pawn threats on appropriate diagonals
        for drow, dcol in pawn_squares:
            r, c = row + drow, col + dcol
            if Board.is_on_board(r, c) and self.is_occupied(r, c) == enemy_pawn:
                return True

        # check for enemy king on adjacent squares
        for drow in [-1, 0, 1]:
            for dcol in [-1, 0, 1]:
                if drow == 0 and dcol == 0: # enemy king guaranteed not to be on the square itself
                    continue
                r, c = row + drow, col + dcol
                if Board.is_on_board(r, c) and self.is_occupied(r, c) == enemy_king:
                    return True
                    
        # check for enemy knights on the 8 (or fewer) L-shaped squares
        knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        for drow, dcol in knight_moves:
            r, c = row + drow, col + dcol
            if Board.is_on_board(r, c) and self.is_occupied(r, c) == enemy_knight:
                return True

        # check sliding moves along the diagonals for enemy bishops or queens
        diagonal_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for drow, dcol in diagonal_dirs:
            r, c = row, col
            while Board.is_on_board(r + drow, c + dcol):
                r += drow
                c += dcol
                target = self.is_occupied(r, c)
                if target is not None:
                    if target in (enemy_bishop, enemy_queen):
                        return True
                    break # regardless of piece encountered, stop checking further in this direction

        # check sliding moves along straight lines for enemy rooks or queens
        straight_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for drow, dcol in straight_dirs:
            r, c = row, col
            while Board.is_on_board(r + drow, c + dcol):
                r += drow
                c += dcol
                target = self.is_occupied(r, c)
                if target is not None:
                    if target in (enemy_rook, enemy_queen):
                        return True
                    break # regardless of piece encountered, stop checking further in this direction

        return False # no threats
    

    # Return True if current player's king is in check, otherwise False
    def is_king_in_check(self):
        channel = 5 if self.white_to_move else 11 # king channel
        pos = self.board[channel].nonzero(as_tuple=True) # king position
        row, col = pos[0].item(), pos[1].item()
        return self.is_threatened(row, col)


    # Return 1 for white win, -1 for black win, 0 for draw;
    # we ignore the 3-fold repetition rule
    def get_result(self):
        white_king_pos = self.board[5].nonzero(as_tuple=True)
        black_king_pos = self.board[11].nonzero(as_tuple=True)

        if self.is_threatened(black_king_pos[0].item(), black_king_pos[1].item()): # if black in check: white wins
            return 1
        elif self.is_threatened(white_king_pos[0].item(), white_king_pos[1].item()): # if white in check: black wins
            return -1
        else:
            return 0 # stalemate


    

    '''
    Functions yielding new boards reachable by different moves
    '''



    # Generate all boards reachable from the current board in a single move
    def generate_next_boards(self):
        yield from self.pawn_moves()
        yield from self.knight_moves()
        yield from self.bishop_moves()
        yield from self.rook_moves()
        yield from self.queen_moves()
        yield from self.king_moves()
        yield from self.castling_moves()


    # Yields new board as long as king doesn't enter check
    def yield_board(self, next_board, last_pawns=None):
        test_board = Board(next_board, self.white_to_move) # board state, assuming we make this move
        if not test_board.is_king_in_check(): # check if the king would enter or be left in check
            yield Board(next_board, not self.white_to_move, self.castling_rights.copy(), last_pawns)


    # Place pawn at (row, col) while handling potential promotion
    def place_pawn(self, row, col, next_board, channel):
        if self.white_to_move:
            back_row = 7
            promotions = range(1, 5)
        else:
            back_row = 0
            promotions = range(7, 11)

        if row == back_row:  # check for promotion
            for promotion in promotions:
                next_board[promotion, row, col] = 1  # place promoted piece
                #yield Board(next_board, not self.white_to_move, self.castling_rights.copy())
                yield from self.yield_board(next_board)
                next_board[promotion, row, col] = 0  # undo promotion to try others
        else:
            next_board[channel, row, col] = 1  # place pawn
            #yield Board(next_board, not self.white_to_move, self.castling_rights.copy())
            yield from self.yield_board(next_board)


    # Generate boards reachable by a pawn move
    def pawn_moves(self):
        if self.white_to_move:
            channel = 0
            drow = 1 # white pawn direction
            start = 1 # white pawn start line
        else:
            channel = 6
            drow = -1 # black pawn direction
            start = 6 # black pawn start line

        # get pawn positions
        positions = self.board[channel].nonzero(as_tuple=False)
        for pos in positions:
            row = pos[0].item()
            col = pos[1].item()

            # long pawn move
            if row == start and self.is_occupied(row+drow, col) is None and self.is_occupied(row+2*drow, col) is None:
                next_board = self.board.clone()
                next_board[channel, row, col] = 0 # lift pawn
                next_board[channel, row+2*drow, col] = 1 # place pawn
                last_pawns = self.last_pawns.copy()
                last_pawns[self.white_to_move] = (row+2*drow, col) # record this long pawn move
                #yield Board(next_board, not self.white_to_move, self.castling_rights.copy(), last_pawns)
                yield from self.yield_board(next_board, last_pawns=last_pawns)
            
            # short pawn move
            if Board.is_on_board(row+drow, col) and self.is_occupied(row+drow, col) is None:
                next_board = self.board.clone()
                next_board[channel, row, col] = 0 # lift pawn
                yield from self.place_pawn(row+drow, col, next_board, channel) # place pawn/handle promotion
            
            # capturing moves
            for dcol in [-1, 1]:
                if Board.is_on_board(row+drow, col+dcol):
                    # normal capture
                    target = self.is_occupied(row+drow, col+dcol)
                    if target is not None and self.is_enemy(target): # if there's an enemy piece
                        next_board = self.board.clone()
                        next_board[channel, row, col] = 0 # lift pawn
                        next_board[target, row+drow, col+dcol] = 0 # capture piece
                        yield from self.place_pawn(row+drow, col+dcol, next_board, channel) # place pawn/handle promotion
                    
                    # en passent
                    if (row, col+dcol) == self.last_pawns[not self.white_to_move]:
                        target = (channel + 6) % 12 # enemy pawn channel (0 -> 6, 6 -> 0)
                        next_board = self.board.clone()
                        next_board[channel, row, col] = 0 # lift pawn
                        next_board[target, row, col+dcol] = 0 # capture enemy pawn
                        next_board[channel, row+drow, col+dcol] = 1 # place pawn (no need to handle promotion since en passent cannot happen on the back ranks)
                        #yield Board(next_board, not self.white_to_move, self.castling_rights.copy())
                        yield from self.yield_board(next_board)


    # Generate boards reachable by a knight move
    def knight_moves(self):
        channel = 1 if self.white_to_move else 7

        # get knight positions
        positions = self.board[channel].nonzero(as_tuple=False)
        for pos in positions:
            row = pos[0].item()
            col = pos[1].item()

            for drow, dcol in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                r, c = row + drow, col + dcol

                if Board.is_on_board(r, c):
                    next_board = self.board.clone()
                    target = self.is_occupied(r, c)

                    # if empty square or capturing an enemy piece
                    if target is None or self.is_enemy(target):
                        next_board[channel, row, col] = 0 # lift knight
                        if target is not None:
                            next_board[target, r, c] = 0 # capture enemy piece
                        next_board[channel, r, c] = 1 # place knight

                        yield from self.yield_board(next_board)
    

    # Model sliding pieces (bishop, rook, queen)
    def sliding_moves(self, channel, directions):
        # get piece positions
        positions = self.board[channel].nonzero(as_tuple=False)
        for pos in positions:
            row = pos[0].item()
            col = pos[1].item()

            # for each direction the piece can slide
            for drow, dcol in directions:
                current_row, current_col = row, col
                # go in direction until end of board or another piece is encountered
                while Board.is_on_board(current_row + drow, current_col + dcol):
                    new_row = current_row + drow
                    new_col = current_col + dcol
                    next_board = self.board.clone()
                    target = self.is_occupied(new_row, new_col)
                    if target is None:
                        next_board[channel, row, col] = 0 # lift piece
                        next_board[channel, new_row, new_col] = 1 # place piece
                        #yield Board(next_board, not self.white_to_move, self.castling_rights.copy())
                        yield from self.yield_board(next_board)
                    else:
                        if self.is_enemy(target):
                            next_board[channel, row, col] = 0 # lift piece
                            next_board[target, new_row, new_col] = 0 # capture piece
                            next_board[channel, new_row, new_col] = 1 # place piece
                            #yield Board(next_board, not self.white_to_move, self.castling_rights.copy())
                            yield from self.yield_board(next_board)
                        break # a piece (enemy or not) was encountered, so stop sliding in this direction
                    current_row, current_col = new_row, new_col # continue sliding in this direction
    

    # Generate boards rechable by a bishop move
    def bishop_moves(self):
        channel = 2 if self.white_to_move else 8
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        yield from self.sliding_moves(channel, directions)


    # Generate boards reachable by a rook move
    def rook_moves(self):
        channel = 3 if self.white_to_move else 9
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        yield from self.sliding_moves(channel, directions)


    # Generate boards rechable by a queen move
    def queen_moves(self):
        channel = 4 if self.white_to_move else 10
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (0, 1), (-1, 0), (0, -1)]
        yield from self.sliding_moves(channel, directions)


    # Generate boards reachable by a king move
    def king_moves(self):
        channel = 5 if self.white_to_move else 11

        # get king position
        pos = self.board[channel].nonzero(as_tuple=True)
        row, col = pos[0].item(), pos[1].item()

        for drow, dcol in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            r, c = row + drow, col + dcol

            if Board.is_on_board(r, c): # ensure move is on board
                next_board = self.board.clone()
                target = self.is_occupied(r, c)

                # if empty square or capturing an enemy piece
                if target is None or self.is_enemy(target):
                    next_board[channel, row, col] = 0 # lift king
                    if target is not None:
                        next_board[target, r, c] = 0 # capture enemy piece
                    next_board[channel, r, c] = 1 # place king in new position

                    yield from self.yield_board(next_board) # ensure king isn't moving into check


    # Generate boards reachable by castling
    def castling_moves(self):
        if self.white_to_move:
            king_channel = 5
            rook_channel = 3
            back_row = 0 # row behind pawn starting line
        else:
            king_channel = 11
            rook_channel = 9
            back_row = 7 # row behind pawn starting line

        # if king hasn't moved and is not in check
        if self.castling_rights.get((king_channel, back_row, 4), False) and not self.is_threatened(back_row, 4):
            # king-side castling: check if the king-side rook hasn't moved
            if self.castling_rights.get((rook_channel, back_row, 7), False):
                if all(self.is_occupied(back_row, col) is None and not self.is_threatened(back_row, col) for col in (5, 6)):
                    next_board = self.board.clone()
                    next_board[king_channel, back_row, 4] = 0  # lift king
                    next_board[rook_channel, back_row, 7] = 0  # lift rook
                    next_board[king_channel, back_row, 6] = 1  # place king
                    next_board[rook_channel, back_row, 5] = 1  # place rook
                    # can yield board directly rather than yielding from self.yield_board
                    # since the conditions for castling guarantee that the king doesn't enter check
                    yield Board(next_board, not self.white_to_move, self.castling_rights.copy())

            # queen-side castling: check if the queen-side rook hasn't moved
            if self.castling_rights.get((rook_channel, back_row, 0), False):
                if all(self.is_occupied(back_row, col) is None and not self.is_threatened(back_row, col) for col in (1, 2, 3)):
                    next_board = self.board.clone()
                    next_board[king_channel, back_row, 4] = 0  # lift king
                    next_board[rook_channel, back_row, 0] = 0  # lift rook
                    next_board[king_channel, back_row, 2] = 1  # place king
                    next_board[rook_channel, back_row, 3] = 1  # place rook
                    # can yield board directly rather than yielding from self.yield_board
                    # since the conditions for castling guarantee that the king doesn't enter check
                    yield Board(next_board, not self.white_to_move, self.castling_rights.copy())