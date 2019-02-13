import copy

from c4bot.c4types import Player

class Board():
    def __init__(self):
        self.grid = []
        for _ in range(6):
            self.grid.append([None] * 7)
        self.heights = [0] * 7

    def get(self, x, y):
        if x < 0 or x > 6 or y < 0 or y > 5:
            return None
        return self.grid[y][x]

    def drop_piece(self, player, column):
        row = self.heights[column]
        assert row < 6
        self.grid[row][column] = player
        self.heights[column] += 1
        return (column, row) # TODO: make namedtuple

    def is_full_column(self, column):
        return self.heights[column] == 6

    def _is_winning_move(self, move, factor_x, factor_y):
        connected = 0
        for offset in range(-3, 4):
            if self.get(move[0] + factor_x * offset, move[1] + factor_y * offset) == self.get(*move):
                connected += 1
            else:
                connected = 0
            if connected == 4:
                return True
        return False

    def is_winning_move(self, move):
        if self._is_winning_move(move, 1, 0):
            return True
        if self._is_winning_move(move, 0, 1):
            return True
        if self._is_winning_move(move, 1, 1):
            return True
        if self._is_winning_move(move, 1, -1):
            return True
        return False

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move

    def apply_move(self, column):
        next_board = copy.deepcopy(self.board)
        move = next_board.drop_piece(self.next_player, column)
        return GameState(next_board, self.next_player.other, self, move)

    def is_over(self):
        if self.last_move == None:
            return False
        if self.board.is_winning_move(self.last_move):
            return True
        board_full = True
        for column in range(7):
            if not self.board.is_full_column(column):
                board_full = False
                break
        return board_full
    
    def is_valid_move(self, column):
        if self.is_over():
            return False
        return not self.board.is_full_column(column)

    def legal_moves(self):
        moves = []
        for column in range(7):
            if self.is_valid_move(column):
                moves.append(column)
        return moves

    @classmethod
    def new_game(cls):
        return GameState(Board(), Player.red, None, None)
