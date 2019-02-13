from c4bot import c4types

PIECE_TO_CHAR = {
    None: '.',
    c4types.Player.red: 'O',
    c4types.Player.yellow: 'X',
}

def print_board(board):
    for y in range(6):
        row = ''
        for x in range(7):
            piece = board.get(x, 5 - y)
            row += PIECE_TO_CHAR[piece] + ' '
        print(row)
    print('0 1 2 3 4 5 6')

def print_move(player, move):
    print('{} plays column {}'.format(player, move))
