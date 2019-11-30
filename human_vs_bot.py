from keras.models import Model, load_model

from c4bot import agent
from c4bot import c4board
from c4bot import c4types
from c4bot.utils import print_board, print_move

def main():
    model = load_model('best.h5')
    encoder = agent.ZeroEncoder()
    bot = agent.ZeroAgent(model, encoder)
    game = c4board.GameState.new_game()
    while not game.is_over():
        print(chr(27) + '[2J')
        print_board(game.board)
        if game.next_player == c4types.Player.yellow:
            move = int(input('-- '))
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)

if __name__ == '__main__':
    main()
