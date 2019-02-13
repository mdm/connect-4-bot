import time

from c4bot import agent
from c4bot import c4board
from c4bot import c4types
from c4bot.utils import print_board, print_move

def main():
    game = c4board.GameState.new_game()
    bots = {
        c4types.Player.red: agent.MCTSAgent(1000, 1.5),
        c4types.Player.yellow: agent.RandomBot(),
    }
    while not game.is_over():
        # time.sleep(0.3)
        # print(chr(27) + '[2J')
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)

if __name__ == '__main__':
    main()
