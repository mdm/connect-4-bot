import time
from keras.models import load_model

from c4bot import agent
from c4bot import c4board
from c4bot import c4types
from c4bot.utils import print_board, print_move

def main():
    latest_model = load_model('latest.h5')
    best_model = load_model('best.h5')
    encoder = agent.ZeroEncoder()
    bots = {
        c4types.Player.red: agent.ZeroAgent(latest_model, encoder),
        # c4types.Player.yellow: agent.ZeroAgent(best_model, encoder),
        c4types.Player.yellow: agent.MCTSAgent(1000, 1.5),
    }
    wins = {
        c4types.Player.red: 0,
        c4types.Player.yellow: 0,
    }
    total_games = 10
    for i in range(total_games):
        if i % 100 ==0:
            print(i)
        game = c4board.GameState.new_game()
        while not game.is_over():
            # time.sleep(0.3)
            # print(chr(27) + '[2J')
            print_board(game.board)
            bot_move = bots[game.next_player].select_move(game)
            print_move(game.next_player, bot_move)
            game = game.apply_move(bot_move)
        if game.board.is_winning_move(game.last_move):
            wins[game.next_player.other] += 1
    print('Wins red: {}%'.format(wins[c4types.Player.red] / total_games * 100))
    print('Wins yellow: {}%'.format(wins[c4types.Player.yellow] / total_games * 100))

if __name__ == '__main__':
    main()
