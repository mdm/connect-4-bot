import sys

from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model, load_model

from c4bot import c4types
from c4bot import c4board
from c4bot.agent import zero

def create_new_model():
    encoder = zero.ZeroEncoder()
    board_input = Input(shape=encoder.shape(), name='board_input')
    pb = board_input
    for i in range(20):
        pb = Conv2D(64, (3, 3),
            padding='same',
            data_format='channels_first',
            activation='relu'
        )(pb)

    policy_conv = Conv2D(2, (1, 1),
        data_format='channels_first',
        activation='relu'
    )(pb)
    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(7, activation='softmax')(policy_flat)

    value_conv = Conv2D(1, (1, 1),
        data_format='channels_first',
        activation='relu'
    )(pb)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(256, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh')(value_hidden)

    model = Model(
        inputs=[board_input],
        outputs=[policy_output, value_output]
    )

    return model

def simulate_game(red_agent, red_collector, yellow_agent, yellow_collector):
    game_state = c4board.GameState.new_game()
    agents = {
        c4types.Player.red: red_agent,
        c4types.Player.yellow: yellow_agent
    }

    red_collector.begin_episode()
    yellow_collector.begin_episode()
    while not game_state.is_over():
        next_move = agents[game_state.next_player].select_move(game_state)
        game_state = game_state.apply_move(next_move)

    if game_state.board.is_winning_move(game_state.last_move):
        if game_state.next_player == c4types.Player.red:
            red_collector.complete_episode(-1)
            yellow_collector.complete_episode(1)
        else:
            red_collector.complete_episode(1)
            yellow_collector.complete_episode(-1)
    else:
        red_collector.complete_episode(0)
        yellow_collector.complete_episode(0)

def gain_experience(latest_model, best_model, num_games, rounds_per_move):
    encoder = zero.ZeroEncoder()
    red_agent = zero.ZeroAgent(latest_model, encoder, rounds_per_move=rounds_per_move, c=2.0)
    yellow_agent = zero.ZeroAgent(best_model, encoder, rounds_per_move=rounds_per_move, c=2.0)
    collector1 = zero.ZeroExperienceCollector()
    collector2 = zero.ZeroExperienceCollector()
    red_agent.set_collector(collector1)
    yellow_agent.set_collector(collector2)

    old_percent = 100
    for i in range(num_games):
        percent = int(i / num_games * 100)
        if percent % 10 == 0 and not percent == old_percent:
            print('{}%'.format(percent))
        old_percent = percent
        simulate_game(red_agent, collector1, yellow_agent, collector2)

    experience = zero.ZeroExperienceBuffer.combine_experience([collector1, collector2])
    return experience, red_agent

def evaluate_model(latest_model, best_model, num_games, rounds_per_move):
    encoder = zero.ZeroEncoder()
    bots = {
        c4types.Player.red: zero.ZeroAgent(latest_model, encoder, rounds_per_move=rounds_per_move),
        c4types.Player.yellow: zero.ZeroAgent(best_model, encoder, rounds_per_move=rounds_per_move),
    }
    wins = {
        c4types.Player.red: 0,
        c4types.Player.yellow: 0,
    }
    
    old_percent = 100
    for i in range(num_games):
        percent = int(i / num_games * 100)
        if percent % 10 == 0 and not percent == old_percent:
            print('{}%'.format(percent))
        old_percent = percent
        game = c4board.GameState.new_game()
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        if game.board.is_winning_move(game.last_move):
            wins[game.next_player.other] += 1

    print('Wins red (latest): {}%'.format(wins[c4types.Player.red] / num_games * 100))
    print('Wins yellow (best): {}%'.format(wins[c4types.Player.yellow] / num_games * 100))

    return wins[c4types.Player.red] / num_games


# MAIN
cycle = 1
while True:
    try:
        latest_model = load_model('latest.h5')
    except OSError:
        latest_model = create_new_model()

    try:
        best_model = load_model('best.h5')
    except OSError:
        best_model = create_new_model()

    print('Training cycle {}:'.format(cycle))
    print('Collecting experience...')
    experience, agent = gain_experience(latest_model, best_model, 100, 50)
    print('Training model...')
    agent.train(experience, 0.01, 2048)
    latest_model.save('latest.h5')
    print('Evaluating model...')
    if evaluate_model(latest_model, best_model, 100, 50) > 0.55:
        print('Replacing best model? YES!')
        latest_model.save('best.h5')
    else:
        print('Replacing best model? NO!')
        
    cycle += 1
