from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model, load_model

from c4bot import c4types
from c4bot import c4board
from c4bot.agent import zero

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


encoder = zero.ZeroEncoder()

board_input = Input(shape=encoder.shape(), name='board_input')
pb = board_input
for i in range(4):
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

model = load_model('best.h5')

rounds_per_move = 1600
red_agent = zero.ZeroAgent(model, encoder, rounds_per_move=rounds_per_move, c=2.0)
yellow_agent = zero.ZeroAgent(model, encoder, rounds_per_move=rounds_per_move, c=2.0)
collector1 = zero.ZeroExperienceCollector()
collector2 = zero.ZeroExperienceCollector()
red_agent.set_collector(collector1)
yellow_agent.set_collector(collector2)

for i in range(10000):
    if i % 100 == 0:
        print(i)
    simulate_game(red_agent, collector1, yellow_agent, collector2)

experience = zero.ZeroExperienceBuffer.combine_experience([collector1, collector2])
red_agent.train(experience, 0.01, 2048)
