import random
from c4bot.agent.base import Agent

class RandomBot(Agent):
    def select_move(self, game_state):
        candidates = []
        for column in range(7):
            if game_state.is_valid_move(column):
                candidates.append(column)
        return random.choice(candidates)  # TODO: handle completely full board
