import random
import numpy as np
from keras.optimizers import SGD

from c4bot import agent

class ZeroEncoder():
    def __init__(self):
        # Planes:
        # 0 -> can be reached immediately
        # 1 - 5 -> needs 1 - 5 moves in column to be reached
        # our pieces
        # opponent pieces
        self.num_planes = 8

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player
      
        for column in range(7):
            height = game_state.board.heights[column]
            for row in range(6):
                slot = game_state.board.get(column, row)
                if slot is not None:
                    if slot == next_player:
                        board_tensor[6][row][column] = 1
                    else:
                        board_tensor[7][row][column] = 1

                if row == height:
                    board_tensor[height][row][column] = 1

        return board_tensor

    def shape(self):
        return self.num_planes, 6, 7

class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}
        for move, p in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(p)
        self.children = {}
        # if parent is None:
        #     print('Priors', priors)

    def moves(self):
        # return self.branches.keys()
        moves = self.branches.keys()
        return random.sample(moves, k=len(moves))

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def get_child(self, move):
        return self.children[move]

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior
        
    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0  # TODO: check if this is ever used

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

class ZeroExperienceCollector:
    def __init__(self):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

class ZeroExperienceBuffer:
    def __init__(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('visit_counts', data=self.visit_counts)
        h5file['experience'].create_dataset('rewards', data=self.rewards)

    @classmethod
    def combine_experience(cls, collectors):
        combined_states = np.concatenate([np.array(c.states) for c in collectors])
        combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
        combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
        return ZeroExperienceBuffer(combined_states, combined_visit_counts, combined_rewards)

    @classmethod
    def load_experience(cls, h5file):
        states = np.array(h5file['experience']['states'])
        visit_counts = np.array(h5file['experience']['visit_counts'])
        rewards = np.array(h5file['experience']['rewards'])
        return ZeroExperienceBuffer(states, visit_counts, rewards)


class ZeroAgent(agent.Agent):
    def __init__(self, model, encoder, rounds_per_move=1600, c=2.0):
        self.model = model
        self.encoder = encoder

        self.collector = None

        self.num_rounds = rounds_per_move
        self.c = c

    def set_collector(self, collector):
        self.collector = collector

    def select_branch(self, node):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        return max(node.moves(), key=score_branch)

    def select_move(self, game_state):
        root = self.create_node(game_state)

        for i in range(self.num_rounds):
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(next_move)

            new_state = node.state.apply_move(next_move)
            child_node = self.create_node(new_state, parent=node)

            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move   
                node = node.parent
                value = -1 * value

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array([root.visit_count(column) for column in range(7)])
            # print(visit_counts)
            self.collector.record_decision(root_state_tensor, visit_counts)

        return max(root.moves(), key=root.visit_count)

    def create_node(self, game_state, move=None, parent=None)        :
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self.model.predict(model_input)
        priors = priors[0]
        value = values[0][0]
        move_priors = {
            column: p
            for column, p in enumerate(priors)
        }
        new_node = ZeroTreeNode(
            game_state,
            value,
            move_priors,
            parent,
            move
        )
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]
        model_input = experience.states
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((num_examples, 1))
        action_target = experience.visit_counts / visit_sums
        value_target = experience.rewards

        self.model.compile(SGD(lr=learning_rate), loss=['categorical_crossentropy', 'mse'])
        self.model.fit(model_input, [action_target, value_target], batch_size=batch_size)
