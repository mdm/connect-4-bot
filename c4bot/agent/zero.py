from c4bot import agent

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

    def moves(self):
        return self.branches.keys()

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

class ZeroAgent(agent.Agent):
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
                node = node.get_child(next_move):
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

    def simulate_game(red_agent, red_collector, yellow_agent, yellow_collector):
        game_state = GameState.new_game()
        agents = {
            Player.red: red_agent,
            Player.yellow: yellow_agent
        }

        red_collector.begin_episode()
        yellow_collector.begin_episode()
        while not game_state.is_over():
            next_move = agents[game_state.next_player].select_move(game_state)
            game_state = game_state.apply_move(next_move)

        if game_state.board.is_winning_move(game_state.last_move):
            if game_state.next_player == Player.red:
                red_collector.complete_episode(-1)
                yellow_collector.complete_episode(1)
            else:
                red_collector.complete_episode(1)
                yellow_collector.complete_episode(-1)
        else:
            red_collector.complete_episode(0)
            yellow_collector.complete_episode(0)

    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]
        model_input = experience.states
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((num_examples, 1))
        action_target = experience.visit_counts / visit_sums
        value_target = experience.rewards

        self.model.compile(SGD(lr=learning_rate), loss=['categorical_crossentropy', 'mse'])
        self.model.fit(model_input, [action_target, value_target], batch_size=batch_size)

