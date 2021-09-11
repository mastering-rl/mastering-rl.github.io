class ModelFreeReinforcementLearner:
    def __init__(self, mdp, bandit, qfunction, alpha=0.1):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction = qfunction

    def execute(self, episodes=100):

        for _ in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            while not self.mdp.is_terminal(state):
                (next_state, reward) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.qfunction)
                q_value = self.qfunction.get_q_value(state, action)
                next_state_value = self.state_value(next_state, next_action)
                delta = reward + self.mdp.discount_factor * next_state_value - q_value
                self.qfunction.update(state, action, self.alpha * delta)
                state = next_state
                action = next_action

    """ Get the value of a state """

    def state_value(self, state, action):
        abstract
