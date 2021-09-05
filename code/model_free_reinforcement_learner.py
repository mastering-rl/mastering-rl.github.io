class ModelFreeReinforcementLearner:
    def __init__(self, mdp, bandit, qfunction, alpha=0.1):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction = qfunction

    def execute(self, episodes=100):

        for i in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            while not self.mdp.is_terminal(state):
                (next_state, reward) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.qfunction)
                new_q_value = self.update(
                    state, action, next_state, next_action, reward
                )
                self.qfunction.update(state, action, new_q_value)
                state = next_state
                action = next_action

    """ Update a Q-function with a new delta """

    def update(self, state, action, next_state, reward):
        abstract
