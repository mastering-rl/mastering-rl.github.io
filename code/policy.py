class Policy:
    pass


class DeterministicPolicy(Policy):
    def select_action(self, state):
        abstract

    def update(self, state, action):
        abstract


class StochasticPolicy(Policy):
    def select_action(self, state, action):
        abstract

    def update(self, state, action, value):
        abstract
