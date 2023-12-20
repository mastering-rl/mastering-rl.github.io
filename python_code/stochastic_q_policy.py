from policy import StochasticPolicy
from qtable import QTable

""" Make a stochastic policy from a qfunction and a mutli-armed bandit.
    This helps to avoid e.g. loops in policies.
    This policy cannot be updated -- it is only for execution.
"""


class StochasticQPolicy(StochasticPolicy):
    def __init__(self, qfunction, actions, bandit):
        self.qfunction = qfunction
        self.actions = actions
        self.bandit = bandit

    def select_action(self, state):
        return self.bandit.select(state, self.actions, self.qfunction)
