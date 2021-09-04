import math
from qlearning import QLearning

class RewardShaping(QLearning):

    def __init__(self, mdp):
        self.mdp = mdp

    def update(self, q_values, state, action, new_state, reward, alpha):
        (_, max_q_value) = self.getMaxQ(q_values, new_state)
        q_value = q_values[(state, action)]
        return q_value + alpha * (reward + mdp.discountFactor * max_q_value - q_value)

mdp = NavigationMDP(discountFactor=0.9, width = 6, height = 4)
q_learning = QLearning(mdp)
print("qLearning")
print(mdp.qFunctionToString(q_learning.qLearning(episodes = 1000)) + "\n")
