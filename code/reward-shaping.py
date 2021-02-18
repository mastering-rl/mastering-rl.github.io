import math
from qlearning import QLearning

class RewardShaping(QLearning)

    def __init__(self, mdp):
        self.mdp = mdp

    def update(self, qValues, state, action, newState, reward, alpha):
        (_, maxQValue) = self.getMaxQ(qValues, newState)
        qValue = qValues[(state, action)]
        return qValue + alpha * (reward + mdp.discountFactor * maxQValue - qValue)

    def potentialFunction(self, state):
        
mdp = NavigationMDP(discountFactor=0.9, width = 6, height = 4)
qLearning = QLearning(mdp)
print("qLearning")
print(mdp.qFunctionToString(qLearning.qLearning(episodes = 1000)) + "\n")
