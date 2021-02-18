import math
from navigation_mdp import *
from multi_armed_bandits import *

class QLearning():

    def __init__(self, mdp):
        self.mdp = mdp

    def qLearning(self, episodes = 1000, alpha = 0.05, decay = 0.2, epsilon = 0.1):
        qValues = self.initialiseQFunction()

        decay = 0.2 
        for i in range(episodes):
            state = mdp.getInitialState()

            while not mdp.isTerminal(state):
                validActions = mdp.getActions(state)
                action = MultiArmedBandits.epsilonGreedy(validActions, state, qValues)
                (newState, reward) = mdp.simulate(state, action)
                newValue = self.update(qValues, state, action, newState, reward, alpha)
                qValues[(state, action)] = newValue
                state = newState

            alpha = max(0.05, alpha * math.exp(-decay * i))
        return qValues


    def update(self, qValues, state, action, newState, reward, alpha):
        (_, maxQValue) = self.getMaxQ(qValues, newState)
        qValue = qValues[(state, action)]
        return qValue + alpha * (reward + mdp.discountFactor * maxQValue - qValue)

    def getMaxQ(self, qValues, state):
        argmaxQ = None
        maxQ = float('-inf')
        for action in self.mdp.getActions(state):
            value = qValues[(state, action)]
            if maxQ < value:
                argMaxQ = action
                maxQ = value
        return (argmaxQ, maxQ)

    def initialiseQFunction(self):
        qValues = dict()
        for state in self.mdp.getStates():
            for action in self.mdp.getActions():
                qValues.update({(state, action): 0.0})
        return qValues

mdp = NavigationMDP(discountFactor=0.9, width = 6, height = 4)
qLearning = QLearning(mdp)
qFunction = qLearning.qLearning(episodes = 100)
print("qLearning")
policy = mdp.extractPolicyFromQFunction(qFunction)
print(mdp.qFunctionToString(qFunction) + "\n")
print(mdp.policyToString(policy))
