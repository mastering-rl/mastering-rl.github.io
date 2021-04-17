import math
from gridworld import *
from multi_armed_bandits import EpsilonGreedy

class ModelFreeReinforcementLearner():

    def __init__(self, mdp, bandit):
        self.mdp = mdp
        self.bandit = bandit

    def execute(self, episodes = 100, alpha = 0.1): abstract

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

    '''
        Return the Q-values only for this state
    '''
    def getQValues(self, state, qValues):
        return {k[1]:v for (k,v) in qValues.items() if k[0] == state}   

class QLearning(ModelFreeReinforcementLearner):
    def execute(self, episodes = 100, alpha = 0.1):
        qValues = self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()

            while not self.mdp.isTerminal(state):
                actions = self.mdp.getActions(state)
                action = self.bandit.select(actions, self.getQValues(state, qValues))
                (nextState, reward) = self.mdp.simulate(state, action)
                newValue = self.update(qValues, state, action, nextState, reward, alpha)
                qValues[(state, action)] = newValue
                state = nextState
            
        return qValues

    
    def update(self, qValues, state, action, nextState, reward, alpha): 
        (_, maxQValue) = self.getMaxQ(qValues, nextState)
        qValue = qValues[(state, action)]
        return qValue + alpha * (reward + self.mdp.discountFactor * maxQValue - qValue)

class SARSA(ModelFreeReinforcementLearner):
    def execute(self, episodes = 100, alpha = 0.1):
        qValues = self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()
            actions = self.mdp.getActions(state)
            action = self.bandit.select(actions, self.getQValues(state, qValues))
            
            while not self.mdp.isTerminal(state):
                (nextState, reward) = self.mdp.simulate(state, action)
                actions = self.mdp.getActions(nextState)
                nextAction = self.bandit.select(actions, self.getQValues(nextState, qValues))
                newValue = self.update(qValues, state, action, nextState, nextAction, reward, alpha)
                qValues[(state, action)] = newValue
                state = nextState
                action = nextAction
            
        return qValues
    
    def update(self, qValues, state, action, nextState, nextAction, reward, alpha): 
        qValue = qValues[(state, action)]
        qValueNext = qValues[(nextState, nextAction)]
        return qValue + alpha * (reward + self.mdp.discountFactor * qValueNext - qValue)


mdp = GridWorld(discountFactor = 0.9, width = 4, height = 3)
qFunction = QLearning(mdp, EpsilonGreedy()).execute(episodes = 1000)
policy = mdp.extractPolicyFromQFunction(qFunction)
print(mdp.qFunctionToString(qFunction))
print(mdp.policyToString(policy))
