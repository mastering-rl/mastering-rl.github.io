class ModelFreeReinforcementLearner():

    def __init__(self, mdp, bandit, alpha = 0.2, initQValues = None):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.initQValues = initQValues

    def execute(self, episodes = 2000): abstract

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
        if self.initQValues == None:
            qValues = dict()
            for state in self.mdp.getStates():
                for action in self.mdp.getActions():
                    qValues.update({(state, action): 0.0})
            return qValues
        else:
            return self.initQValues

    '''
        Return the Q-values only for this state
    '''
    def getQValues(self, state, qValues):
        return {k[1]:v for (k,v) in qValues.items() if k[0] == state}   

from multi_armed_bandits import EpsilonGreedy

class QLearning(ModelFreeReinforcementLearner):
    def execute(self, episodes = 100):
        qValues = self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()

            while not self.mdp.isTerminal(state):
                actions = self.mdp.getActions(state)
                action = self.bandit.select(actions, self.getQValues(state, qValues))
                (nextState, reward) = self.mdp.execute(state, action)
                newValue = self.update(qValues, state, action, nextState, reward)
                qValues[(state, action)] = newValue
                state = nextState
            
        return qValues

    
    def update(self, qValues, state, action, nextState, reward): 
        (_, maxQValue) = self.getMaxQ(qValues, nextState)
        qValue = qValues[(state, action)]
        return qValue + self.alpha * (reward + self.mdp.discountFactor * maxQValue - qValue)

class SARSA(ModelFreeReinforcementLearner):
    def execute(self, episodes = 100):
        qValues = self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()
            actions = self.mdp.getActions(state)
            action = self.bandit.select(actions, self.getQValues(state, qValues))
            
            while not self.mdp.isTerminal(state):
                (nextState, reward) = self.mdp.execute(state, action)
                actions = self.mdp.getActions(nextState)
                nextAction = self.bandit.select(actions, self.getQValues(nextState, qValues))
                newValue = self.update(qValues, state, action, nextState, nextAction, reward)
                qValues[(state, action)] = newValue
                state = nextState
                action = nextAction
            
        return qValues
    
    def update(self, qValues, state, action, nextState, nextAction, reward): 
        qValue = qValues[(state, action)]
        qValueNext = qValues[(nextState, nextAction)]
        return qValue + self.alpha * (reward + self.mdp.discountFactor * qValueNext - qValue)


if __name__ == "__main__":
    from gridworld import *
    '''
    print("==========\nQ-learning\n==========")
    mdp = GridWorld(discountFactor = 0.9, width = 4, height = 3)
    qFunction = QLearning(mdp, EpsilonGreedy()).execute(episodes = 1000)
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))

    print("=====\nSARSA\n=====")
    mdp = GridWorld(discountFactor = 0.9, width = 4, height = 3)
    qFunction = SARSA(mdp, EpsilonGreedy()).execute(episodes = 1000)
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))

    print("==========\nQ-learning\n==========")
    mdp = CliffWorld()
    qFunction = QLearning(mdp, EpsilonGreedy(epsilon = 0.2)).execute(episodes = 2000)
    print(mdp.qFunctionToString(qFunction))
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.policyToString(policy))
    qLearningRewards = mdp.getRewards()

    print("=====\nSARSA\n=====")
    mdp = CliffWorld()
    qFunction = SARSA(mdp, EpsilonGreedy(epsilon = 0.2)).execute(episodes = 2000)
    print(mdp.qFunctionToString(qFunction))
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.policyToString(policy))
    sarsaRewards = mdp.getRewards()

    from plot import *
    Plot.plotRewardsPerEpisode(["Q-learning", "SARSA"], [qLearningRewards, sarsaRewards])
    '''

    print("==========\nQ-learning\n==========")
    mdp = CliffWorld()
    qFunction = QLearning(mdp, EpsilonGreedy(epsilon = 0.2)).execute(episodes = 2000)
    QLearning(mdp, EpsilonGreedy(epsilon = 0.0), initQValues = qFunction).execute(episodes = 2000)
    qLearningRewards = mdp.getRewards()

    print("=====\nSARSA\n=====")
    mdp = CliffWorld()
    qFunction = SARSA(mdp, EpsilonGreedy(epsilon = 0.2)).execute(episodes = 2000)
    SARSA(mdp, EpsilonGreedy(epsilon = 0.0), initQValues = qFunction).execute(episodes = 2000)
    sarsaRewards = mdp.getRewards()

    from plot import *
    Plot.plotRewardsPerEpisode(["Q-learning", "SARSA"], [qLearningRewards, sarsaRewards])
