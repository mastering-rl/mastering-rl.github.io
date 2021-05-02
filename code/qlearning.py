class ModelFreeReinforcementLearner():

    def __init__(self, mdp, bandit, alpha = 0.1, initQValues = None):
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

        latest = []
        length = 30
        previousAverage = 0

        for i in range(episodes):
            state = self.mdp.getInitialState()

            totalReward = 0 
            while not self.mdp.isTerminal(state):
                actions = self.mdp.getActions(state)
                action = self.bandit.select(actions, self.getQValues(state, qValues))
                (nextState, reward) = self.mdp.execute(state, action)
                newValue = self.update(qValues, state, action, nextState, reward)
                qValues[(state, action)] = newValue
                state = nextState
                totalReward += reward


            if len(latest) == length:
                average = sum(latest) / length
                if abs(previousAverage - average) < 0.05:
                    print("%d episodes" % i)
                    break
                previousAverage = average
                latest = []
            else:
                latest += [totalReward]
                
            
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


from collections import defaultdict

class LinearSARSA(ModelFreeReinforcementLearner):
    def __init__(self, mdp, bandit, featureExtractor, alpha = 0.2, initQValues = None, weights = None):
        super().__init__(mdp, bandit, alpha = alpha, initQValues = initQValues)
        self.featureExtractor = featureExtractor
        self.weights = weights
        
    def execute(self, episodes = 100):
        self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()
            actions = self.mdp.getActions(state)
            action = self.bandit.select(actions, self.getQValues(actions, state))

            while not self.mdp.isTerminal(state):
                (nextState, reward) = self.mdp.execute(state, action)
                actions = self.mdp.getActions(nextState)
                nextAction = self.bandit.select(actions, self.getQValues(actions, nextState))
                newValue = self.update(state, action, nextState, nextAction, reward)
                state = nextState
                action = nextAction
            
        return qValues
    
    '''
        Return the Q-value for a state-action pair
    '''
    def getQValue(self, state, action):
        qValue = 0.0
        featureValues = self.featureExtractor.extractFeatures(state, action, self.mdp.getActions())
        for i in range(len(featureValues)):
            qValue += featureValues[i] * self.weights[i]
        print("Q(%s,%s) = %f" % (str(state), str(action), qValue))
        return qValue
    
    def update(self, state, action, nextState, nextAction, reward):
        qValue = self.getQValue(state, action)
        qValueNext = self.getQValue(nextState, action)
        delta = self.alpha * (reward + self.mdp.discountFactor * qValueNext - qValue)
        featureValues = self.featureExtractor.extractFeatures(state, action, self.mdp.getActions())
        print(self.weights)
        print(featureValues)
        print(delta)
        print(qValue)
        print(qValueNext)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (delta * featureValues[i])
        print("\t => " + str(self.weights))
        print()
        
    def initialiseQFunction(self):
        if self.weights == None:
            actions = self.mdp.getActions()
            self.weights = self.featureExtractor.initialiseWeights(actions)

    '''
        Return the Q-values only for this state
    '''
    def getQValues(self, actions, state):
        qValues = dict()
        for action in actions:
            qValues[action] = self.getQValue(state, action)
        return qValues

#class FeatureExtractor():
#    def extractFeatures(self, state, action): abstract

class GridWorldFeatureExtractor():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def initialiseWeights(self, actions):
        weights = []
        for action in actions:
            weights += [0.0, 0.0]
        return weights
    
    def extractFeatures(self, state, action, actions):
        featureValues = []
        for a in actions:
            if a == action and state != GridWorld.TERMINAL:
                featureValues += [self.width - state[0], self.height - state[1]]
            else:
                featureValues += [0.0, 0.0]
        return featureValues
    

if __name__ == "__main__":
    from gridworld import *
    
    print("==========\nQ-learning\n==========")
    #mdp = GridWorld(discountFactor = 0.9, width = 16, height = 12)
    mdp = GridWorld(discountFactor=0.9, width = 40, height = 24, goals = [((35,20), 1), ((35,19), -1)])

    import time
    start = time.time_ns()
    qFunction = QLearning(mdp, EpsilonGreedy()).execute(episodes = 1000)
    finish = time.time_ns()
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))
    print("Q-learning execution time = %f" % ((finish - start) / 1000000))

    '''
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

    print("=====\nSARSA\n=====")
    mdp = CliffWorld()
    qFunction = SARSA(mdp, EpsilonGreedy(epsilon = 0.2)).execute(episodes = 2000)
    print(mdp.qFunctionToString(qFunction))
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.policyToString(policy))

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

    from plot import Plot
    Plot.plotRewardsPerEpisode(["Q-learning", "SARSA"], [qLearningRewards, sarsaRewards])

    '''

    '''
    print("==========\nLinearSarsa\n==========")
    width = 4
    height = 3
    mdp = GridWorld(discountFactor = 0.9, width = width, height = height)
    featureExtractor = GridWorldFeatureExtractor(width = width, height = height)
    qFunction = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor).execute(episodes = 1000)
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))
    '''
