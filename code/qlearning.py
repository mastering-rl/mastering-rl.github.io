from collections import defaultdict

class QFunction():

    '''
        Update this Q-Function with a new value
    '''
    def update(self, state, action, value): abstract

    '''
        Get a Q value for a given state-action pair
    '''
    def getQValue(self, state, action): abstract

    '''
        Get the best action and its Q-value for this state
    '''
    def getMaxQ(self, state, actions): abstract

class QTable(QFunction):
    def __init__(self, default = 0.0):
        self.qTable = defaultdict(lambda : 0.0)

    def update(self, state, action, value):
        self.qTable[(state, action)] = value

    def getQValue(self, state, action):
        return self.qTable[(state, action)]

    def getMaxQ(self, state, actions):
        argmaxQ = None
        maxQ = float('-inf')
        for action in actions:
            value = self.qTable[(state, action)]
            if maxQ < value:
                argMaxQ = action
                maxQ = value
        return (argmaxQ, maxQ)

class ModelFreeReinforcementLearner():

    # how many episodes to take an average for determining convergence
    length = 30

    def __init__(self, mdp, bandit, alpha = 0.1, convergenceEpsilon = float('-inf'), initQValues = None):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.convergenceEpsilon = convergenceEpsilon
        self.initQValues = initQValues

        self.latestRewards = []
        self.previousAverage = 0.0

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


    '''
        Return true if and only if this has converged, defined as the last self.length episodes
        having an average reward within self.convergenceEpsilon of the previous 30 episodes
    '''
    def isConverged(self, episodeReward):
        converged = False
        if len(self.latestRewards) == self.length:
            average = sum(self.latestRewards) / self.length
            if abs(self.previousAverage - average) < self.convergenceEpsilon:
                converged = True
            self.previousAverage = average
            self.latestRewards = []
        else:
            self.latestRewards += [episodeReward]
        return converged

from multi_armed_bandits import EpsilonGreedy

class QLearning(ModelFreeReinforcementLearner):
    def execute(self, episodes = 100):
        qValues = self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()

            episodeReward = 0 
            while not self.mdp.isTerminal(state):
                actions = self.mdp.getActions(state)
                action = self.bandit.select(actions, self.getQValues(state, qValues))
                (nextState, reward) = self.mdp.execute(state, action)
                newValue = self.update(qValues, state, action, nextState, reward)
                qValues[(state, action)] = newValue
                state = nextState
                episodeReward += reward

            if self.isConverged(episodeReward):
                print("converged at %d episodes" % i)
                break
            
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

            episodeReward = 0
            while not self.mdp.isTerminal(state):
                (nextState, reward) = self.mdp.execute(state, action)
                actions = self.mdp.getActions(nextState)
                nextAction = self.bandit.select(actions, self.getQValues(nextState, qValues))
                newValue = self.update(qValues, state, action, nextState, nextAction, reward)
                qValues[(state, action)] = newValue
                state = nextState
                action = nextAction
                episodeReward += reward

            if self.isConverged(episodeReward):
                break

        return qValues

    def update(self, qValues, state, action, nextState, nextAction, reward): 
        qValue = qValues[(state, action)]
        qValueNext = qValues[(nextState, nextAction)]
        return qValue + self.alpha * (reward + self.mdp.discountFactor * qValueNext - qValue)

class LinearSARSA(ModelFreeReinforcementLearner):
    def __init__(self, mdp, bandit, featureExtractor, alpha = 0.1, convergenceEpsilon = float('-inf'), initQValues = None, weights = None):
        super().__init__(mdp, bandit, alpha = alpha, convergenceEpsilon = convergenceEpsilon, initQValues = initQValues)
        self.featureExtractor = featureExtractor
        self.weights = weights
        
    def execute(self, episodes = 100):
        self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()
            actions = self.mdp.getActions(state)
            action = self.bandit.select(actions, self.getQValues(actions, state))

            episodeReward = 0
            while not self.mdp.isTerminal(state):
                (nextState, reward) = self.mdp.execute(state, action)
                actions = self.mdp.getActions(nextState)
                nextAction = self.bandit.select(actions, self.getQValues(actions, nextState))
                newValue = self.update(state, action, nextState, nextAction, reward)
                state = nextState
                action = nextAction
                episodeReward += reward

            if self.isConverged(episodeReward):
                print("converged at %d episodes" % i)
                break
    
    '''
        Return the Q-value for a state-action pair
    '''
    def getQValue(self, state, action):
        qValue = 0.0
        featureValues = self.featureExtractor.extractFeatures(state, action)
        for i in range(len(featureValues)):
            qValue += featureValues[i] * self.weights[i]
        return qValue
    
    def update(self, state, action, nextState, nextAction, reward):
        qValue = self.getQValue(state, action)
        qValueNext = self.getQValue(nextState, nextAction)
        delta = self.alpha * (reward + self.mdp.discountFactor * qValueNext - qValue)

        # update the weights
        featureValues = self.featureExtractor.extractFeatures(state, action)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (delta * featureValues[i])
        
    def initialiseQFunction(self):
        if self.weights == None:
            self.weights = self.featureExtractor.initialiseWeights()

    '''
        Return the Q-values only for this state
    '''
    def getQValues(self, actions, state):
        qValues = dict()
        for action in actions:
            qValues[action] = self.getQValue(state, action)
        return qValues

    def getLinearFunction(self):
        linearFunction = dict()
        i = 0
        for action in self.mdp.getActions():
            linearFunction[(action, "width")] = self.weights[i]
            linearFunction[(action, "height")] = self.weights[i + 1]
            i += 2
        return linearFunction

    def getQTable(self):
        qValues = dict()
        for state in self.mdp.getStates():
            for action in self.mdp.getActions(state):
                qValues[(state, action)] = self.getQValue(state, action)
        return qValues
    
class FeatureExtractor():
    def extractFeatures(self, state, action): abstract

class GridWorldStateFeatureExtractor(FeatureExtractor):
    def __init__(self, width, height):
        self.width = width
        self.height = height

class GridWorldFeatureExtractor(FeatureExtractor):
    def __init__(self, mdp):
        self.mdp = mdp

    def initialiseWeights(self):
        weights = []
        for action in self.mdp.getActions():
            weights += [0.0, 0.0, 0.0]
        return weights
    
    def extractFeatures(self, state, action):
        goal = (self.mdp.width, self.mdp.height)
        x = 0
        y = 1
        featureValues = []
        for a in self.mdp.getActions():
            if a == action and state != GridWorld.TERMINAL:
                featureValues += [1 - ((goal[x] - state[x]) / goal[x])]
                featureValues += [1 - ((goal[y] - state[y]) / goal[y])]
                featureValues += [1 - ((goal[x] - state[x] + goal[y] - state[y]) / (goal[x] + goal[y]))]
            else:
                featureValues += [0.0, 0.0, 0.0]
        return featureValues

class RewardShapedQLearning(QLearning):
    def __init__(self, mdp, bandit, potential, alpha = 0.1, convergenceEpsilon = float('-inf')):
        super().__init__(mdp, bandit, alpha = alpha, convergenceEpsilon = convergenceEpsilon)
        self.potential = potential
        
    def update(self, qValues, state, action, nextState, reward): 
        (_, maxQValue) = self.getMaxQ(qValues, nextState)
        qValue = qValues[(state, action)]
        statePotential = self.potential.getPotential(state)
        nextStatePotential = self.potential.getPotential(nextState)
        potential = reward + self.mdp.discountFactor *  nextStatePotential - statePotential
        return qValue + self.alpha * (reward + potential + self.mdp.discountFactor * maxQValue - qValue)

class PotentialFunction():
    def getPotential(self, state): abstract

class GridWorldPotentialFunction(PotentialFunction):

    from gridworld import GridWorld
    
    def __init__(self, mdp):
        self.mdp = mdp
        
    def getPotential(self, state):
        if state != GridWorld.TERMINAL:
            goal = (self.mdp.width, self.mdp.height)
            x = 0
            y = 1
            return 1 - ((goal[x] - state[x] + goal[y] - state[y]) / (goal[x] + goal[y]))
        else:
            return 0.0
        

if __name__ == "__main__":
    from gridworld import *

    
    print("==========\nQ-learning\n==========")
    #mdp = GridWorld(discountFactor = 0.9, width = 16, height = 12)
    
    mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])

    import time
    start = time.time_ns()
    qFunction = QLearning(mdp, EpsilonGreedy()).execute(episodes = 100)
    finish = time.time_ns()
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))
    print("Q-learning execution time = %f" % ((finish - start) / 1000000))
    qLearningRewards = mdp.getRewards()
    
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
    qLearningRewards = mdp.getRewards()

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
 

    print("==========\nLinearSarsa\n==========")
    mdp = GridWorld(discountFactor = 0.9, noise=0.1, goals=[((3,2),1)])
    featureExtractor = GridWorldFeatureExtractor(mdp)
    linearSarsa = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor)
    linearSarsa.execute(episodes = 200)
    qFunction = linearSarsa.getQTable()
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))

    
    mdp = GridWorld()
    featureExtractor = GridWorldFeatureExtractor(mdp)
    linearSarsa = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor)
    linearSarsa.execute(episodes = 200)
    qFunction = linearSarsa.getQTable()
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))

    '''
    
    print("==========\nReward shaping\n==========")
    mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
    potential = GridWorldPotentialFunction(mdp)
    start = time.time_ns()
    qFunction = RewardShapedQLearning(mdp, EpsilonGreedy(), potential).execute(episodes = 100)
    finish = time.time_ns()
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))
    print("Reward Shaped Q-learning execution time = %f" % ((finish - start) / 1000000))
    rewardShapedRewards = mdp.getRewards()

    from plot import Plot
    Plot.plotEpisodeLength(["Q-learning", "Reward shaping"], [qLearningRewards, rewardShapedRewards])
