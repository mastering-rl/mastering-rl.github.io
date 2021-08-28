from qtable import QTable

class ModelFreeReinforcementLearner():

    def __init__(self, mdp, bandit, qfunction = QTable(), alpha = 0.1):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction = qfunction

    def execute(self, episodes = 100):

        for i in range(episodes):
            state = self.mdp.getInitialState()
            actions = self.mdp.getActions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            while not self.mdp.isTerminal(state):
                (nextState, reward) = self.mdp.execute(state, action)
                actions = self.mdp.getActions(nextState)
                nextAction = self.bandit.select(nextState, actions, self.qfunction)
                newQValue = self.update(state, action, nextState, nextAction, reward)
                self.qfunction.update(state, action, newQValue)
                state = nextState
                action = nextAction

    '''
        Update a Q-function with a new delta
    '''
    def update(self, state, action, nextState, reward): abstract


from multi_armed_bandits import EpsilonGreedy

class QLearning(ModelFreeReinforcementLearner):
    def update(self, state, action, nextState, nextAction, reward): 
        (_, maxQValue) = self.qfunction.getMaxQ(nextState, self.mdp.getActions(nextState))
        qValue = self.qfunction.getQValue(state, action)
        return qValue + self.alpha * (reward + self.mdp.discountFactor * maxQValue - qValue)
    
class SARSA(ModelFreeReinforcementLearner):
    def update(self, state, action, nextState, nextAction, reward): 
        qValue = self.qfunction.getQValue(state, action)
        qValueNext = self.qfunction.getQValue(nextState, nextAction)
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
    print("==========\nQ-learning: Gridworld\n==========")
    mdp = GridWorld()
    print(mdp.visualise())
    #mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])

    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes = 1000)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    print(mdp.policyToString(policy))
    qLearningRewards = mdp.getRewards()

      
    print("=====\nSARSA: Gridworld\n=====")
    mdp = GridWorld(discountFactor = 0.9, width = 4, height = 3)
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes = 1000)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    print(mdp.policyToString(policy))

    print("==========\nQ-learning: Cliffworld\n==========")
    
    mdp = CliffWorld()
    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(epsilon = 0.2), qfunction).execute(episodes = 2000)
    print(mdp.qFunctionToString(qfunction))
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.policyToString(policy))
    # Execute policy (using epsilon greedy with epsilon = 0.0
    QLearning(mdp, EpsilonGreedy(epsilon = 0.0), qfunction = qfunction).execute(episodes = 2000)
    qLearningRewards = mdp.getRewards()

    print("=====\nSARSA: Cliffworld\n=====")    
    mdp = CliffWorld()
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(epsilon = 0.2), qfunction).execute(episodes = 2000)
    print(mdp.qFunctionToString(qfunction))
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.policyToString(policy))
    #mdp.visualiseQFunction(qFunction, title="SARSA: Cliffworld", showText=True)
    # Execute policy (using epsilon greedy with epsilon = 0.0
    SARSA(mdp, EpsilonGreedy(epsilon = 0.0), qfunction = qfunction).execute(episodes = 2000)
    sarsaRewards = mdp.getRewards()

    from plot import Plot
    Plot.plotRewardsPerEpisode(["Q-learning", "SARSA"], [qLearningRewards, sarsaRewards])

    '''
    print("==========\nLinearSarsa: Gridworld one terminal state\n==========")
    mdp = GridWorld(discountFactor = 0.9, noise=0.1, goals=[((3,2),1)])
    featureExtractor = GridWorldFeatureExtractor(mdp)
    linearSarsa = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor)
    linearSarsa.execute(episodes = 200)
    qFunction = linearSarsa.getQTable()
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    mdp.visualiseQFunction(qFunction, title="LinearSarsa: Gridworld one terminal state", showText=True)
    print(mdp.policyToString(policy))

    #print("==========\nLinearSarsa: Gridworld both terminal states\n==========")
    #mdp = GridWorld()
    #featureExtractor = GridWorldFeatureExtractor(mdp)
    #linearSarsa = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor)
    #linearSarsa.execute(episodes = 200)
    #qFunction = linearSarsa.getQTable()
    #policy = mdp.extractPolicyFromQFunction(qFunction)
    #print(mdp.qFunctionToString(qFunction))
    #print(mdp.policyToString(policy))

    print("=========\nQ-Learning with Reward shaping: Gridworld large\n========")
    mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
    potential = GridWorldPotentialFunction(mdp)
    qFunction = RewardShapedQLearning(mdp, EpsilonGreedy(), potential).execute(episodes = 100)
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))
    rewardShapedRewards = mdp.getRewards()

    mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
    qFunction = QLearning(mdp, EpsilonGreedy()).execute(episodes = 100)
    qLearningRewards = mdp.getRewards()

    from plot import Plot
    Plot.plotEpisodeLength(["Q-learning", "Reward shaping"], [qLearningRewards, rewardShapedRewards])
    '''
