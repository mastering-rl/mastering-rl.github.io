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
        delta = self.alpha * (reward + self.mdp.discountFactor * maxQValue - qValue)
        return delta
    
class SARSA(ModelFreeReinforcementLearner):
    def update(self, state, action, nextState, nextAction, reward): 
        qValue = self.qfunction.getQValue(state, action)
        qValueNext = self.qfunction.getQValue(nextState, nextAction)
        delta = self.alpha * (reward + self.mdp.discountFactor * qValueNext - qValue)
        return delta


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

    def numFeatures(self):
        return 3

    def numActions(self):
        return len(self.mdp.getActions())
    
    def extract(self, state, action):
        goal = (self.mdp.width, self.mdp.height)
        x = 0
        y = 1
        featureValues = []
        for a in self.mdp.getActions():
            if a == action and state != GridWorld.TERMINAL:
                featureValues += [1 - ((goal[x] - state[x]) / goal[x]) + 0.01]
                featureValues += [1 - ((goal[y] - state[y]) / goal[y]) + 0.01]
                featureValues += [1 - ((goal[x] - state[x] + goal[y] - state[y]) / (goal[x] + goal[y])) + 0.01]
            else:
                featureValues += [0.0, 0.0, 0.0]
        return featureValues

class RewardShapedQLearning(QLearning):
    def __init__(self, mdp, bandit, potential, qfunction = QTable(), alpha = 0.1):
        super().__init__(mdp, bandit, qfunction = qfunction, alpha = alpha)
        self.potential = potential

    def update(self, state, action, nextState, nextAction, reward): 
        (_, maxQValue) = self.qfunction.getMaxQ(nextState, self.mdp.getActions(nextState))
        qValue = self.qfunction.getQValue(state, action)
        statePotential = self.potential.getPotential(state)
        nextStatePotential = self.potential.getPotential(nextState)
        potential = reward + self.mdp.discountFactor *  nextStatePotential - statePotential
        delta = self.alpha * (reward + potential + self.mdp.discountFactor * maxQValue - qValue)
        return delta


class PotentialFunction():
    def getPotential(self, state): abstract

from gridworld import GridWorld

class GridWorldPotentialFunction(PotentialFunction):

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
    '''
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

    print("==========\nLinearSarsa: Gridworld one terminal state\n==========")
    from linear_qfunction import LinearQFunction
    mdp = GridWorld(discountFactor = 0.9, noise=0.1, goals=[((3,2),1)])
    features = GridWorldFeatureExtractor(mdp)
    qfunction = LinearQFunction(features)
    SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes = 2000)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    #mdp.visualiseQFunction(qFunction, title="LinearSarsa: Gridworld one terminal state", showText=True)
    print(mdp.policyToString(policy))
    
    print("==========\nLinearSarsa: Gridworld both terminal states\n==========")
    mdp = GridWorld()
    featureExtractor = GridWorldFeatureExtractor(mdp)
    linearSarsa = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor)
    linearSarsa.execute(episodes = 200)
    qFunction = linearSarsa.getQTable()
    policy = mdp.extractPolicyFromQFunction(qFunction)
    print(mdp.qFunctionToString(qFunction))
    print(mdp.policyToString(policy))
    '''
    
    print("=========\nQ-Learning with Reward shaping: Gridworld large\n========")
    mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
    #mdp = GridWorld()
    potential = GridWorldPotentialFunction(mdp)
    qfunction = QTable()
    RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes = 500)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    print(mdp.policyToString(policy))
    rewardShapedRewards = mdp.getRewards()

    qfunction = QTable()
    mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
    #mdp = GridWorld()
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes = 500)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    print(mdp.policyToString(policy))    
    qLearningRewards = mdp.getRewards()

    from plot import Plot
    Plot.plotEpisodeLength(["Q-learning", "Reward shaping"], [qLearningRewards, rewardShapedRewards])
