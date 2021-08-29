from collections import defaultdict

import random
import math

from qtable import QTable

class MultiArmedBandit():

    '''
        Select an action for this state given from a list given a Q-function
    '''
    def select(self, state, actions, qfunction): abstract

    '''
        Reset a multi-armed bandit to its initial configuration.
    '''
    def reset(self):
        self.__init__()

    '''
        Run a bandit algorithm for a number of episodes, with each
        episode being a set length.
    '''
    def runBandit(self, episodes = 20, episodeLength = 1000, drift = True):

        #the actions available
        actions = [0, 1, 2, 3, 4]

        #a dummy state
        state = 1

        rewards = []
        for episode in range(0, episodes):
            self.reset()

            # The probability of receiving a payoff of 1 for each action
            probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]

            N = defaultdict(lambda: 0)
            qtable = QTable()

            episodeRewards = []
            for step in range(0, episodeLength):

                # Halfway through the episode, change the probabilities
                if drift and step == episodeLength / 2:
                    probabilities = [0.5, 0.2, 0.0, 0.3, 0.3]

                #select an action
                action = self.select(state, actions, qtable)

                r = random.random()
                reward = 0
                if r < probabilities[action]:
                    reward = 5

                episodeRewards += [reward]

                N[action] = N[action] + 1
                #newValue = qtable.getQValue(state, action) - (qtable.getQValue(state, action) / N[action]) + (reward / N[action])
                qtable.update(state, action, (reward / N[action]) - (qtable.getQValue(state, action) / N[action]) )
                
            rewards += [episodeRewards]

        return rewards

class EpsilonGreedy(MultiArmedBandit):

    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def reset(self):
        None

    def select(self, state, actions, qfunction):
        r = random.random()
        # select a random action with epsilon probability
        if r < self.epsilon:
            return random.choice(actions)
        else:
            (argMaxQ, _) = qfunction.getMaxQ(state, actions)
            return argMaxQ

class EpsilonDecreasing(MultiArmedBandit):

    def __init__(self, epsilon = 0.2, alpha = 0.999):
        self.epsilonGreedyBandit = EpsilonGreedy(epsilon)
        self.initialEpsilon = epsilon
        self.alpha = alpha

    def reset(self):
        self.epsilonGreedyBandit = EpsilonGreedy(self.initialEpsilon)

    def select(self, state, actions, qfunction):
        result = self.epsilonGreedyBandit.select(state, actions, qfunction)
        self.epsilonGreedyBandit.epsilon *= self.alpha
        return result

class Softmax(MultiArmedBandit):

    def __init__(self, tau = 1.0):
        self.tau = tau

    def reset(self):
        None

    def select(self, state, actions, qfunction):

        # calculate the denominator for the softmax strategy
        sum = 0.0
        for action in actions:
            sum += math.exp(qfunction.getQValue(state, action) / self.tau)

        r = random.random()
        cumulativeProbability = 0.0
        result = None
        for action in actions:
            probability = math.exp(qfunction.getQValue(state, action) / self.tau) / sum
            if r >= cumulativeProbability and r <= cumulativeProbability + probability:
                result = action
            cumulativeProbability += probability

        return result


class UpperConfidenceBounds(MultiArmedBandit):

    def __init__(self):
        self.total = 0
        self.N = dict() #number of times each action has been chosen

    def select(self, state, actions, qfunction):

        # First execute each action one time
        for action in actions:
            if not action in self.N.keys():
                self.N[action] = 1
                self.total += 1
                return action

        maxActions = []
        maxValue = float('-inf')
        for action in actions:
            N = self.N[action]
            value = qfunction.getQValue(state, action) + math.sqrt((2 * math.log(self.total)) / N)
            if value > maxValue:
                maxActions = [action]
                maxValue = value
            elif value == maxValue:
                maxActions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(maxActions)
        self.N[result] = self.N[result] + 1
        self.total += 1
        return result


def plotEpsilonGreedy(drift = False):
    epsilon000 = EpsilonGreedy(epsilon = 0.00).runBandit(drift = drift)
    epsilon005 = EpsilonGreedy(epsilon = 0.05).runBandit(drift = drift)
    epsilon01 = EpsilonGreedy(epsilon = 0.1).runBandit(drift = drift)
    epsilon02 = EpsilonGreedy(epsilon = 0.2).runBandit(drift = drift)
    epsilon04 = EpsilonGreedy(epsilon = 0.4).runBandit(drift = drift)
    epsilon08 = EpsilonGreedy(epsilon = 0.8).runBandit(drift = drift)
    epsilon10 = EpsilonGreedy(epsilon = 1.0).runBandit(drift = drift)

    Plot.plotRewards(["epsilon = 0.0", "epsilon = 0.05", "epsilon = 0.1", "epsilon = 0.2", 
                      "epsilon = 0.4", "epsilon = 0.8", "epsilon = 1.0"],
                     [epsilon000, epsilon005, epsilon01, epsilon02, epsilon04, epsilon08, epsilon10])


def plotEpsilonDecreasing(drift = False):
    alpha09 = EpsilonDecreasing(alpha = 0.9).runBandit(drift = drift)
    alpha099 = EpsilonDecreasing(alpha = 0.99).runBandit(drift = drift)
    alpha0999 = EpsilonDecreasing(alpha = 0.999).runBandit(drift = drift)
    alpha1 = EpsilonDecreasing(alpha = 1.0).runBandit(drift = drift)

    Plot.plotRewards(["alpha = 0.9", "alpha = 0.99", "alpha= 0.999", "alpha = 1.0"],
                     [alpha09, alpha099, alpha0999, alpha1])


def plotSoftmax(drift = False):
    tau10 = Softmax(tau = 1.0).runBandit(drift = drift)
    tau11 = Softmax(tau = 1.1).runBandit(drift = drift)
    tau15 = Softmax(tau = 1.5).runBandit(drift = drift)
    tau20 = Softmax(tau = 2.0).runBandit(drift = drift)

    Plot.plotRewards(["tau = 1.0", "tau = 1.1", "tau = 1.5", "tau = 2.0"],
                     [tau10, tau11, tau15, tau20])

def plotComparison(drift = False):
    epsilonGreedy = EpsilonGreedy(epsilon = 0.1).runBandit(drift = drift)
    epsilonDecreasing = EpsilonDecreasing(alpha = 0.99).runBandit(drift = drift)
    softmax = Softmax(tau = 1.0).runBandit(drift = drift)
    ucb = UpperConfidenceBounds().runBandit(drift = drift)

    Plot.plotRewards(["Epsilon Greedy (epsilon = 0.1)", "Epsilon Decreasing (alpha = 0.99)", "Softmax (tau = 1.0)", "UCB"],
                     [epsilonGreedy, epsilonDecreasing, softmax, ucb])

if __name__ == "__main__":

    from plot import Plot

    plotEpsilonGreedy()
    plotEpsilonDecreasing()
    plotSoftmax(drift = False)
    plotSoftmax(drift = True)
    plotComparison(drift = False)
    plotComparison(drift = True)
