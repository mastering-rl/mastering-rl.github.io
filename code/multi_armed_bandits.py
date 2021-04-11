import random
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

class MultiArmedBandit():


    '''
        Select an action given Q-values for each action.
    '''
    def select(self, action, qValues): abstract

    '''
        Reset a multi-armed bandit to its initial configuration.
    '''
    def reset(self):
        self.__init__()

    '''
        Run a bandit algorithm for a number of episodes, with each
        episode being a set length.
    '''
    def runBandit(self, episodes = 10000, episodeLength = 2000):

        #the actions available
        actions = [0, 1, 2, 3, 4]

        # The probability of receiving a payoff of 1 for each action
        probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]

        rewards = []
        for episode in range(0, episodes):
            self.reset()

            qValues = dict()
            N = dict()
            for action in actions:
                qValues[action] = 0.0
                N[action] = 0

            episodeRewards = []
            for step in range(0, episodeLength):
                #select an action
                action = self.select(actions, qValues)

                r = random.random()
                reward = 0
                if r < probabilities[action]:
                    reward = 5

                episodeRewards += [reward]

                N[action] = N[action] + 1

                qValues[action] = qValues[action] - (qValues[action] / N[action])
                qValues[action] = qValues[action] + reward / N[action]

            rewards += [episodeRewards]

        print(qValues)
        return rewards

    def getAverageRewards(rewards):
        averageRewards = []
        #calculate the average reward for each step in an episode
        for step in range(len(rewards[0])):
            sum = 0.0
            for episode in range(len(rewards)):
                sum += rewards[episode][step]
            averageRewards += [sum/len(rewards)]
        return averageRewards


    '''
    Plot the rewards of a series of multi-armed bandit decisions.
    '''
    def plotRewards(labels, rewardList):
        x = np.linspace(0, len(rewardList[0][0]), len(rewardList[0][0]))
        index = 0
        linestyles = ['--', '-', ':', '-.']
        for rewards in rewardList:
            y = MultiArmedBandit.getAverageRewards(rewards)
            #y_smooth = spline(x, y, x)
            y_smoothed = gaussian_filter1d(y, sigma=5)
            plt.plot(x, y_smoothed,
                    label = labels[index],
                    linestyle = linestyles[index % len(linestyles)])
            index += 1

        plt.legend()
        plt.show()
        

class EpsilonGreedy(MultiArmedBandit):

    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def reset(self):
        None

    def select(self, actions, qValues):
        r = random.random()
        # select a random action with epsilon probability
        if r < self.epsilon:
            return random.choice(actions)
        else:
            maxActions = []
            maxValue = float('-inf')
            for action in actions:
                value = qValues[action]
                if value > maxValue:
                    maxActions = [action]
                    maxValue = value
                elif value == maxValue:
                    maxActions += [action]
                    
            # if there are multiple actions with the highest value
            # choose one randomly
            return random.choice(maxActions)

class EpsilonDecreasing(MultiArmedBandit):

    def __init__(self, epsilon = 0.2, alpha = 0.999):
        self.epsilonGreedyBandit = EpsilonGreedy(epsilon)
        self.initialEpsilon = epsilon
        self.alpha = alpha

    def reset(self):
        self.epsilonGreedyBandit = EpsilonGreedy(self.initialEpsilon)
    
    def select(self, actions, qValues):
        
        result = self.epsilonGreedyBandit.select(actions, qValues)
        self.epsilonGreedyBandit.epsilon *= self.alpha
        return result

class Softmax(MultiArmedBandit):
    
    def __init__(self, tau = 1.0):
        self.tau = tau

    def reset(self):
        None
    
    def select(self, actions, qValues):

        # calculate the denominator for the softmax strategy
        sum = 0.0
        for action in actions:
            sum += math.exp(qValues[action] / self.tau)

        r = random.random()
        cumulativeProbability = 0.0
        result = None
        for action in actions:
            probability = math.exp(qValues[action] / self.tau) / sum
            if r >= cumulativeProbability and r <= cumulativeProbability + probability:
                result = action
            cumulativeProbability += probability

        return result


class UpperConfidenceBounds(MultiArmedBandit):

    def __init__(self):
        self.total = 0
        self.N = dict() #number of times each action has been chosen

    def select(self, actions, qValues):

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
            value = qValues[action] + math.sqrt((2 * math.log(self.total)) / N)
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

def plotEpsilonGreedy():
    epsilon005 = EpsilonGreedy(epsilon = 0.05).runBandit()
    epsilon01 = EpsilonGreedy(epsilon = 0.1).runBandit()
    epsilon02 = EpsilonGreedy(epsilon = 0.2).runBandit()
    epsilon04 = EpsilonGreedy(epsilon = 0.4).runBandit()
    epsilon08 = EpsilonGreedy(epsilon = 0.8).runBandit()
    epsilon10 = EpsilonGreedy(epsilon = 1.0).runBandit()

    MultiArmedBandit.plotRewards(["epsilon= 0.05", "epsilon = 0.1", "epsilon = 0.2", "epsilon = 0.4", "epsilon = 0.8", "epsilon = 1.0"],
                                 [epsilon005, epsilon01, epsilon02, epsilon04, epsilon08, epsilon10])


def plotEpsilonDecreasing():
    alpha09 = EpsilonDecreasing(alpha = 0.9).runBandit()
    alpha099 = EpsilonDecreasing(alpha = 0.99).runBandit()
    alpha0999 = EpsilonDecreasing(alpha = 0.999).runBandit()
    alpha1 = EpsilonDecreasing(alpha = 1.0).runBandit()
    

    MultiArmedBandit.plotRewards(["alpha= 0.9", "alpha = 0.99", "alpha= 0.999", "alpha = 1.0"],
                                 [alpha09, alpha099, alpha0999, alpha1])



def plotSoftmax():
    tau10 = Softmax(tau = 1.0).runBandit()
    tau11 = Softmax(tau = 1.1).runBandit()
    tau15 = Softmax(tau = 1.5).runBandit()
    tau20 = Softmax(tau = 2.0).runBandit()

    MultiArmedBandit.plotRewards(["tau = 1.0", "tau = 1.1", "tau = 1.5", "tau = 2.0"],
                                 [tau10, tau11, tau15, tau20])

def plotComparison():
    epsilonGreedy = EpsilonGreedy().runBandit()
    epsilonDecreasing = EpsilonDecreasing().runBandit()
    softmax = Softmax().runBandit()
    ucb = UpperConfidenceBounds().runBandit()

    MultiArmedBandit.plotRewards(["epsilonGreedy", "epsilonDecreasing", "softmax", "ucb"],
                                 [epsilonGreedy, epsilonDecreasing, softmax, ucb])

#plotEpsilonGreedy()
#plotEpsilonDecreasing()
#plotSoftmax()
plotComparison()




