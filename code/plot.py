import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

class Plot():

    # Average rewards over a window
    windowSize = 250

    '''
    Calculate the average reward per step over all episodes of a simulation.
    '''
    def getAverageRewardsPerStep(rewards):
        averageRewards = []
        #calculate the average reward for each step in an episode
        for step in range(len(rewards[0])):
            sum = 0.0
            for episode in range(len(rewards)):
                sum += rewards[episode][step]
            averageRewards += [sum/len(rewards)]
        return averageRewards

    '''
        Calculate the average reward for each episode,
        averaging the last 'windowSize' number of episodes
    '''
    def getAverageRewardsPerEpisode(rewards):
        summedRewards = []
 
        for episode in rewards:
            summedRewards += [sum(episode)]
            
        averageRewards = []
        for i in range(len(summedRewards)):
            window = summedRewards[max(0, i - Plot.windowSize): i + 1]
            averageRewards += [sum(window)/len(window)]
        return averageRewards

    
    '''
    Plot the rewards per step of several methods.
    '''
    def plotRewards(labels, rewardList):
        x = np.linspace(0, len(rewardList[0][0]), len(rewardList[0][0]))
        index = 0
        linestyles = ['--', '-', ':', '-.']
        for rewards in rewardList:
            y = Plot.getAverageRewardsPerStep(rewards)
            y_smoothed = gaussian_filter1d(y, sigma=5)
            plt.plot(x, y_smoothed,
                    label = labels[index],
                    linestyle = linestyles[index % len(linestyles)])
            index += 1

        plt.xlabel("Step")
        plt.ylabel("Average reward per step")
        plt.legend()
        plt.show()

    '''
    Plot the rewards per episode of several methods.
    '''
    def plotRewardsPerEpisode(labels, rewardList):
        index = 0
        linestyles = ['--', '-', ':', '-.']
        for rewards in rewardList:
            y = Plot.getAverageRewardsPerEpisode(rewards)
            x = np.linspace(0, len(y), len(y))
            y_smoothed = gaussian_filter1d(y, sigma=5)
            plt.plot(x, y_smoothed,
                    label = labels[index],
                    linestyle = linestyles[index % len(linestyles)])
            index += 1

        plt.xlabel("Episode")
        plt.ylabel("Average reward per episode")
        plt.legend()
        plt.show()
