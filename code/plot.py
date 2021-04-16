import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

class Plot():

    '''
        Calculate the average reward per step over all episodes of a simulation.
    '''
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
    Plot the rewards of several methods.
    '''
    def plotRewards(labels, rewardList):
        x = np.linspace(0, len(rewardList[0][0]), len(rewardList[0][0]))
        index = 0
        linestyles = ['--', '-', ':', '-.']
        for rewards in rewardList:
            y = Plot.getAverageRewards(rewards)
            y_smoothed = gaussian_filter1d(y, sigma=5)
            plt.plot(x, y_smoothed,
                    label = labels[index],
                    linestyle = linestyles[index % len(linestyles)])
            index += 1

        plt.legend()
        plt.show()
