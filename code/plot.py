import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

class Plot():

    # Average rewards over a window
    window_size = 250

    '''
    Calculate the average reward per step over all episodes of a simulation.
    '''
    def get_average_rewards_per_step(rewards):
        average_rewards = []
        #calculate the average reward for each step in an episode
        for step in range(len(rewards[0])):
            sum = 0.0
            for episode in range(len(rewards)):
                sum += rewards[episode][step]
            average_rewards += [sum/len(rewards)]
        return average_rewards

    '''
        Calculate the average reward for each episode,
        averaging the last 'windowSize' number of episodes
    '''
    def get_average_rewards_per_episode(rewards):
        summed_rewards = []
 
        for episode in rewards:
            summed_rewards += [sum(episode)]
            
        average_rewards = []
        for i in range(len(summed_rewards)):
            window = summed_rewards[max(0, i - Plot.window_size): i + 1]
            average_rewards += [sum(window)/len(window)]
        return average_rewards

    '''
        Calculate the length of each episode
    '''
    def get_episode_length(rewards):
        episode_lengths = []

        # Omit the first episode as it is (usually) random
        for episode in rewards[1:]:
            episode_lengths += [len(episode)]

        return episode_lengths
    
    '''
    Plot the rewards per step of several methods.
    '''
    def plot_rewards(labels, reward_list):
        x = np.linspace(0, len(reward_list[0][0]), len(reward_list[0][0]))
        index = 0
        linestyles = ['--', '-', ':', '-.']
        for rewards in reward_list:
            y = Plot.get_average_rewards_per_step(rewards)
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
    def plot_rewards_per_episode(labels, reward_list):
        index = 0
        linestyles = ['--', '-', ':', '-.']
        for rewards in reward_list:
            y = Plot.get_average_rewards_per_episode(rewards)
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

    '''
    Plot the average length of episode.
    '''
    def plot_episode_length(labels, reward_list):
        index = 0
        linestyles = ['--', '-', ':', '-.']
        for rewards in reward_list:
            y = Plot.get_episode_length(rewards)
            x = np.linspace(0, len(y), len(y))
            y_smoothed = gaussian_filter1d(y, sigma=2)
            plt.plot(x, y_smoothed,
                    label = labels[index],
                    linestyle = linestyles[index % len(linestyles)])
            index += 1

        plt.xlabel("Episode")
        plt.ylabel("Episode length")
        plt.legend()
        plt.show()
