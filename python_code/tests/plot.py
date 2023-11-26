import matplotlib.pyplot as plt
import numpy as np


class Plot:

    # Average rewards over a window
    DEFAULT_WINDOW_SIZE = 250

    # Background colour for a plot
    BACKGROUND_COLOUR = '#EAEAEA'

    """
    Calculate the average reward per episode over a set of episodes
    """

    def get_average_step_rewards(runs):
        num_runs, num_episodes = len(runs), len(runs[0])
        max_num_steps = 0
        for run in runs:
            for episode in run:
                if len(episode) > max_num_steps:
                    max_num_steps = len(episode)

        average_reward_episodes = [
            [0 for j in range(max_num_steps)] for i in range(num_episodes)
        ]

        # calculate the average reward for each episode in each run
        for episode in range(num_episodes):
            num_steps = 0
            for run in range(num_runs):
                if len(runs[run][episode]) > num_steps:
                    num_steps = len(runs[run][episode])

            for step in range(num_steps):
                step_rewards = []
                for run in range(num_runs):
                    if step < len(runs[run][episode]):
                        step_rewards += [runs[run][episode][step]]
                average_reward_episodes[episode][step] = np.mean(step_rewards)

        return average_reward_episodes

    """
    Calculate the average reward per step over all episodes of a simulation.
    """

    def get_average_rewards_per_step(rewards):
        average_rewards = []
        # calculate the average reward for each step in an episode
        for step in range(len(rewards[0])):
            sum = 0.0
            for episode in range(len(rewards)):
                sum += rewards[episode][step]
            average_rewards += [sum / len(rewards)]
        return average_rewards

    """
        Calculate the average reward for each episode,
        averaging the last 'window_size' number of episodes
    """

    def get_average_rewards_per_episode(rewards, window_size=DEFAULT_WINDOW_SIZE):
        summed_rewards = []

        for episode in rewards:
            summed_rewards += [sum(episode)]

        average_rewards = []
        for i in range(len(summed_rewards)):
            window = summed_rewards[max(0, i - window_size) : i + 1]
            average_rewards += [sum(window) / len(window)]
        return average_rewards

    """
        Calculate the exponential moving average of a list of rewards
    """

    def get_ema(rewards, smoothing_factor=0.9):
        smoothed_rewards = []
        for reward in rewards:
            if smoothed_rewards == []:
                smoothed_rewards = [reward]
            else:
                smoothed_rewards += [
                    smoothed_rewards[-1] * smoothing_factor
                    + reward * (1 - smoothing_factor)
                ]
        return smoothed_rewards

    """
        Calculate the length of each episode
    """

    def get_episode_length(rewards):
        episode_lengths = []

        # Omit the first episode as it is (usually) random
        for episode in rewards[1:]:
            episode_lengths += [len(episode)]

        return episode_lengths

    """
    Plot the rewards per step of several methods.
    """

    def plot_rewards(labels, reward_list):
        x = np.linspace(0, len(reward_list[0][0]), len(reward_list[0][0]))
        index = 0
        linestyles = ["--", "-", ":", "-."]
        for rewards in reward_list:
            y = Plot.get_average_rewards_per_step(rewards)
            y_smoothed = Plot.get_ema(y)
            plt.plot(
                x,
                y_smoothed,
                label=labels[index],
                linestyle=linestyles[index % len(linestyles)],
            )
            index += 1

        plt.xlabel("Step")
        plt.ylabel("Average reward per step")
        plt.legend()
        plt.show()

    def plot_cumulative_rewards(labels, reward_list, smoothing_factor=0.95, episodes_per_evaluation=1):
        x = np.linspace(0, len(reward_list[0]), len(reward_list[0]))
        index = 0
        linestyles = ["-", "--", ":", "-."]
        for rewards in reward_list:
            y_smoothed = Plot.get_ema(rewards, smoothing_factor=smoothing_factor)
            plt.plot(
                x*episodes_per_evaluation,
                y_smoothed,
                label=labels[index],
                linestyle=linestyles[index % len(linestyles)],
            )
            index += 1

        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.legend()
        plt.gca().set_facecolor(Plot.BACKGROUND_COLOUR)
        plt.grid(color='white', linewidth=1.5)
        plt.show()

    """
    Plot the rewards per episode of several methods.
    """

    def plot_rewards_per_episode(labels, reward_list, window_size=DEFAULT_WINDOW_SIZE):
        index = 0
        linestyles = ["--", "-", ":", "-."]
        for rewards in reward_list:
            y = Plot.get_average_rewards_per_episode(rewards, window_size=window_size)
            x = np.linspace(0, len(y), len(y))
            y_smoothed = Plot.get_ema(y)
            plt.plot(
                x,
                y_smoothed,
                label=labels[index],
                linestyle=linestyles[index % len(linestyles)],
            )
            index += 1

        plt.xlabel("Episode")
        plt.ylabel("Average reward per episode")
        plt.legend()
        plt.show()

    """
    Plot the rewards per episode of several runs of the same method - 
    don't distinguish between lines.
    """

    def plot_multirun_rewards_per_episode(reward_list, method_label):
        index = 0
        for rewards in reward_list:
            y = Plot.get_average_rewards_per_episode(rewards)
            x = np.linspace(0, len(y), len(y))
            y_smoothed = Plot.get_ema(y)
            plt.plot(x, y_smoothed, color="#2222aa")
            index += 1

        plt.xlabel("Episode")
        plt.ylabel("Av reward per ep - {0}".format(method_label))
        plt.show()

    """
    Plot the average length of episode.
    """

    def plot_episode_length(labels, reward_list):
        index = 0
        linestyles = ["--", "-", ":", "-."]
        for rewards in reward_list:
            y = Plot.get_episode_length(rewards)
            x = np.linspace(0, len(y), len(y))
            y_smoothed = Plot.get_ema(y)
            plt.plot(
                x,
                y_smoothed,
                label=labels[index],
                linestyle=linestyles[index % len(linestyles)],
            )
            index += 1

        plt.xlabel("Episode")
        plt.ylabel("Episode length")
        plt.legend()
        plt.show()
