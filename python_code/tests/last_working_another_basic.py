import sys
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

from IPython import display

import torch

from deep_q_network import DQN
from ale_wrapper import ALEWrapper
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from experience_replay_learner import ExperienceReplayLearner
from experience_replay_learner import ReplayMemory


# version = "CartPole-v1"
version = "Freeway-ramDeterministic-v4"
policy_name = "Freeway.policy"
#version = "ALE/Frogger-ram-v5"
#policy_name = "Frogger-small.policy"
# version = "ALE/KingKong-ram-v5"
# version = "ALE/Riverraid-ram-v5"
# policy_name = "Riverraid.policy"

env = ALEWrapper(version)


# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()


if int(sys.argv[1]) == 1:
    extend_existing_policy = True
elif int(sys.argv[1]) == 0:
    extend_existing_policy = False
else:
    print("Need to specify [0,1] whether to extend existing policy")
    sys.exit()

plt.ion()


def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.plot(rewards_t.numpy())
    # Take episode averages and plot them over a window
    window = 25
    if len(rewards_t) >= window:
        means = rewards_t.unfold(0, window, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(window - 1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Get number of actions from gym action space
action_space = len(env.get_actions())
# Get the number of state observations
# state, info = env.get_initial_state()
state = env.get_initial_state()
state_space = len(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_space, action_space).to(device)
target_net = DQN(state_space, action_space).to(device)
target_net.load_state_dict(policy_net.state_dict())

if extend_existing_policy:
    policy_net.load_state_dict(torch.load(policy_name))
    target_net.load_state_dict(policy_net.state_dict())

    print("loading existing policy " + policy_name)


import numpy as np
import time

def main():

    learner = ExperienceReplayLearner(env, EpsilonDecreasing(), policy_net, target_net)
    start = time.time()
    episode_rewards = learner.execute(episodes=30)
    end = time.time()
    #print(
    #    ("\n{:.2f}, {:.2f}, " + str(episode_rewards)).format(
    #    np.mean(episode_rewards), end - start
    #)
    #)
    print(
        ("\n{:.2f}, {:.2f}, ").format(np.mean(episode_rewards), end - start)
    )

    torch.save(policy_net.state_dict(), policy_name)

    #print("Complete")
    sys.exit()
    plot_rewards(episode_rewards, show_result=True)
    plt.ioff()
    plt.show()

    policy_net.load_state_dict(torch.load(policy_name))

    mdp = ALEWrapper(version, render_mode="human")
    bandit = EpsilonGreedy(epsilon=0.00)
    state = mdp.get_initial_state()
    done = False
    while not done:
        actions = mdp.get_actions()
        action = bandit.select(state, actions, policy_net)
        next_state, reward, done = mdp.execute(state, action)

        # Move to the next state
        state = next_state

import cProfile
if __name__ == "__main__":
    main()
    cProfile.run("main()", sort="cumulative")