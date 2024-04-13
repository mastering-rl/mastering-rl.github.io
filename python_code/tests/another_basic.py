import sys
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

from collections import namedtuple, deque

from deep_q_network import DQN
from ale_wrapper import ALEWrapper
from experience_replay_learner import ExperienceReplayLearner
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing

    
# if GPU is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_device(device)
#torch.set_default_device('cuda')

version = "Freeway-ramDeterministic-v4"
policy_name = "Freeway.policy"
#version = "ALE/Frogger-ram-v5"
#policy_name = "Frogger-small.policy"
# version = "ALE/KingKong-ram-v5"
# version = "ALE/Riverraid-ram-v5"
# policy_name = "Riverraid.policy"



if int(sys.argv[1]) == 1:
    extend_existing_policy = True
elif int(sys.argv[1]) == 0:
    extend_existing_policy = False
else:
    print("Need to specify [0,1] whether to extend existing policy")
    sys.exit()

import numpy as np
import time

def main():

    mdp = ALEWrapper(version)

    # Get number of actions from gym action space
    action_space = len(mdp.get_actions())

    # Get the number of state observations
    state_space = len(mdp.get_initial_state())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_space, action_space)
    target_net = DQN(state_space, action_space)

    if extend_existing_policy:
        policy_net.load_state_dict(torch.load(policy_name))
        target_net.load_state_dict(policy_net.state_dict())

        print("loading existing policy " + policy_name)


    learner = ExperienceReplayLearner(mdp, EpsilonDecreasing(), policy_net, target_net)
    start = time.time()
    episode_rewards = learner.execute(episodes=20)
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