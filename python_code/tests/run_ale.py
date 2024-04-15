import sys
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

from ale_wrapper import ALEWrapper
from q_policy import QPolicy
from deep_q_network import DQN
from deep_nn_policy import DeepNeuralNetworkPolicy

def main():

    # The default environment is Freeway. 
    # Try choosing some others: ALE/Frogger-ram-v5, ALE/KingKong-ram-v5, ALE/Riverraid-ram-v5
    version = "Freeway-ramDeterministic-v4"
    policy_name = "Freeway.policy"

    if len(sys.argv) > 1:
        version = sys.argv[1]
        policy_name = sys.argv[2]

    # if GPU is to be used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    mdp = ALEWrapper(version, render_mode="human")

    # Get number of actions from gym action space
    action_space = len(mdp.get_actions())

    # Get the number of state observations
    state_space = len(mdp.get_initial_state())

    '''
    policy_qfunction = DQN(state_space, action_space)
    policy_qfunction.load(policy_name)
    
    policy = QPolicy(policy_qfunction)
    '''
    policy = DeepNeuralNetworkPolicy(mdp, state_space=state_space, action_space=action_space)
    policy.load(policy_name)

    state = mdp.get_initial_state()
    done = False
    while not done:
        actions = mdp.get_actions()
        action = policy.select_action(state, actions)
        next_state, reward, done = mdp.execute(state, action)

        # Move to the next state
        state = next_state

import cProfile
if __name__ == "__main__":
    main()
    cProfile.run("main()", sort="cumulative")