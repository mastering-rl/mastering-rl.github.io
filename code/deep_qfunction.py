import random

import torch
import torch.nn as nn
from qfunction import QFunction
from torch.optim import Adam


class DeepQFunction(QFunction):
    """
        A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework. See PyTorch documentation: https://pytorch.org/
        """

    def __init__(self, mdp, state_space, action_space, hiddem_dim=64, alpha=0.001) -> None:
        self.mdp = mdp
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha

        # Use a sequential neural network as follows:
        #   1) First layer takes in the state vector.
        #   2) We need to add hidden layers as passed in the __init__ function. The hidden layers allows for non-linear
        #      representations.
        #   3) We need non-linear activation function between layers.
        #   4) We use a multi-headed q-function that has the same number of outputs as the action-space to generate a
        #       q-value for each action. This avoids having to do a pass through the network for each action.
        self.q_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hiddem_dim),
            nn.ReLU(),
            nn.Linear(in_features=hiddem_dim, out_features=hiddem_dim),
            nn.ReLU(),
            nn.Linear(in_features=hiddem_dim, out_features=self.action_space)
        )
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)

        # A two-way mapping from actions to integer IDs for ordinal encoding
        actions = self.mdp.get_actions()
        self.action_to_id = {actions[i]: i for i in range(len(actions))}
        self.id_to_action = {action_id: action for action, action_id in self.action_to_id.items()}

    def update(self, state, action, delta):
        # train the network based on the squared error. This ensures that the loss is positive.
        self.optimiser.zero_grad()  # reset gradients to zero
        (delta ** 2).backward()  # back-propagate the loss through the network
        self.optimiser.step()  # do a gradient descent step with the optimiser

    def get_q_value(self, state, action):
        # convert the state into a tensor
        state = self.encode_state(state)
        q_values = self.q_network(state)

        q_value = q_values[self.action_to_id[action]]  # index q-values by action

        return q_value

    def get_max_q(self, state, actions):
        # convert the state into a tensor
        state = torch.as_tensor(self.encode_state(state), dtype=torch.float32)

        # since we have a multi-headed q-function, we only need to pass through the network once
        q_values = self.q_network(state)
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            value = q_values[self.action_to_id[action]].item()
            if max_q < value:
                arg_max_q = action
                max_q = value
            # If these actions have the same Q-value, randomly choose one
            elif max_q == value:
                arg_max_q = random.choice([arg_max_q, action])
        return (arg_max_q, max_q)

    """
    use this to turn the state into a tensor. It also handles the terminal states, which we need to turn into a number
    representation to pass through the q-network.
    """
    @staticmethod
    def encode_state(state):
        if state == ('terminal', 'terminal'):
            state = (-1, -1)
        return torch.as_tensor(state, dtype=torch.float32)
