from qfunction import QFunction
import torch
import torch.nn as nn
import random
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F


class DeepQFunction(QFunction):
    """
        A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework. See PyTorch documentation: https://pytorch.org/
        """

    def __init__(self, state_space, action_space, hiddem_dim=64, learning_rate=0.001) -> None:
        # Use a sequential neural network as follows:
        #   1) First layer takes in the state vector.
        #   2) We need to add hidden layers as passed in the __init__ function. The hidden layers allows for non-linear
        #      representations.
        #   3) We need non-linear activation function between layers.
        #   4) We need to output a categorical distribution which has the same size as the action-space to generate a
        #       q-value for each action.
        self.q_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hiddem_dim),
            nn.ReLU(),
            nn.Linear(in_features=hiddem_dim, out_features=hiddem_dim),
            nn.ReLU(),
            nn.Linear(in_features=hiddem_dim, out_features=action_space)
        )
        self.optimiser = Adam(self.q_network.parameters(), lr=learning_rate)

    def update(self, state, action, delta):
        # train the network based on the squared error. This ensures that the loss is positive.
        state = self.encode_state(state)
        q_values = self.q_network(state)
        # q_value = q_values[self.action_to_int(action)]  # index q-values by action
        # q_loss = (q_value - td_backup) ** 2
        (delta**2).backward()
        self.optimiser.step()

    def get_q_value(self, state, action):
        # convert the state into a tensor
        state = self.encode_state(state)
        q_values = self.q_network(state)

        # index q-values by action
        q_value = q_values[self.action_to_int(action)]

        # ensure that we return a float value not a tensor
        return q_value

    def get_max_q(self, state, actions):
        # convert the state into a tensor
        state = self.encode_state(state)

        # since we have a multi-headed q-function, we only need to pass through the network once
        q_values = self.q_network(state)
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            value = q_values[self.action_to_int(action)]
            if max_q < value:
                arg_max_q = action
                max_q = value
            # If these actions have the same Q-value, randomly choose one
            elif max_q == value:
                arg_max_q = random.choice([arg_max_q, action])
        return (arg_max_q, max_q)

    @staticmethod
    def encode_state(state):
        if state == ('terminal', 'terminal'):
            state = (-1,-1)
        return torch.as_tensor(state, dtype=torch.float32)

    @staticmethod
    def action_to_int(action):
        if action == "\u25C4":
            return 0
        if action == "\u25B2":
            return 1
        if action == "\u25BA":
            return 2
        if action == "\u25BC":
            return 3
        if action == "terminate":
            return 4

    @staticmethod
    def int_to_action(action):
        if action == 0:
            return "\u25C4"
        if action == 1:
            return "\u25B2"
        if action == 2:
            return "\u25BA"
        if action == 3:
            return "\u25BC"
        if action == 4:
            return "terminate"
