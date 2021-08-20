from abc import ABC, abstractmethod
import random
from mdp import *
import math
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

class PolicyGradient(ABC):
    def __init__(self, mdp, gamma, alpha) -> None:
        super().__init__()
        self.alpha = alpha  # learning rate (gradient update step-size)
        self.gamma = gamma  # discount rate
        self.mdp = mdp

    @abstractmethod
    def update(self, states, actions, rewards):
        """
        Update the policy
        """
        raise NotImplementedError

    @abstractmethod
    def act(self, state):
        """
        Select and action based on the policy of the agent
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, episodes=100):
        """
        Generate and store an entire episode trajectory to use to update the agent
        """
        raise NotImplementedError

    def discounted_rewards(self, rewards):
        """
        Generate a list of the discounted future rewards at that step.
            discounted_rewards[0] =  gamma^0 * r_{1} + ... +  gamma^T * r_{T+1}
            discounted_rewards[1] =  gamma^0 * r_{2} + ... +  gamma^(T-1) * r_{T+1}
            ...
            discounted_rewards[t] =  gamma^0 * r_{t+1} + ... +  gamma^(T-t) * r_{T+1}
            ...
            discounted_rewards[T-2] =  gamma^0 * r_{T-1} + ... +  gamma^2 * r_{T+1}
            discounted_rewards[T-1] =  gamma^0 * r_{T} + ... +  gamma^1 * r_{T+1}

        Notice that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
        We can use that pattern to populate the discounted_rewards array.
        """
        T = len(rewards)
        discounted_future_rewards = [0 for _ in range(T)]
        # the final discounted reward is just the reward you get at that step
        discounted_future_rewards[T - 1] = rewards[T - 1]
        for t in reversed(range(0, T - 1)):
            discounted_future_rewards[t] = rewards[t] + discounted_future_rewards[t + 1] * self.gamma
        return discounted_future_rewards


class LogisticRegressionPolicyGradient(PolicyGradient):
    """
    Logistic regression based policy gradient agent used to make decision where there are only two possible actions. Our
    goal is to learn parameters to the logistic function that optimises decision-making in the environment. Since the
    output of a logistic regression function is between 0 and 1, it represents the policy of taking an action. 1 minus
    that probability is the probability of taking the other action. Importantly, it is differentiable, so we
    can update the policy using the policy gradient mechanism!
    """

    def __init__(self, mdp, num_params=2, alpha=0.1, gamma=0.95) -> None:
        super().__init__(mdp=mdp, gamma=gamma, alpha=alpha)
        self.theta = [random.random() for _ in range(num_params)]  # a vector of policy parameters

    def update(self, states, actions, rewards):
        """
        Update our policy parameters according to the gradient descent formula:
            theta <- theta + alpha * gamma^t * G * nabla J(theta)
        G is the total future discounted reward received in the episode:
            G <- gamma^0 * r_{t+1} + ... +  gamma^(T - t) * r_{T+1}
        """
        discounted_future_rewards = self.discounted_rewards(rewards)
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            discounted_future_reward = discounted_future_rewards[t]
            gradient_log_pi = self.gradient_log_pi(state, action)
            # update each parameter
            for i in range(len(self.theta)):
                self.theta[i] += self.alpha * (self.gamma ** t) * discounted_future_reward * gradient_log_pi[i]

    def act(self, state):
        """
        To determine an action, we use our logistic function to generate probabilities given the state. We then use
        these probabilities sample our action stochastically.
        """
        prob_left, prob_right = self.get_probabilities(state)

        # with a probability of prob_left go left, otherwise go right
        if random.random() < prob_left:
            return self.mdp.LEFT
        else:
            return self.mdp.RIGHT

    def execute(self, episodes=100):
        for i in range(episodes):
            # Use these to store the states, actions and rewards for the episode. These form the trajectory which is
            # used to update the agent at the end of the episode.
            actions = []
            states = []
            rewards = []

            state = self.mdp.getInitialState()
            episode_reward = 0
            while not self.mdp.isTerminal(state):
                action = self.act(state)
                next_state, reward = self.mdp.execute(state, action)

                # store the information from that step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                episode_reward += reward

            self.update(states=states, actions=actions, rewards=rewards)

    def get_probabilities(self, state):
        """
        Determines the probability distribution given the state and current policy
        """
        # calculate y as the linearly weight product of the policy parameters (theta) and the state
        y = self.dot_product(state, self.theta)

        # pass y through the logistic regression function to convert it to a probability
        p = self.logistic_function(y)

        return p, 1 - p

    def gradient_log_pi(self, state, action):
        """
        This computes the gradient of the log of the policy (pi) which is needed to get the gradient of the objective
        (J).
        our policy is a logistic regression, using the policy parameters (theta).
                  pi(left|state)  = 1 / (1 + e^(-theta * state))
                  pi(right|state) = 1 / (1 + e^(theta * state))
        When we apply a logarithmic transformation and take the gradient we end up with:
                  grad_log_pi(left|state) = state - state * pi(left|state)
                  grad_log_pi(right|state) = - state * pi(0|state)
        """
        y = self.dot_product(state, self.theta)
        if action == self.mdp.LEFT:
            return [s_i - s_i * self.logistic_function(y) for s_i in state]
        else:
            return [- s_i * self.logistic_function(y) for s_i in state]

    @staticmethod
    def logistic_function(y):
        """
        Standard logistic function which we will use to transform our policy values into probabilties.
        """
        return 1 / (1 + math.exp(-y))

    @staticmethod
    def dot_product(vec1, vec2):
        """
        Function to compute the dot product between two vectors:
            = v1[0] * v2[0] + v1[1] * v2[1] + ... + v1[n-1] * v2[n-1]
        """
        return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])


class DeepPolicyGradient(PolicyGradient):
    """
    A policy gradient agent using a neural network to represent the agent's policy. One of the restrictions of the
    logistic regression agent is that it can only make decisions when the size of the action space is two. Neural
    networks allow us to handle higher dimensional action-spaces. Additionally it allows us to represent non-linear
    policies.
    This class uses PyTorch for the neural network framework. See PyTorch documentation: https://pytorch.org/
    """

    def __init__(self, mdp, state_space, action_space, hidden_dim=64, alpha=0.1, gamma=0.95) -> None:
        super().__init__(mdp=mdp, gamma=gamma, alpha=alpha)
        self.mdp = mdp
        self.state_space = state_space
        self.action_space = action_space

        # we want to define our policy network in the instantiation. Use a sequential neural network as follows:
        #   1) First layer takes in the state vector. Therefore it needs to be the size of the state space
        #   2) We need to add hidden layers as passed in the __init__ function. The hidden layers allows for non-linear
        #      policy representation.
        #   3) We need non-linear activation function between layers. Also important for non-linearity.
        #   4) We need to output a categorical distribution which has the same size as the action-space.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),  # have a non-linear activation function between layers
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space)
        )
        # make optimisers for the policy network. This will be used to update the weights using gradient descent during
        # the update stage.
        self.optimiser = Adam(self.policy_network.parameters(), lr=self.alpha)

    def act(self, state):
        """
        This function allows us to do a forward pass through the network in order to get our action logits which we will
        turn into a categorical distribution. We can then use this distribution to draw an action from the action-space
        """
        action_logits = self.policy_network(state)
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)  # we also want to get the log prob of the action for the policy gradient update step
        return action, log_prob

    def evaluate_actions(self, states, actions):
        action_logits = self.policy_network(states)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions.squeeze(-1))
        return log_prob.view(1, -1)

    def execute(self, episodes=100):
        for i in range(episodes):
            actions = []
            states = []
            rewards = []
            action_log_probs = []

            state = self.mdp.getInitialState()
            episode_reward = 0
            while not self.mdp.isTerminal(state):
                # turn the state into a tensor such that it can be passed into the network, which requires a tensor of
                # floats
                state_tensor = torch.as_tensor(state, dtype=torch.float32)

                action, action_log_prob = self.act(state_tensor)
                # take the action (which is a tensor) and convert it into an action for the mdp
                next_state, reward = self.mdp.execute(state, self.convert_from_tensor_to_action(action))

                # store the information from that step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                action_log_probs.append(action_log_prob)

                state = next_state
                episode_reward += reward

            # print(f"episode reward = {episode_reward}")
            self.update(states=states, actions=actions, rewards=rewards)

    def update(self, states, actions, rewards):
        # convert to tensors such that we can use torch mechanisms to compute the gradient and update the node weights
        # in the network.
        returns = torch.as_tensor(self.discounted_rewards(rewards), dtype=torch.float32)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions)

        action_log_probs = self.evaluate_actions(states, actions)

        loss = -(action_log_probs * returns).mean()
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()  # make a gradient descent step

    def convert_from_tensor_to_action(self, action):
        if action == 0:
            return self.mdp.LEFT
        if action == 1:
            return self.mdp.RIGHT
        if action == 2:
            return self.mdp.UP
        if action == 3:
            return self.mdp.DOWN
        if action == 4:
            return self.mdp.TERMINATE


if __name__ == '__main__':
    from gridworld import OneDimensionalGridWorld, GridWorld

    print("==========\nLogistic Regression Policy Gradient: 1D Gridworld\n==========")
    # make a GridWorld that only has two dimensions
    one_dimensional_gridworld = OneDimensionalGridWorld(width=11, initialState=(5, 0), goals=[((0, 0), -1), ((10, 0), 1)])
    one_dimensional_gridworld.visualiseImage()
    pgAgent = LogisticRegressionPolicyGradient(one_dimensional_gridworld,
                                               num_params=len(one_dimensional_gridworld.getInitialState()),  # need a weight for each part of the state-space
                                               alpha=0.1,
                                               gamma=0.95)
    one_dimensional_gridworld.visualise_policy_probabilities(pgAgent)
    pgAgent.execute(episodes=10)
    one_dimensional_gridworld.visualise_policy_probabilities(pgAgent)
    pgAgent.execute(episodes=100)
    one_dimensional_gridworld.visualise_policy_probabilities(pgAgent)
    pgAgent.execute(episodes=1000)
    one_dimensional_gridworld.visualise_policy_probabilities(pgAgent)
    pgAgent.execute(episodes=10000)
    one_dimensional_gridworld.visualise_policy_probabilities(pgAgent)

    print("==========\nDeep Policy Gradient: 2D Gridworld\n==========")
    two_dimensional_gridworld = GridWorld()
    two_dimensional_gridworld.visualiseImage()
    deepPgAgent = DeepPolicyGradient(two_dimensional_gridworld,
                                         state_space=len(two_dimensional_gridworld.getInitialState()), action_space=4)
    deepPgAgent.execute(episodes=10)
    deepPgAgent.execute(episodes=100)
    deepPgAgent.execute(episodes=1000)
    deepPgAgent.execute(episodes=10000)
