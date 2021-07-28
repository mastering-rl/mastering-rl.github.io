from abc import ABC, abstractmethod
import random
from mdp import *
import math


class PolicyGradient(ABC):

    @abstractmethod
    def update(self, state, action, reward):
        """
        Update the policy
        """
        raise NotImplementedError

    def act(self, state):
        """
        Select and action based on the policy of the agent
        """
        raise NotImplementedError

    def execute(self, episodes=100):
        """
        Generate and store an entire episode trajectory to use to update the agent
        """
        raise NotImplementedError


class LogisticRegressionPolicyGradient(PolicyGradient):
    """
    Logistic regression based policy gradient agent used to make decision where there are only two possible actions. Our
    goal is to learn parameters to the logistic function that optimises decision-making in the environment. Since the
    output of a logistic regression function is between 0 and 1, it represents the policy of taking an action. 1 minus
    that probability is the probability of taking the other action. Importantly, it is differentiable, so we
    can update the policy using the policy gradient mechanism!
    """

    def __init__(self, mdp, num_params=2, alpha=0.1, gamma=0.95) -> None:
        super().__init__()
        self.mdp = mdp
        self.theta = [random.random() for _ in range(num_params)]  # a vector of policy parameters
        self.alpha = alpha  # learning rate (gradient update step-size)
        self.gamma = gamma  # discount rate

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

    def logistic_function(self, y):
        """
        Standard logistic function which we will use to transform our policy values into probabilties.
        """
        return 1 / (1 + math.exp(-y))

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
    def dot_product(vec1, vec2):
        """
        Function to compute the dot product between two vectors:
            = v1[0] * v2[0] + v1[1] * v2[1] + ... + v1[n-1] * v2[n-1]
        """
        return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])

if __name__ == '__main__':
    from gridworld import OneDimensionalGridWorld

    print("==========\nLogistic Policy Regression: Cliffworld\n==========")
    # make a GridWorld that only has two dimensions
    mdp = OneDimensionalGridWorld(width=11, initialState=(5, 0), goals=[((0, 0), -1), ((10, 0), 1)])
    mdp.visualiseImage()
    pgAgent = LogisticRegressionPolicyGradient(mdp,
                                               num_params=len(mdp.getInitialState()),  # need a weight for each part of the state-space
                                               alpha=0.1,
                                               gamma=0.95)
    pgAgent.execute(episodes=1000)
