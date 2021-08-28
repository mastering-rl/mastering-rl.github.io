from abc import ABC, abstractmethod


class PolicyGradientBase(ABC):
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
