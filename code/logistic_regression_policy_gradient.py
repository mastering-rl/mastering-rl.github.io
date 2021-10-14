import math
import random

from abstract_policy_gradient import PolicyGradientBase


class LogisticRegressionPolicyGradientBase(PolicyGradientBase):
    """
    Logistic regression based policy gradient agent used to make decision where there are only two possible actions. Our
    goal is to learn parameters to the logistic function that optimises decision-making in the environment. Since the
    output of a logistic regression function is between 0 and 1, it represents the policy of taking an action. 1 minus
    that probability is the probability of taking the other action. Importantly, it is differentiable, so we
    can update the policy using the policy gradient mechanism!
    """

    def __init__(self, mdp, policy, num_params=2, alpha=0.1) -> None:
        super().__init__(mdp=mdp, policy=policy, alpha=alpha)
        self.theta = [
            random.random() for _ in range(num_params)
        ]  # a vector of policy parameters

    def update(self, states, actions, rewards):
        """
        Update our policy parameters according to the gradient descent formula:
            theta <- theta + alpha * gamma^t * G * nabla J(theta)
        G is the total future discounted reward received in the episode:
            G <- gamma^0 * r_{t+1} + ... +  gamma^(T - t) * r_{T+1}
        """
        for t in range(len(states)):
            gradient_log_pi = self.gradient_log_pi(states[t], actions[t])
            # update each parameter
            for i in range(len(self.theta)):
                self.theta[i] += (
                    self.alpha * (self.gamma ** t) * rewards[t] * gradient_log_pi[i]
                )

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
            return [-s_i * self.logistic_function(y) for s_i in state]

    """ Standard logistic function """

    @staticmethod
    def logistic_function(y):
        return 1 / (1 + math.exp(-y))

    """ Compute the dot product between two vectors """

    @staticmethod
    def dot_product(vec1, vec2):
        return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])


if __name__ == "__main__":
    from gridworld import GridWorld
    from gridworld import OneDimensionalGridWorld
    from logistic_regression_policy import LogisticRegressionPolicy

    print("==========\nLogistic Regression Policy Gradient: 1D Gridworld\n==========")
    # make a GridWorld that only has two dimensions
    one_dimensional_gridworld = OneDimensionalGridWorld(
        width=11, initial_state=(5, 0), goals=[((0, 0), -1), ((10, 0), 1)]
    )
    # one_dimensional_gridworld.visualise_as_image()
    policy = LogisticRegressionPolicy(
        actions=[GridWorld.LEFT, GridWorld.RIGHT],
        num_params=2,
        theta=None,
        alpha=0.1,
        gamma=one_dimensional_gridworld.get_discount_factor(),
    )
    pg_agent = LogisticRegressionPolicyGradientBase(
        one_dimensional_gridworld,
        policy,
        num_params=len(
            one_dimensional_gridworld.get_initial_state()
        ),  # need a weight for each part of the state-space
        alpha=0.1,
        gamma=one_dimensional_gridworld.get_discount_factor(),
    )
    pg_agent.execute(episodes=1000)
    one_dimensional_gridworld.visualise_stochastic_policy(policy)
