from abstract_policy_gradient import PolicyGradientBase
from mdp import *
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F


class DeepPolicyGradientBase(PolicyGradientBase):
    """
    A policy gradient agent using a neural network to represent the agent's policy. One of the restrictions of the
    logistic regression agent is that it can only make decisions when the size of the action space is two. Neural
    networks allow us to handle higher dimensional action-spaces. Additionally it allows us to represent non-linear
    policies.
    This class uses PyTorch for the neural network framework. See PyTorch documentation: https://pytorch.org/
    """

    def __init__(self, mdp, state_space, action_space, hidden_dim=64, alpha=0.001, gamma=0.95) -> None:
        super().__init__(mdp=mdp, gamma=gamma, alpha=alpha)
        self.mdp = mdp
        self.state_space = state_space
        self.action_space = action_space

        # we want to define our policy network in the instantiation. Use a sequential neural network as follows:
        #   1) First layer takes in the state vector. Therefore it needs to be the size of the number of state features
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
        log_prob = action_distribution.log_prob(
            action)  # we also want to get the log prob of the action for the policy gradient update step
        return action, log_prob

    def get_probabilities(self, state):
        """
        Return the probabilities of each action given the state of the environment. This is used for the stochastic
        policy visualisation tool.
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.policy_network(state)
        # a softmax layer turns action logits into probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()
        # probabilities are in the order: prob_left, prob_right, prob_up, prob_down
        return probabilities

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

            state = self.mdp.get_initial_state()
            episode_reward = 0
            while not self.mdp.is_terminal(state):
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
    from gridworld import GridWorld
    from gif_player import GifPlayer

    print("==========\nDeep Policy Gradient: 2D Gridworld\n==========")
    two_dimensional_gridworld = GridWorld()

    gif_player = GifPlayer("Deep Policy Gradient: 2D Gridworld")

    # two_dimensional_gridworld.visualise_as_image()

    deep_pg_agent = DeepPolicyGradientBase(two_dimensional_gridworld,
                                         state_space=len(two_dimensional_gridworld.get_initial_state()), action_space=4)
    _, _, img = two_dimensional_gridworld.visualise_stochastic_policy_as_image(deep_pg_agent, two_dimensional=True)
    gif_player.add_image(img)

    deep_pg_agent.execute(episodes=10)
    _, _, img = two_dimensional_gridworld.visualise_stochastic_policy_as_image(deep_pg_agent, two_dimensional=True)
    gif_player.add_image(img)

    deep_pg_agent.execute(episodes=100)
    _, _, img = two_dimensional_gridworld.visualise_stochastic_policy_as_image(deep_pg_agent, two_dimensional=True)
    gif_player.add_image(img)

    # deep_pg_agent.execute(episodes=1000)
    # _, _, img = two_dimensional_gridworld.visualise_stochastic_policy_as_image(deep_pg_agent, two_dimensional=True)
    # gif_player.add_image(img)
    #
    # deep_pg_agent.execute(episodes=10000)
    # _, _, img = two_dimensional_gridworld.visualise_stochastic_policy_as_image(deep_pg_agent, two_dimensional=True)
    # gif_player.add_image(img)

    gif_player.show(block=True)
