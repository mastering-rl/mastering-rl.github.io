from gridworld import GridWorld
from reinforce import REINFORCE
from deep_nn_policy import DeepNeuralNetworkPolicy
from tests.plot import Plot

from python_code.learners.policy_gradient import PolicyGradient
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.policies.deep_nn_policy import DeepNeuralNetworkPolicy

gridworld = GridWorld()
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
policy = DeepNeuralNetworkPolicy(state_space, action_space)
rewards = REINFORCE(gridworld, policy).execute(episodes=2000)
gridworld_image = gridworld.visualise_stochastic_policy(policy)
Plot.plot_cumulative_rewards(["REINFORCE"], [rewards], smoothing_factor=0.8)

gridworld.visualise_policy(policy)
