from gridworld import GridWorld
from policy_gradient import PolicyGradient
from deep_nn_policy import DeepNeuralNetworkPolicy


gridworld = GridWorld()
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
policy = DeepNeuralNetworkPolicy(state_space, action_space)
PolicyGradient(gridworld, policy).execute(episodes=1000)
gridworld_image = gridworld.visualise_stochastic_policy(policy)

gridworld.visualise_policy(policy)
