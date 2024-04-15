from gridworld import GridWorld
from qlearning import QLearning
from deep_qfunction import DeepQFunction
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

from deep_q_network import DQN

gridworld = GridWorld()
qfunction = DQN(state_space=len(gridworld.get_initial_state()), action_space=5)
# qfunction = DeepQFunction(gridworld, state_space=len(gridworld.get_initial_state()), action_space=5, hidden_dim=16)
QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=100)
gridworld.visualise_q_function_as_image(qfunction)
policy = QPolicy(qfunction)
gridworld.visualise_policy_as_image(policy)
