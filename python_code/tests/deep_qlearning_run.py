from gridworld import GridWorld
from qlearning import QLearning
from deep_qfunction import DeepQFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

gridworld = GridWorld()
qfunction = DeepQFunction(gridworld, state_space=len(gridworld.get_initial_state()), action_space=5, hidden_dim=16)
QLearning(gridworld, EpsilonGreedy(), qfunction, alpha=1.0).execute(episodes=100)
gridworld.visualise_q_function_as_image(qfunction)
policy = qfunction.extract_policy(gridworld)
gridworld.visualise_policy_as_image(policy)
