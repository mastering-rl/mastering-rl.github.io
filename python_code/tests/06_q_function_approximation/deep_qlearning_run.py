from python_code.learners.qlearning import QLearning
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.deep_q_function import DeepQFunction

gridworld = GridWorld()
action_space = len(gridworld.get_actions())
state_space = len(gridworld.get_initial_state())
qfunction = DeepQFunction(state_space, action_space)
rewards = QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=300)
policy = QPolicy(qfunction)
gridworld.visualise_q_function(qfunction)
gridworld.visualise_policy_as_image(policy)
