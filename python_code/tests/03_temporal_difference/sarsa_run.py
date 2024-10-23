from python_code.learners.sarsa import SARSA
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable

gridworld = GridWorld()
qfunction = QTable()
SARSA(gridworld, EpsilonGreedy(), qfunction).execute(episodes=100)
gridworld.visualise_q_function(qfunction)

policy = QPolicy(qfunction)
gridworld.visualise_policy(policy)