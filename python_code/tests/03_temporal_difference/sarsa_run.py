from python_code.learners.sarsa import SARSA
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable

mdp = GridWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes=100)
print(mdp.q_function_to_string(qfunction))

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))
