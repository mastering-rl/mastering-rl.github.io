from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

from python_code.learners.sarsa import SARSA
from python_code.markov_decision_processes.gridworld import CliffWorld
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable

mdp = CliffWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)
# print(mdp.q_function_to_string(qfunction))

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))
