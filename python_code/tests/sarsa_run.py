from gridworld import GridWorld
from qtable import QTable
from sarsa import SARSA
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = GridWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(), qfunction).execute()
print(mdp.q_function_to_string(qfunction))

policy = qfunction.extract_policy(mdp)
print(mdp.policy_to_string(policy))
