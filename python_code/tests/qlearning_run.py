from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

print("==========\nTabular Q-learning: Gridworld\n==========")

mdp = GridWorld()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))

policy = qfunction.extract_policy(mdp)
print(mdp.policy_to_string(policy))
