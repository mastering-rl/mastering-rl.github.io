from contested_crossing import ContestedCrossing
from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

print("==========\nTabular Q-learning: Contested crossing\n==========")

"""
mdp = ContestedCrossing()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=50)
print(mdp.q_function_to_string(qfunction))
policy = qfunction.extract_policy(mdp)
mdp.visualise_policy(policy, "Policy plot", mode=0)
mdp.visualise_policy(policy, "Path plot", mode=1)
"""

qfunction = QTable()
mdp = ContestedCrossing()
rewards = [] 
for i in range(200):
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=20)
    policy = qfunction.extract_policy(mdp)
    rewards += mdp.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(["Q-learning"], [rewards], smoothing_factor=0.0)
Plot.plot_cumulative_rewards(["Q-learning"], [rewards])
