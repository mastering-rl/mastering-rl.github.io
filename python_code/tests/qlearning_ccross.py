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

episodes = 2000
episodes_per_evaluation = 20
qfunction = QTable()
mdp = ContestedCrossing()
policy = qfunction.extract_policy(mdp)
rewards = mdp.execute_policy(policy, episodes=1, random_on_duplicate=True)
for _ in range(int(episodes / episodes_per_evaluation)):
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=episodes_per_evaluation)
    policy = qfunction.extract_policy(mdp)
    rewards += mdp.execute_policy(policy, episodes=1, random_on_duplicate=True)


Plot.plot_cumulative_rewards(
    ["Q-learning"],
    [rewards],
    smoothing_factor=0.0,
    episodes_per_evaluation=episodes_per_evaluation,
)
Plot.plot_cumulative_rewards(
    ["Q-learning"], [rewards], episodes_per_evaluation=episodes_per_evaluation
)
