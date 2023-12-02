from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

print("==========\nTabular Q-learning: Gridworld\n==========")

mdp = GridWorld()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))

policy = qfunction.extract_policy(mdp)
print(mdp.policy_to_string(policy))

episodes = 2000
episodes_per_evaluation = 20
qfunction = QTable()
mdp = GridWorld()
policy = qfunction.extract_policy(mdp)
rewards = mdp.execute_policy(policy, episodes=episodes_per_evaluation)
for _ in range(int(episodes / episodes_per_evaluation)):
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=episodes_per_evaluation)
    policy = qfunction.extract_policy(mdp)
    rewards += mdp.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(
    ["Q-learning"],
    [rewards],
    smoothing_factor=0.0,
    episodes_per_evaluation=episodes_per_evaluation,
)
Plot.plot_cumulative_rewards(
    ["Q-learning"], [rewards], episodes_per_evaluation=episodes_per_evaluation
)
