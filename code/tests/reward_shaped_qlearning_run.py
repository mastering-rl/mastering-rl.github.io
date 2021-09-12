from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from reward_shaped_qlearning import RewardShapedQLearning
from gridworld_potential_function import GridWorldPotentialFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

print("==========\nTabular Q-learning: Gridworld\n==========")
mdp = GridWorld()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = qfunction.extract_policy(mdp)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))
q_learning_rewards = mdp.get_rewards()

print("==========\nReward Shaped Q-learning: Gridworld\n==========")
mdp = GridWorld()
qfunction = QTable()
potential = GridWorldPotentialFunction(mdp)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute()
policy = qfunction.extract_policy(mdp)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))
reward_shaped_rewards = mdp.get_rewards()

from plot import Plot

Plot.plot_episode_length(
    ["Tabular Q-learning", "Reward shaping"],
    [q_learning_rewards, reward_shaped_rewards],
)
