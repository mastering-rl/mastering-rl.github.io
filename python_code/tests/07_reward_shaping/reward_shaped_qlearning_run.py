from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

from python_code.learners.qlearning import QLearning
from python_code.learners.reward_shaping.gridworld_potential_function import (
    GridWorldPotentialFunction,
)
from python_code.learners.reward_shaping.reward_shaped_qlearning import (
    RewardShapedQLearning,
)
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable

print("==========\nTabular Q-learning: Gridworld\n==========")
mdp = GridWorld(width=15, height=12, goals=[((14, 11), 1), ((13, 11), -1)])
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=200)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))
q_learning_rewards = mdp.get_rewards()

print("==========\nReward Shaped Q-learning: Gridworld\n==========")
mdp = GridWorld(width=15, height=12, goals=[((14, 11), 1), ((13, 11), -1)])
qfunction = QTable()
potential = GridWorldPotentialFunction(mdp)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=200)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))
reward_shaped_rewards = mdp.get_rewards()

from plot import Plot

Plot.plot_episode_length(
    ["Tabular Q-learning", "Reward shaping"],
    [q_learning_rewards, reward_shaped_rewards],
)
