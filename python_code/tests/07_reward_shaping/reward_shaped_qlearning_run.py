from python_code.gridworld_bad_potential_function import GridWorldBadPotentialFunction
from python_code.learners.qlearning import QLearning
from python_code.learners.reward_shaping.gridworld_potential_function import (
    GridWorldPotentialFunction,
)
from python_code.learners.reward_shaping.reward_shaped_qlearning import (
    RewardShapedQLearning,
)
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.multi_armed_bandit.softmax import Softmax
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable
from python_code.tests.plot import Plot

print("==========\nTabular Q-learning: Gridworld\n==========")
mdp = GridWorld(width=10, height=7, goals=[((9, 6), 1), ((8, 6), -1)])
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))
q_learning_rewards = mdp.get_rewards()

print("==========\nReward Shaped Q-learning: Gridworld\n==========")
mdp = GridWorld(width=10, height=7, goals=[((9, 6), 1), ((8, 6), -1)])
qfunction = QTable()
potential = GridWorldPotentialFunction(mdp)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))
shaped_rewards = mdp.get_rewards()

Plot.plot_episode_length(
    ["Tabular Q-learning", "Reward shaping"],
    [q_learning_rewards, shaped_rewards],
)

print("==========\nBad Reward Shaped Q-learning: Gridworld\n==========")
mdp = GridWorld()
qfunction = QTable()
potential = GridWorldBadPotentialFunction(mdp)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
bad_shaped_rewards = mdp.get_rewards()

Plot.plot_episode_length(
    ["Tabular Q-learning 10x7", "Reward shaping 10x7", "Bad reward shaping 4x3"],
    [q_learning_rewards, shaped_rewards, bad_shaped_rewards],
)
