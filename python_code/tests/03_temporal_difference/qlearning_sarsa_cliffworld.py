from python_code.learners.qlearning import QLearning
from python_code.learners.sarsa import SARSA
from python_code.markov_decision_processes.gridworld import CliffWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable
from python_code.tests.plot import Plot

# Train using Q-learning
mdp = CliffWorld()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)

# Exrract the policy
policy = QPolicy(qfunction)

# Execute the policy and get all rewards: 2000 training and 2000 test
mdp.execute_policy(policy, episodes=2000)
q_learning_rewards = mdp.get_rewards()

# Train using SARSA
mdp = CliffWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)

# Execute the policy
policy = QPolicy(qfunction)

mdp.execute_policy(policy, episodes=2000)
sarsa_rewards = mdp.get_rewards()

Plot.plot_rewards_per_episode(
    ["Q-learning", "SARSA"], [q_learning_rewards, sarsa_rewards]
)
