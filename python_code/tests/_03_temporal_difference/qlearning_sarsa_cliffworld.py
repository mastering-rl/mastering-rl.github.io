from python_code.learners.qlearning import QLearning
from python_code.learners.sarsa import SARSA
from python_code.markov_decision_processes.gridworld import CliffWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable
from python_code.tests.plot import Plot

## Train using Q-learning

cliffworld = CliffWorld()
cliffworld_image = cliffworld.visualise()

qfunction = QTable()
rewards = QLearning(cliffworld, EpsilonGreedy(epsilon=0.2), qfunction).execute(
    episodes=2000
)

policy = QPolicy(qfunction)
cliffworld.visualise_policy(policy)

## Train using SARSA

cliffworld = CliffWorld()
qfunction = QTable()
rewards = SARSA(cliffworld, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)

policy = QPolicy(qfunction)
cliffworld.visualise_policy(policy)