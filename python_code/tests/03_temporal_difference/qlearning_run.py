from python_code.learners.qlearning import QLearning
from python_code.markov_decision_processes.contested_crossing import ContestedCrossing
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable
from python_code.tests.plot import Plot

gridworld = GridWorld()
qfunction = QTable()
QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=100)
gridworld.visualise_q_function(qfunction, "Q-Function", grid_size=1.5)

policy = QPolicy(qfunction)
gridworld.visualise_policy(policy)


qfunction = QTable()
mdp = GridWorld()
rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=2000)
Plot.plot_cumulative_rewards(["Q-learning"], [rewards])

qfunction = QTable()
mdp = ContestedCrossing()
rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=2000)
Plot.plot_cumulative_rewards(["Q-learning"], [rewards])
