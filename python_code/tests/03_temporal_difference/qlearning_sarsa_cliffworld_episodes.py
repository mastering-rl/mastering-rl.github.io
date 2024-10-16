import random

from python_code.markov_decision_processes.gridworld import CliffWorld
from python_code.tests.compare_convergence_curves import qlearning_vs_sarsa
from python_code.tests.plot import Plot

mdp_q = CliffWorld()
mdp_s = CliffWorld()

qlearning_vs_sarsa(mdp_q, mdp_s, episodes=1000)
