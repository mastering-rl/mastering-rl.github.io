from python_code.markov_decision_processes.contested_crossing import ContestedCrossing
from python_code.markov_decision_processes.gridworld import CliffWorld, GridWorld
from python_code.tests.compare_convergence_curves import qlearning_vs_nstep
from python_code.tests.plot import Plot

mdp_q = GridWorld()
mdp_s = GridWorld()
# mdp_q = ContestedCrossing()
# mdp_s = ContestedCrossing()
qlearning_vs_nstep(mdp_q, mdp_s, episodes=2000, n=3)
