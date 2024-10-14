from tests.compare_convergence_curves import qlearning_vs_sarsa
from tests.plot import Plot

from python_code.markov_decision_processes.contested_crossing import ContestedCrossing

mdp_q = ContestedCrossing()
mdp_s = ContestedCrossing()

qlearning_vs_sarsa(mdp_q, mdp_s, episodes=2000)

mdp_q = ContestedCrossing()
mdp_s = ContestedCrossing()

qlearning_vs_sarsa(mdp_q, mdp_s, episodes=20000)
