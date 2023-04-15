from contested_crossing import ContestedCrossing
from qlearning_vs_sarsa import qlearning_vs_sarsa
from tests.plot import Plot

mdp_q = ContestedCrossing()
mdp_s = ContestedCrossing()

qlearning_vs_sarsa(mdp_q, mdp_s, episodes=2000)

mdp_q = ContestedCrossing()
mdp_s = ContestedCrossing()

qlearning_vs_sarsa(mdp_q, mdp_s, episodes=20000)