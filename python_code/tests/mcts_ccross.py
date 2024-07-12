from multi_armed_bandit.ucb import UpperConfidenceBounds

from python_code.learners.single_agent_mcts import SingleAgentMCTS
from python_code.markov_decision_processes.contested_crossing import ContestedCrossing
from python_code.policies.q_policy import QPolicy

# from graph_visualisation import GraphVisualisation
from python_code.qfunctions.qtable import QTable

mdp = ContestedCrossing()
qfunction = QTable()
root_node = SingleAgentMCTS(mdp, qfunction, UpperConfidenceBounds()).mcts(timeout=0.1)
# mdp.visualise_q_function(qfunction)
policy = QPolicy(qfunction)
mdp.visualise_policy(policy, "MCTS Policy", mode=0)
mdp.visualise_policy(policy, "MCTS Policy", mode=1)

# gv = GraphVisualisation(max_level=6)
# graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
# graph.view()
