from contested_crossing import ContestedCrossing
#from graph_visualisation import GraphVisualisation
from qtable import QTable
from single_agent_mcts import SingleAgentMCTS
from multi_armed_bandit.ucb import UpperConfidenceBounds

mdp = ContestedCrossing()
qfunction = QTable()
root_node = SingleAgentMCTS(mdp, qfunction, UpperConfidenceBounds()).mcts(timeout=0.1)
#mdp.visualise_q_function(qfunction)
print("he")
policy = qfunction.extract_policy(mdp)
mdp.visualise_policy(policy, "MCTS Policy", mode=0)

#gv = GraphVisualisation(max_level=6)
#graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
#graph.view()
