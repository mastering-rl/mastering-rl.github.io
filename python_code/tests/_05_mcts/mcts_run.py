from python_code.learners.single_agent_mcts import SingleAgentMCTS
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.ucb import UpperConfidenceBounds
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.qtable import QTable
from python_code.utils.graph_visualisation import GraphVisualisation

gridworld = GridWorld()
qfunction = QTable()
root_node = SingleAgentMCTS(gridworld, qfunction, UpperConfidenceBounds()).mcts(
    timeout=0.03
)
gv = GraphVisualisation(max_level=6)
graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
graph.view()

gridworld.visualise_q_function(qfunction)

policy = QPolicy(qfunction)
gridworld.visualise_policy(policy)
