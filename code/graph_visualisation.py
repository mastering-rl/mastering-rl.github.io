from graphviz import Digraph, Graph

class GraphVisualisation():

    def __init__(self, max_level = float('inf')):
        self.max_level = max_level

    def single_agent_mcts_to_graph(self, state_node, filename='mcts'):
        g = Digraph('G', filename=filename, format='png')
        self.state_node_to_graph(g, state_node, level = 0)
        return g

    def state_node_id(self, state_node):
        return "V(" + str(state_node.state) + "." + str(state_node.id) + ") = " + str(round(state_node.get_value(), 3)) +\
               "\\nN = " + str(state_node.visits)

    def environment_node_id(self, environment_node):
        return "V(" + str(environment_node.id) + ") = " + str(round(environment_node.get_value(), 3)) +\
               "\\nN = " + str(environment_node.visits)
    
    def state_node_to_graph(self, g, state_node, level):
        for action in state_node.children.keys():
            g.edge(self.state_node_id(state_node), self.environment_node_id(state_node.children[action]), action)

        if level <= self.max_level:
            for action in state_node.children.keys():
                self.environment_node_to_graph(g, state_node.children[action], level)

    def environment_node_to_graph(self, g, environment_node, level):
        for child in environment_node.children:
            g.node(self.environment_node_id(environment_node), self.environment_node_id(environment_node), style='filled', shape='point', width='0.25')
            g.edge(self.environment_node_id(environment_node), self.state_node_id(child), str(child.probability))

        for child in environment_node.children:
            self.state_node_to_graph(g, child, level + 1)

    def node_to_graph(self, game, node, filename='backward_induction', print_state = False, print_value = False):
        graph = Graph('G', filename=filename, format='png')
        self.game_node(graph, game, node, visited = [], level = 1, print_state = print_state, print_value = print_value)
        return graph

    def node_to_string(self, game, node, print_state, print_value):
        result = ""
        if print_state:
            result += game.toString(node.state) + "\\n"
        if len(node.children) == 0 or print_value:
            result += "("
            result += ", ".join([str(node.value[player]) for player in node.value.keys()])
            result += ")"
        return result

    def game_node(self, graph, game, node, visited, level, print_state, print_value):
        
        if node.id not in visited:
            graph.node(str(node.id), label=self.node_to_string(game, node, print_state, print_value), xlabel = str(node.player_turn) if node.player_turn is not None else "")
            if level <= self.max_level:
                for key in node.children.keys():
                    child = node.children[key]
                    self.game_node(graph, game, child, visited, level + 1, print_state, print_value)
                    penwidth = '3.0' if child.isBestAction else '1.0'
                    graph.edge(str(node.id), str(child.id), str(key), penwidth = penwidth)
            visited += [node.id]
            


