from graphviz import Digraph, Graph

class GraphVisualisation():

    def __init__(self, maxLevel = float('inf')):
        self.maxLevel = maxLevel

    def singleAgentMCTSToGraph(self, stateNode, filename='mcts'):
        g = Digraph('G', filename=filename, format='png')
        self.stateNodeToGraph(g, stateNode, level = 0)
        return g

    def stateNodeID(self, stateNode):
        return "V(" + str(stateNode.state) + "." + str(stateNode.id) + ") = " + str(round(stateNode.getValue(), 3)) +\
               "\\nN = " + str(stateNode.visits)

    def environmentNodeID(self, environmentNode):
        return "V(" + str(environmentNode.id) + ") = " + str(round(environmentNode.getValue(), 3)) +\
               "\\nN = " + str(environmentNode.visits)
    
    def stateNodeToGraph(self, g, stateNode, level):
        for action in stateNode.children.keys():
            g.edge(self.stateNodeID(stateNode), self.environmentNodeID(stateNode.children[action]), action)

        if level <= self.maxLevel:
            for action in stateNode.children.keys():
                self.environmentNodeToGraph(g, stateNode.children[action], level)

    def environmentNodeToGraph(self, g, environmentNode, level):
        for child in environmentNode.children:
            g.node(self.environmentNodeID(environmentNode), self.environmentNodeID(environmentNode), style='filled', shape='point', width='0.25')
            g.edge(self.environmentNodeID(environmentNode), self.stateNodeID(child), str(child.probability))

        for child in environmentNode.children:
            self.stateNodeToGraph(g, child, level + 1)

    def nodeToGraph(self, game, node, filename='backward_induction', printState = False, printValue = False):
        graph = Graph('G', filename=filename, format='png')
        self.gameNode(graph, game, node, visited = [], level = 1, printState = printState, printValue = printValue)
        return graph

    def nodeToString(self, game, node, printState, printValue):
        result = ""
        if printState:
            result += game.toString(node.state) + "\\n"
        if len(node.children) == 0 or printValue:
            result += "("
            result += ", ".join([str(node.value[player]) for player in node.value.keys()])
            result += ")"
        return result

    def gameNode(self, graph, game, node, visited, level, printState, printValue):
        
        if node.id not in visited:
            graph.node(str(node.id), label=self.nodeToString(game, node, printState, printValue), xlabel = str(node.playerTurn) if node.playerTurn is not None else "")
            if level <= self.maxLevel:
                for key in node.children.keys():
                    child = node.children[key]
                    self.gameNode(graph, game, child, visited, level + 1, printState, printValue)
                    penwidth = '3.0' if child.isBestAction else '1.0'
                    graph.edge(str(node.id), str(child.id), str(key), penwidth = penwidth)
            visited += [node.id]
            


