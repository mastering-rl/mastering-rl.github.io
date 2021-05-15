from graphviz import Digraph, Graph

class GraphVisualisation():

    def __init__(self, maxLevel = 3):
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

    def nodeToGraph(self, game, node, filename='backward_induction'):
        graph = Graph('G', filename=filename, format='png')
        #for child in node.children.keys():
        #    self.gameNode(graph, game, node.children[child], level = 0)
        self.gameNode(graph, game, node, visited = [], level = 0)
        return graph

    def gameNode(self, graph, game, node, visited, level):
        graph.node(str(node.id), style='filled', shape='point', width='0.2')
        if node.id not in visited:
            for key in node.children.keys():
                penwidth = '3.0' if node.children[key].isBestAction else '1.0'
                
                #graph.edge(str(node.id) + "\\n" + game.toString(node.state), str(node.children[key].id) + "\\n" + game.toString(node.children[key].state), str(key), arrowType="diamond", penwidth = penwidth)
                #graph.edge(game.toString(node.state), game.toString(node.children[key].state), str(key))
                graph.edge(str(node.id), str(node.children[key].id), str(key), penwidth = penwidth)
            visited += [node.id]
            

        if level <= self.maxLevel:
            for key in node.children.keys():
                self.gameNode(graph, game, node.children[key], visited, level)
