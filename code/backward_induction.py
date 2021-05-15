from collections import defaultdict

class GameNode:

    # record a unique node id to distinguish duplicated states
    nextNodeID = 0

    def __init__(self, state, playerTurn, equilibrium, isBestAction = False, children = dict()):
        self.state = state
        self.playerTurn = playerTurn
        self.equilibrium = equilibrium
        self.isBestAction = isBestAction
        self.children = children

        self.id = GameNode.nextNodeID
        GameNode.nextNodeID += 1
        
class BackwardInduction:
    def __init__(self, game, doCache = False):
        self.game = game
        self.doCache = doCache
        self.cache = dict()

    def backwardInduction(self, state):

        stateKey = self.game.toString(state)
        if self.doCache and stateKey in self.cache.keys():
            return self.cache[stateKey]

        if self.game.isTerminal(state):
            return GameNode(state, None, self.game.getReward(state))

        bestChild = None
        bestAction = None
        player = self.game.getPlayerTurn(state)
        children = dict()
        for action in self.game.getActions(state):
            nextState = self.game.getTransition(state, action)
            child = self.backwardInduction(nextState)
            if bestChild is None or child.equilibrium[player] > bestChild.equilibrium[player]:
                if bestChild is not None:
                    bestChild.isBestAction = False
                child.isBestAction = True
                bestChild = child
            children[action] = child
        node = GameNode(state, player, bestChild.equilibrium, isBestAction = True, children = children)
        if self.doCache:
            self.cache[stateKey] = node
        return node


if __name__ == "__main__":

    from tictactoe import TicTacToe

    import time
    
    initialState = [['x', 'x', ' '],
                    ['o', ' ', ' '],
                    ['x', 'o', ' ']]

    tictactoe = TicTacToe()
    initialState = tictactoe.getInitialState()
    start = time.time_ns()
    backwardInduction = BackwardInduction(tictactoe)
    solution = backwardInduction.backwardInduction(initialState)
    finish = time.time_ns()
    print("Non-cached execution time = %f" % ((finish - start) / 1000000))

    tictactoe = TicTacToe()
    initialState = tictactoe.getInitialState()
    start = time.time_ns()
    backwardInduction = BackwardInduction(tictactoe, doCache = True)
    solution = backwardInduction.backwardInduction(initialState)
    finish = time.time_ns()
    print("Cached execution time = %f" % ((finish - start) / 1000000))

    from graph_visualisation import GraphVisualisation
    
    #gv = GraphVisualisation(maxLevel = 5)
    #g = gv.nodeToGraph(tictactoe, solution)
    #g.view()
