from extensive_form_game import GameNode

class BackwardInduction:
    def __init__(self, game, doCache = False):
        self.game = game
        self.doCache = doCache
        self.cache = dict()

    def backwardInduction(self, state):

        if self.game.isTerminal(state):
            node = GameNode(state, None, self.game.getReward(state))
            return node

        bestChild = None
        bestAction = None
        player = self.game.getPlayerTurn(state)
        children = dict()
        for action in self.game.getActions(state):
            nextState = self.game.getTransition(state, action)
            child = self.backwardInduction(nextState)
            if bestChild is None or child.value[player] > bestChild.value[player]:
                if bestChild is not None:
                    bestChild.isBestAction = False
                child.isBestAction = True
                bestChild = child
            children[action] = child
        node = GameNode(state, player, bestChild.value, children = children)
        return node

    def backwardInductionWithCache(self, state):

        stateKey = self.game.toString(state)
        if self.doCache and stateKey in self.cache.keys():
            return self.cache[stateKey]

        if self.game.isTerminal(state):
            node = GameNode(state, None, self.game.getReward(state))
            if self.doCache:
                self.cache[stateKey] = node
            return node

        bestChild = None
        bestAction = None
        player = self.game.getPlayerTurn(state)
        children = dict()
        for action in self.game.getActions(state):
            nextState = self.game.getTransition(state, action)
            child = self.backwardInduction(nextState)
            if bestChild is None or child.value[player] > bestChild.value[player]:
                if bestChild is not None:
                    bestChild.isBestAction = False
                child.isBestAction = True
                bestChild = child
            children[action] = child
        node = GameNode(state, player, bestChild.value, children = children)
        if self.doCache:
            self.cache[stateKey] = node
        return node



if __name__ == "__main__":

    from tictactoe import TicTacToe

    import time
    
    tictactoe = TicTacToe()
    initialState = tictactoe.getInitialState()
    start = time.time_ns()
    backwardInduction = BackwardInduction(tictactoe)
    #solution = backwardInduction.backwardInduction(initialState)
    finish = time.time_ns()
    print("Non-cached execution time = %f" % ((finish - start) / 1000000))

    tictactoe = TicTacToe()
    initialState = tictactoe.getInitialState()
    start = time.time_ns()
    initialState = [['x', 'o', 'o'],
                    [' ', ' ', 'x'],
                    [' ', ' ', ' ']]
    nextState = tictactoe.getTransition(initialState, (1, 1))
    backwardInduction = BackwardInduction(tictactoe, doCache = False)
    solution = backwardInduction.backwardInduction(nextState)
    finish = time.time_ns()
    print("Cached execution time = %f" % ((finish - start) / 1000000))

    from graph_visualisation import GraphVisualisation
    
    gv = GraphVisualisation()
    graph = gv.nodeToGraph(tictactoe, solution, printState = True, printValue = True)
    graph.view()
