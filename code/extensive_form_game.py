class ExtensiveFormGame:

    ''' Get the list of players for this game as a list [1, ..., N] '''
    def getPlayers(self): abstract

    ''' Get the valid actions at a state '''
    def getActions(self, state): abstract

    ''' Return the state resulting from playing an action in a state '''
    def getTransition(self, state, action): abstract

    ''' Return the reward for a state, return as a dictionary mapping players to rewards '''
    def getReward(self, state, action, nextState): abstract

    ''' Return true if and only if state is a terminal state of this game '''
    def isTerminal(self, state): abstract

    ''' Return the player who selects the action at this state (whose turn it is) '''
    def getPlayerTurn(self, state): abstract
    
    ''' Return the initial state of this game '''
    def getInitialState(self): abstract

    ''' Return a game tree for this game '''
    def gameTree(self):
        return self.stateToNode(self.getInitialState())

    def stateToNode(self, state):
        if self.isTerminal(state):
            node = GameNode(state, None, self.getReward(state))
            return node

        player = self.getPlayerTurn(state)
        children = dict()
        for action in self.getActions(state):
            nextState = self.getTransition(state, action)
            child = self.stateToNode(nextState)
            children[action] = child
        node = GameNode(state, player, None, children = children)
        return node

class GameNode:

    # record a unique node id to distinguish duplicated states
    nextNodeID = 0

    def __init__(self, state, playerTurn, value, isBestAction = False, children = dict()):
        self.state = state
        self.playerTurn = playerTurn
        self.value = value
        self.isBestAction = isBestAction
        self.children = children

        self.id = GameNode.nextNodeID
        GameNode.nextNodeID += 1
        
