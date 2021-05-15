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
