from extensive_form_game import ExtensiveFormGame

EMPTY = ' '
NOUGHT = 'o'
CROSS = 'x'

class TicTacToe(ExtensiveFormGame):



    ''' Initialise a TicTacToe game '''
    def __init__(self):
        self.players = [CROSS, NOUGHT]
    
    ''' Get the list of players for this game as a list [1, ..., N] '''
    def getPlayers(self):
        return self.players

    ''' Get the valie actions at a state '''
    def getActions(self, state):

        #use a nicer variable name for this implementation
        board = state
        
        actions = []
        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == EMPTY:
                    actions += [(x,y)]
        return actions

    '''
        Deep copy this state
    '''
    def copy(self, state):
        nextState = []
        for x in range(len(state)):
            newRow = []
            for y in range(len(state[x])):
                 newRow += [state[x][y]]
            nextState += [newRow]
        return nextState

    ''' Return the state resulting from playing an action in a state '''
    def getTransition(self, state, action):
        nextState = self.copy(state)
        nextState[action[0]][action[1]] = self.getPlayerTurn(state)
        return nextState

    ''' Return the reward for a state '''
    def getReward(self, state):
        winner = self.getWinner(state)
        if winner == None:
            return {CROSS:0, NOUGHT:0}
        elif winner == CROSS:
            return {CROSS:1, NOUGHT:-1}
        elif winner == NOUGHT:
            return {CROSS:-1, NOUGHT:1}
        
    def countEmpty(self, board):
        empty = 0
        for x in range(len(board)):
            for y in range(len(board[x])):
                if board[x][y] == EMPTY:
                    empty += 1
        return empty
    
    ''' Return true if and only if state is a terminal state of this game '''
    def isTerminal(self, state):
        return self.countEmpty(state) == 0 or self.getWinner(state) is not None


    ''' Return the player who selects the action at the current state (whose turn it is) '''
    def getPlayerTurn(self, state):

        #use a nicer variable name for this implementation
        board = state
        
        empty = self.countEmpty(board)
        
        #crosses starts the game, so if there is an odd number of empty cells, it is crosses turn
        if empty % 2 == 0:
            return NOUGHT
        else:
            return CROSS
    
    ''' Return the initial state of this game '''
    def getInitialState(self):
        board = [[EMPTY, EMPTY, EMPTY],
                 [EMPTY, EMPTY, EMPTY],
                 [EMPTY, EMPTY, EMPTY]]
        return board

    def getWinner(self, state):

        #use a nicer variable name for this implementation
        board = state

        #check columns
        for x in range(0, len(board)):
            noughts = 0
            crosses = 0
            for y in range(0, len(board[x])):
                if board[x][y] == NOUGHT:
                    noughts += 1
                elif board[x][y] == CROSS:
                    crosses += 1
            if noughts == len(board[0]):
                return NOUGHT
            elif crosses == len(board[0]):
                return CROSS

        #check rows
        for y in range(0, len(board[0])):
            noughts = 0
            crosses = 0
            for x in range(0, len(board)):
                if board[x][y] == NOUGHT:
                    noughts += 1
                elif board[x][y] == CROSS:
                    crosses += 1
            if noughts == len(board):
                return NOUGHT
            elif crosses == len(board):
                return CROSS

        #check top-left to bottom-right diagonal
        if board[0][0] == NOUGHT and board[1][1] == NOUGHT and board[2][2] == NOUGHT:
            return NOUGHT
        elif board[0][0] == CROSS and board[1][1] == CROSS and board[2][2] == CROSS:
            return CROSS

        #check bottom-left to top-right diagonal
        if board[0][2] == NOUGHT and board[1][1] == NOUGHT and board[2][0] == NOUGHT:
            return NOUGHT
        elif board[0][2] == CROSS and board[1][1] == CROSS and board[2][0] == CROSS:
            return CROSS

        #no winner
        return None
            

    def toString(self, state):
        """
        Formats a board as a string replacing cell values with enum names.
        Args:
            board (numpy.ndarray): two dimensional array representing the board
                after the move
        Returns:
            str: the board represented as a string
        """
        # Join columns using '|' and rows using line-feeds
        result = str('\\n'.join(['|'.join([item for item in row]) for row in state]))
        return result

if __name__ == "__main__":
    tictactoe = TicTacToe()
    state = tictactoe.getInitialState()
    state = tictactoe.getTransition(state, (0,0))
    state = tictactoe.getTransition(state, (1,2))
    state = tictactoe.getTransition(state, (1,1))
    state = tictactoe.getTransition(state, (2,1))
    assert tictactoe.getWinner(state) is None
    assert tictactoe.getReward(state) == {CROSS:0, NOUGHT:0}
    
    state = tictactoe.getTransition(state, (2,2))
    print(tictactoe.toString(state))
    assert tictactoe.getWinner(state) == CROSS
    assert tictactoe.getReward(state) == {CROSS:1, NOUGHT:-1}

    # play a random game
    import random
    state = tictactoe.getInitialState()
    while not tictactoe.isTerminal(state):
        actions = tictactoe.getActions(state)
        state = tictactoe.getTransition(state, random.choice(actions))
        print(tictactoe.toString(state) + "\n")
    print("winner is %s" % tictactoe.getWinner(state))
        
    
    
