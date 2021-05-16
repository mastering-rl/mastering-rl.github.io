from extensive_form_game import ExtensiveFormGame

ONE = "1"
TWO = "2"

class AbstractExtensiveFormGame(ExtensiveFormGame):

    ''' Get the list of players for this game as a list [1, ..., N] '''
    def getPlayers(self):
        return [ONE, TWO]

    ''' Get the valid actions at a state '''
    def getActions(self, state):
        actions = dict()
        actions[1] = ["A", "B"]
        actions[2] = ["C", "D"]
        actions[3] = ["E", "F"]
        actions[7] = ["G", "H"]

        if state in actions.keys():
            return actions[state]
        else:
            return []

    ''' Return the state resulting from playing an action in a state '''
    def getTransition(self, state, action):
        transitions = dict()
        if state == 1:
            transitions["A"] = 2
            transitions["B"] = 3
        elif state == 2:
            transitions["C"] = 4
            transitions["D"] = 5
        elif state == 3:
            transitions["E"] = 6
            transitions["F"] = 7
        elif state == 7:
            transitions["G"] = 8
            transitions["H"] = 9
        else: 
            transitions[action] = []
        return transitions[action]

    ''' Return the reward for a state, return as a dictionary mapping players to rewards '''
    def getReward(self, state):
        rewards = dict()
        if state in [4,5,6,8,9]:
            rewards[4] = {ONE:3, TWO: 8}
            rewards[5] = {ONE:8, TWO: 3}
            rewards[6] = {ONE:5, TWO: 5}
            rewards[8] = {ONE:2, TWO: 10}
            rewards[9] = {ONE:1, TWO: 0}
            return rewards[state]
        else:
            return {ONE:0, TWO:0}

    ''' Return true if and only if state is a terminal state of this game '''
    def isTerminal(self, state):
        return state in [4,5,6,8,9]

    ''' Return the player who selects the action at this state (whose turn it is) '''
    def getPlayerTurn(self, state):
        if state in [1,7]:
            return ONE
        else:
            return TWO
    
    ''' Return the initial state of this game '''
    def getInitialState(self):
        return 1

    def toString(self, state):
        return str(state)

if __name__ == "__main__":
    from backward_induction import BackwardInduction

    game = AbstractExtensiveFormGame()
    backwardInduction = BackwardInduction(game)
    solution = backwardInduction.backwardInduction(game.getInitialState())
    from graph_visualisation import GraphVisualisation
    
    gv = GraphVisualisation(maxLevel = 5)
    graph = gv.nodeToGraph(game, game.gameTree(), printValues = False)
    graph.view()
