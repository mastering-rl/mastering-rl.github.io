
class MDP:
    def getStates(self): abstract
    def getActions(self, state): abstract
    def getTransitions(self, state, action): abstract
    def getReward(self, state, action, nextState): abstract
    def getDiscountFactor(self): abstract
    def getInitialStates(self): abstract
    def getGoalStates(self): abstract

    ''' Returns a new state and a reward for executing action in state '''
    def simulate(self, state, action): abstract


class NavigationMDP(MDP):

    # labels for terminate action and terminal state
    TERMINATE = 'terminate'
    TERMINAL = ('terminal', 'terminal')

    def __init__(self, width = 4, height = 3,
                 discountFactor = 0.9, 
                 blockedStates = [(1,1)],
                 goals = [((3,2), 1), ((3,1), -1)]):
        self.width = width
        self.height = height
        self.blockedStates = blockedStates
        self.discountFactor = discountFactor
        self.goalStates = dict(goals)

    def getStates(self):
        states = [self.TERMINAL]
        for x in range(self.width):
            for y in range(self.height):
                if not (x, y) in self.blockedStates:
                    states.append((x,y))
        return states

    def getActions(self, state=None):

        if (state == None):
            return ['N', 'S', 'E', 'W', self.TERMINATE]

        actions = []
        for action in ['N', 'S', 'E', 'W', self.TERMINATE]:
            for (newState, probability) in self.getTransitions(state, action):
                if probability > 0:
                    actions.append(action)
        return actions

    def getGoalStates(self):
        return self.goalStates

    def validAdd(self, state, newState, probability):
        # if the next state is blocked, stay in the same state
        if (newState in self.blockedStates):
            return [(state, probability)]

        # move to the next space if it is not off the grid
        (x, y) = newState
        if (x >= 0 and x < self.width and y >= 0 and y < self.height):
            return [((x, y), probability)]
 
        # if off the grid, state in the same state
        return [(state, probability)]

    def getTransitions(self, state, action):
        transitions = []

        if state == self.TERMINAL:
            return [(self.TERMINAL, 1.0)]

        (x, y) = state
        if state in self.getGoalStates().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]

        elif action == 'N':
            transitions += self.validAdd(state, (x, y + 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == 'S':
            transitions += self.validAdd(state, (x, y - 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == 'E':
            transitions += self.validAdd(state, (x + 1, y), 0.8)
            transitions += self.validAdd(state, (x, y - 1), 0.1)
            transitions += self.validAdd(state, (x, y + 1), 0.1)

        elif action == 'W':
            transitions += self.validAdd(state, (x - 1, y), 0.8)
            transitions += self.validAdd(state, (x, y - 1), 0.1)
            transitions += self.validAdd(state, (x, y + 1), 0.1)

        return transitions

    def getReward(self, state, action, newState):
       reward = 0.0
       if state in self.getGoalStates().keys() and newState == self.TERMINAL:
          reward = self.getGoalStates().get(state)
       return reward

    def getDiscountFactor(self):
        return self.discountFactor

    ''' Convert a grid world value function to a formatted string '''
    def valueFunctionToString(self, values):
        line = " {:-^{n}}\n".format("", n=len(" | +0.00")*self.width + 1)
        result = line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " | #####"
                else:
                    result += " | {:+0.2f}".format(values[(x, y)])
            result += " |\n"
            result += line

        return result


class ValueIteration():

    def __init__(self, mdp):
        self.mdp = mdp

    ''' To run until convergence (minus theta), set iterations to -1 '''
    def valueIteration(self, iterations = -1, theta = 0.01):
        values = self.initialiseValueFunction()
        for _ in range(iterations):
           delta = 0
           newValues = self.initialiseValueFunction()
           for state in mdp.getStates():
               oldValue = values[state]
               actionValues = dict()
               for action in mdp.getActions(state):
                   newValue = 0.0
                   for (newState, probability) in mdp.getTransitions(state, action):
                       reward = mdp.getReward(state, action, newState)
                       newValue += probability * (reward + (mdp.getDiscountFactor() * values[newState]))
                   actionValues.update({action: newValue})                
               newValues.update({state: max(actionValues.values())})
           values = newValues

        return newValues

    def initialiseValueFunction(self):
        values = dict()
        for state in self.mdp.getStates():
            values.update({state: 0.0})
        return values




mdp = NavigationMDP()

valueIteration = ValueIteration(mdp)

for i in [1, 2, 3, 4, 5, 10, 1000]:
#for i in [1]:
    print("After iteration " + str(i))
    #mdp.valueFunctionToString(valueIteration.valueIteration(iterations = i))
    print(mdp.valueFunctionToString(valueIteration.valueIteration(iterations = i)) + "\n")

