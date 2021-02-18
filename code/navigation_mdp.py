from mdp import *

class NavigationMDP(MDP):

    # labels for terminate action and terminal state
    TERMINATE = 'terminate'
    TERMINAL = ('terminal', 'terminal')
    LEFT = '\u25C4'
    UP  = '\u25B2'
    RIGHT = '\u25BA'
    DOWN = '\u25BC'
    
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

        actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.TERMINATE]
        #actions = ["Up", "Down", "Left", "Right", "X"]
        if (state == None):
            return actions

        validActions= []
        for action in actions:
            for (newState, probability) in self.getTransitions(state, action):
                if probability > 0:
                    validActions.append(action)
                    break
        return validActions

    def getInitialState(self):
        return (0,0)

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

        elif action == self.UP:
            transitions += self.validAdd(state, (x, y + 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == self.DOWN:
            transitions += self.validAdd(state, (x, y - 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == self.RIGHT:
            transitions += self.validAdd(state, (x + 1, y), 0.8)
            transitions += self.validAdd(state, (x, y - 1), 0.1)
            transitions += self.validAdd(state, (x, y + 1), 0.1)

        elif action == self.LEFT:
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

    def isTerminal(self, state):
        if state == self.TERMINAL:
            return True
        return False

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

    ''' Convert a grid world Q function to a formatted string '''
    def qFunctionToString(self, qValues):
        leftArrow = '\u25C4'
        upArrow = '\u25B2'
        rightArrow = '\u25BA'
        downArrow = '\u25BC'
        
        space = " |               "

        line = "  "
        for x in range(self.width):
            line += "---------------- "
        line += "\n"

        result = line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in self.getGoalStates().keys():
                    result += space
                else:
                    result += " |       {}       ".format(upArrow)
            result += " |\n"
            
            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in self.getGoalStates().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(qValues[((x, y), self.UP)])
            result += " |\n"
            
            for x in range(self.width):
                result += space
            result += " |\n"
            
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " |     #####     "
                elif (x, y) in self.getGoalStates().keys():
                    result += " |     {:+0.2f}     ".format(qValues[((x, y), self.TERMINATE)])
                else:
                    result += " | {}{:+0.2f}  {:+0.2f}{}".format(leftArrow, qValues[((x, y), self.LEFT)], qValues[((x, y), self.RIGHT)], rightArrow)
            result += " |\n"

            for x in range(self.width):
                result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in self.getGoalStates().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(qValues[((x, y), self.DOWN)])
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in self.getGoalStates().keys():
                    result += space
                else:
                    result += " |       {}       ".format(downArrow)
            result += " |\n"
            result += line        
        return result

    ''' Convert a grid world policy to a formatted string '''
    def policyToString(self, policy):
        line = " {:-^{n}}\n".format("", n=len(" |  N ")*self.width + 1)
        result = line 
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " | ###"
                else:
                    action = "T" if policy[(x, y)] == self.TERMINATE else policy[(x, y)]
                    result += " |  " + action + " "
            result += " |\n"
            result += line

        return result
