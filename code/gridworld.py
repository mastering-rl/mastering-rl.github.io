from mdp import *

class GridWorld(MDP):

    # labels for terminate action and terminal state
    TERMINATE = 'terminate'
    TERMINAL = ('terminal', 'terminal')
    LEFT = '\u25C4'
    UP  = '\u25B2'
    RIGHT = '\u25BA'
    DOWN = '\u25BC'
    
    def __init__(self,
                 noise = 0.1,
                 width = 4, height = 3,
                 discountFactor = 0.9, 
                 blockedStates = [(1,1)],
                 actionCost = 0.0,
                 goals = [((3,2), 1), ((3,1), -1)]):
        self.noise = noise
        self.width = width
        self.height = height
        self.blockedStates = blockedStates
        self.discountFactor = discountFactor
        self.actionCost = actionCost
        self.goalStates = dict(goals)

        # A list of lists that  records all rewards given at each step for each episode of a simulated gridworld
        self.rewards = []
        self.episodeRewards = []  # The rewards for the current episode

    def getStates(self):
        states = [self.TERMINAL]
        for x in range(self.width):
            for y in range(self.height):
                if not (x, y) in self.blockedStates:
                    states.append((x,y))
        return states

    def getActions(self, state=None):

        actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.TERMINATE]
        if state == None:
            return actions

        validActions = []
        for action in actions:
            for (newState, probability) in self.getTransitions(state, action):
                if probability > 0:
                    validActions.append(action)
                    break
        return validActions

    def getInitialState(self):
        self.episodeRewards = []
        return (0,0)

    def getGoalStates(self):
        return self.goalStates

    def validAdd(self, state, newState, probability):
        # if the next state is blocked, stay in the same state
        if newState in self.blockedStates:
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
            if action == self.TERMINATE:
                return [(self.TERMINAL, 1.0)]
            else:
                return []

        # probability of not slipping left or right
        straight = 1 - (2 * self.noise)

        (x, y) = state
        if state in self.getGoalStates().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]

        elif action == self.UP:
            transitions += self.validAdd(state, (x, y + 1), straight)
            transitions += self.validAdd(state, (x - 1, y), self.noise)
            transitions += self.validAdd(state, (x + 1, y), self.noise)

        elif action == self.DOWN:
            transitions += self.validAdd(state, (x, y - 1), straight)
            transitions += self.validAdd(state, (x - 1, y), self.noise)
            transitions += self.validAdd(state, (x + 1, y), self.noise)

        elif action == self.RIGHT:
            transitions += self.validAdd(state, (x + 1, y), straight)
            transitions += self.validAdd(state, (x, y - 1), self.noise)
            transitions += self.validAdd(state, (x, y + 1), self.noise)

        elif action == self.LEFT:
            transitions += self.validAdd(state, (x - 1, y), straight)
            transitions += self.validAdd(state, (x, y - 1), self.noise)
            transitions += self.validAdd(state, (x, y + 1), self.noise)

        return transitions

    def getReward(self, state, action, newState):
        reward = 0.0
        if state in self.getGoalStates().keys() and newState == self.TERMINAL:
            reward = self.getGoalStates().get(state)
        else:
            reward = self.actionCost
        step = len(self.episodeRewards)
        self.episodeRewards += [reward * (self.discountFactor ** step)]
        return reward

    def getDiscountFactor(self):
        return self.discountFactor

    def isTerminal(self, state):
        if state == self.TERMINAL:
            self.rewards += [self.episodeRewards]
            return True
        return False


    '''
        Returns a list of lists, which records all rewards given at each step
        for each episodeof a simulated gridworld
    ''' 
    def getRewards(self):
        return self.rewards
        

    ''' Visualise a grid world problem as a formatted string '''
    def visualise(self):
        leftArrow = '\u25C4'
        upArrow = '\u25B2'
        rightArrow = '\u25BA'
        downArrow = '\u25BC'
        
        space = " |              "
        block = " | #############"

        line = "  "
        for x in range(self.width):
            line += "--------------- "
        line += "\n"

        result = line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.getGoalStates().keys():
                    result += space
                elif (x, y) in self.blockedStates:
                    result += block
                else:
                    result += " |       {}      ".format(upArrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.getInitialState():
                    result += " |     _____    "
                elif (x, y) in self.blockedStates:
                    result += block
                else:
                    result += space
            result += " |\n"
            
            for x in range(self.width):
                if (x, y) == self.getInitialState():
                    result += " |    ||o  o|   "
                elif (x, y) in self.blockedStates:
                    result += block
                else:
                    result += space
            result += " |\n"

            
            for x in range(self.width):
                if (x, y) == self.getInitialState():
                    result += " | {}  ||  * |  {}".format(leftArrow, rightArrow)
                elif (x, y) in self.blockedStates:
                    result += block
                elif (x, y) in self.getGoalStates().keys():
                    result += " |     {:+0.2f}    ".format(self.getGoalStates()[(x,y)])
                else:
                    result += " | {}           {}".format(leftArrow, rightArrow)
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.getInitialState():
                    result += " |    ||====|   ".format(leftArrow, rightArrow)
                elif (x, y) in self.blockedStates:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) == self.getInitialState():
                    result += " |     -----    "
                elif (x, y) in self.blockedStates:
                    result += block
                else:
                    result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.getGoalStates().keys():
                    result += space
                elif (x, y) in self.blockedStates:
                    result += block
                else:
                    result += " |       {}      ".format(downArrow)
            result += " |\n"
            result += line        
        return result

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
                    result += " |     {:+0.2f}     ".format(MDP.getQValue(qValues, (x, y), self.UP))
            result += " |\n"
            
            for x in range(self.width):
                result += space
            result += " |\n"
            
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " |     #####     "
                elif (x, y) in self.getGoalStates().keys():
                    result += " |     {:+0.2f}     ".format(MDP.getQValue(qValues, (x, y), self.TERMINATE))
                else:
                    result += " | {}{:+0.2f}  {:+0.2f}{}".format(leftArrow, MDP.getQValue(qValues, (x, y), self.LEFT), MDP.getQValue(qValues, (x, y), self.RIGHT), rightArrow)
            result += " |\n"

            for x in range(self.width):
                result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in self.getGoalStates().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(MDP.getQValue(qValues, (x, y),  self.DOWN))
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

class CliffWorld(GridWorld):
    def __init__(self, noise = 0.0, discountFactor = 1.0, width = 6, height = 4,
                 blockedStates = [], actionCost = -0.05,
                 goals = [((1,0), -5), ((2,0), -5), ((3,0), -5), ((4,0), -5), ((5,0), 0)]):
        super().__init__(noise = noise, discountFactor = discountFactor,
                         width = width, height = height,
                         blockedStates = blockedStates, actionCost = actionCost, goals = goals)
                     
