from collections import defaultdict
import matplotlib.pyplot as plt

from mdp import *
from rendering_utils import *


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
                 initialState = (0, 0),
                 goals = [((3,2), 1), ((3,1), -1)]):
        self.noise = noise
        self.width = width
        self.height = height
        self.blockedStates = blockedStates
        self.discountFactor = discountFactor
        self.actionCost = actionCost
        self.initialState = initialState
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
        return self.initialState

    def getGoalStates(self):
        return self.goalStates

    def validAdd(self, state, newState, probability):
        # if the next state is blocked, stay in the same state
        if probability == 0.0:
            return []

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

        # merge any duplicate outcomes
        # TODO: change the transitions data structure into a dictionary
        merged = defaultdict(lambda: 0.0)
        for (state, probability) in transitions:
            merged[state] = merged[state] + probability

        transitions = []
        for outcome in merged.keys():
            transitions += [(outcome, merged[outcome])]

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

    '''
        Create a gridworld from an array of strings: one for each line
        - First line is rewards as a dictionary from cell to value: {'A': 1, ...}
        - space is an empty cell
        - # is a blocked cell
        - @ is the agent (initial state)
        - new 'line' is a new row
        - a letter is a cell with a reward for transitioning into that cell. The reward defined by the first line.
    '''
    @staticmethod
    def create(string):
        # Parse the reward on the first line
        import ast
        rewards = ast.literal_eval(string[0])

        width = 0
        height = len(string) - 1

        blockedCells = []
        initialState = (0,0)
        goals = []
        row = 0
        for nextRow in string[1:]:
            column = 0
            for cell in nextRow:
                if cell == '#':
                    blockedCells += [(column, row)]
                elif cell == '@':
                    initialState = (column, row)
                elif cell.isalpha():
                    goals += [((column, row), rewards[cell])]
                column += 1
            width = max(width, column)
            row += 1
        return GridWorld(width = width, height = height, blockedStates = blockedCells, initialState = initialState, goals = goals)

    @staticmethod
    def open(file):
        file = open(file, "r")
        string = file.read().splitlines()
        file.close()
        return GridWorld.create(string)

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
        for y in range(self.height):
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
        for y in range(self.height): # - 1, -1, -1):
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
        for y in range(self.height): # - 1, -1, -1):
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

    ''' Visualise a gridworld problem as a small string '''
    def visualise_small(self):
        line = " {:-^{n}}\n".format("", n=len("| N")*self.width + 1)
        result = line
        for y in range(self.height): # - 1, -1, -1):
            result += " "
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += "|##"
                elif (x, y) in self.goalStates.keys():
                    result += "|{:+d}".format(self.goalStates[(x,y)])
                elif (x, y) == self.initialState:
                    result += "|@@"
                else:
                    result += "|  "
            result += "|\n"
            result += line

        return result

    ''' Convert a grid world policy to a formatted string '''
    def policyToString(self, policy):
        line = " {:-^{n}}\n".format("", n=len(" |  N ")*self.width + 1)
        result = line
        for y in range(self.height): # - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " | ###"
                else:
                    action = "T" if policy[(x, y)] == self.TERMINATE else policy[(x, y)]
                    result += " |  " + action + " "
            result += " |\n"
            result += line

        return result

    ''' visualise the gridworld problem as a matplotlib image '''
    def visualiseImage(self, tileSize=32, agentPosition=None, title=""):
        widthPx = self.width * tileSize
        heightPx = self.height * tileSize
        currentPosition = self.getInitialState() if agentPosition is None else agentPosition
        img = [[[0, 0, 0] for _ in range(widthPx)] for _ in range(heightPx)]

        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                if (x, y) in self.goalStates:
                    self.renderTile(x, y, tileSize, img, 'goal')
                elif (x, y) in self.blockedStates:
                    self.renderTile(x, y, tileSize, img, 'blocked')
                elif (x, y) == currentPosition:
                    self.renderTile(x, y, tileSize, img, 'agent')
                else:
                    self.renderTile(x, y, tileSize, img, 'empty')

        plt.imshow(img, origin='lower', interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Grid World {title}')
        plt.show()

    '''Render each tile individually depending on the current state of the cell'''
    def renderTile(self, x, y, tileSize, img, tileType=None):
        ymin = y * tileSize
        ymax = (y + 1) * tileSize
        xmin = x * tileSize
        xmax = (x + 1) * tileSize

        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                if i == ymin or i == ymax-1 or j == xmin or j == xmax+1:
                    drawGridLines(i, j, img)
                else:
                    if tileType == 'goal':
                        renderGoal(i, j, img, reward=self.goalStates[(x, y)], reward_max=max(self.getGoalStates().values()), reward_min=min(self.getGoalStates().values()))
                    elif tileType == 'blocked':
                        renderBlockedTile(i, j, img)
                    elif tileType == 'agent':
                        renderAgent(i, j, img, center_x=xmin + tileSize / 2, center_y=ymin + tileSize / 2, radius=tileSize / 4)
                    elif tileType == 'empty':
                        img[i][j] = [0, 0, 0]
                    else:
                        raise ValueError("Invalid tile type")

    '''Visualise the value function using a heat-map where green is high value and red is low value'''
    def visualiseValueFunction(self, valueDict, title=''):
        values = [[0 for _ in range(self.width)] for _ in range(self.height)]
        plt.imshow(values, origin='lower', cmap=makeRedWhiteGreenCmap())
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    values[y][x] = 0
                    plt.text(x, y, '#', horizontalalignment='center', verticalalignment='center')
                else:
                    values[y][x] = valueDict[(x, y)]
                    plt.text(x, y, f'{values[y][x]:.2f}', horizontalalignment='center', verticalalignment='center')
        plt.imshow(values, origin='lower', cmap=makeRedWhiteGreenCmap())
        plt.axis('off')
        plt.title(f'Value Function {title}')
        plt.show()

    ''' Visualise the Q-function with a matplotlib visual'''

    def visualiseQFunction(self, qValues, title, tileSize=32, showText=False):
        widthPx = self.width * tileSize
        heightPx = self.height * tileSize
        img = [[[0, 0, 0] for _ in range(widthPx)] for _ in range(heightPx)]
        rewardMax = max(self.getGoalStates().values())  # provide these to scale the colours between the highest and lowest value
        rewardMin = min(self.getGoalStates().values())
        # Render the grid
        for y in range(0, self.height):
            for x in range(0, self.width):
                # draw in the blocked states as a black and white mesh
                if (x, y) in self.blockedStates:
                    renderFullBlockedTile(x * tileSize, y * tileSize, tileSize, img)
                    continue
                # draw goal states
                if (x, y) in self.goalStates:
                    renderFullGoalTile(x * tileSize, y * tileSize, tileSize, img, reward=self.goalStates[(x, y)], rewardMax=rewardMax, rewardMin=rewardMin)
                    continue

                # draw the action value for action available in each cell
                # Break the grid up into 4 sections, using triangles that meet in the middle. The base of the triangle points toward the direction of the action
                renderActionQValue(tileSize, x, y, self.UP, qValues, img, showText, v_text_offset=8,
                                   rewardMax=rewardMax, rewardMin=rewardMin)
                renderActionQValue(tileSize, x, y, self.DOWN, qValues, img, showText, v_text_offset=-8,
                                   rewardMax=rewardMax, rewardMin=rewardMin)
                renderActionQValue(tileSize, x, y, self.LEFT, qValues, img, showText, h_text_offset=-8,
                                   rewardMax=rewardMax, rewardMin=rewardMin)
                renderActionQValue(tileSize, x, y, self.RIGHT, qValues, img, showText, h_text_offset=8,
                                   rewardMax=rewardMax, rewardMin=rewardMin)

        plt.imshow(img, origin='lower', interpolation='bilinear')
        plt.title(f'Q Function: {title}')
        plt.axis('off')
        plt.show()

    ''' Visualise the policy of the agent with a matplotlib visual '''
    def visualisePolicy(self, policy, title):
        # make values negative -1 to get a white background with the 'Greys' cmap matplotlib. Values have no actual
        # meaning in this visualisation.
        result = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    plt.text(x, y, '\u2592', horizontalalignment='center', verticalalignment='center')
                else:
                    action = "T" if policy[(x, y)] == self.TERMINATE else policy[(x, y)]
                    plt.text(x, y, action, horizontalalignment='center', verticalalignment='center')
        plt.imshow(result, cmap='Greys', origin='lower')
        ax = plt.gca()

        # set the ticks to get a clear grid lines
        ax.set_xticks([i for i in range(self.width)])
        ax.set_yticks([j for j in range(self.height)])
        ax.set_xticks([i+0.5 for i in range(self.width)], minor=True)
        ax.set_yticks([j+0.5 for j in range(self.height)], minor=True)

        ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
        plt.title(f'Policy: {title}')
        # plt.axis('off')
        plt.show()

    def execute(self, state, action):
        if state in self.goalStates:
            return MDP.execute(self, state=state, action=self.TERMINATE)
        return super().execute(state, action)

class CliffWorld(GridWorld):
    def __init__(self, noise = 0.0, discountFactor = 1.0, width = 6, height = 4,
                 blockedStates = [], actionCost = -0.05,
                 goals = [((1,0), -5), ((2,0), -5), ((3,0), -5), ((4,0), -5), ((5,0), 0)]):
        super().__init__(noise = noise, discountFactor = discountFactor,
                         width = width, height = height,
                         blockedStates = blockedStates, actionCost = actionCost, goals = goals)


class OneDimensionalGridWorld(GridWorld):
    """
    A one dimensional GridWorld class to use with the Logistic regression policy gradient.
    This allows actions [left, right] and terminates when the agent reaches the goal state without having to use a
    terminate action.
    """

    def __init__(self, noise=0.1, width=4, discountFactor=0.9, actionCost=0.0, initialState=(0, 0),
                 goals=[((0, 0), -1), ((10, 0), 1)]):
        super().__init__(noise=noise, width=width, height=1, blockedStates=[], discountFactor=discountFactor,
                         actionCost=actionCost,
                         initialState=initialState, goals=goals)

    def execute(self, state, action):
        # if we are in a goal state then terminate automatically execute a terminate action to immediately terminate
        if state in self.goalStates:
            return MDP.execute(self, state=state, action=self.TERMINATE)
        return super().execute(state, action)

    def visualise_policy_probabilities(self, agent, tileSize=32, title='Action Probabilities'):
        widthPx = self.width * tileSize
        heightPx = 1 * tileSize
        img = [[[0, 0, 0] for _ in range(widthPx)] for _ in range(heightPx)]

        # Render the grid
        for x in range(0, self.width):
            prob_left, prob_right = agent.get_probabilities((x, 0))
            if (x, 0) in self.goalStates:
                self.renderTile(x, 0, tileSize, img, 'goal')
            else:
                self.renderTile(x, 0, tileSize, img, 'empty')
                renderActionProbability(tileSize, x, 0, self.LEFT, prob_left, h_text_offset=-8)
                renderActionProbability(tileSize, x, 0, self.RIGHT, prob_right, h_text_offset=8)

        plt.imshow(img, origin='lower', interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Grid World {title}')
        plt.show()


if __name__ == "__main__":
    small = GridWorld(width = 8, height = 6)
    print(small.visualise_small())
    small.visualiseImage(title="Small")

    medium = gridworld = GridWorld(width = 16, height = 12)
    print(medium.visualise_small())
    medium.visualiseImage(title="Medium")
