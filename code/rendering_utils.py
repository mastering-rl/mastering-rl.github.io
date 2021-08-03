import math
import matplotlib.colors as colours
from mdp import *
import matplotlib.pyplot as plt

COLOURS = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'purple': [112, 39, 195],
    'grey': [100, 100, 100],
    'white': [255, 255, 255],
    'black': [0, 0, 0],
    'yellow': [255, 255, 0]
}

# Action symbols from gridworld
LEFT = '\u25C4'
UP = '\u25B2'
RIGHT = '\u25BA'
DOWN = '\u25BC'

'''Draw the grid lines to differentiate discrete states'''
def drawGridLines(i, j, img):
    img[i][j] = COLOURS['grey']


'''Draw a triangle based on size, center, direction and colour'''
def drawTriangle(tile_origin, tile_size, img, colour='red', direction='up'):
    origin_x, origin_y = tile_origin
    for x in range(origin_x + 1, origin_x + tile_size - 1):
        for y in range(origin_y + 1, origin_y + tile_size -1):
            if direction == DOWN:
                if y < origin_y + tile_size//2 and x + y < origin_x + origin_y + tile_size and x - origin_x > y - origin_y:
                    img[y][x] = colour
            elif direction == UP:
                if y > origin_y + tile_size // 2 and x + y > origin_x + origin_y + tile_size and x - origin_x < y - origin_y:
                    img[y][x] = colour
            elif direction == LEFT:
                if x < origin_x + tile_size // 2 and x - origin_x < y - origin_y and x + y < origin_x + origin_y + tile_size:
                    img[y][x] = colour
            elif direction == RIGHT:
                if x > origin_x + tile_size // 2 and x - origin_x > y - origin_y and x + y > origin_x + origin_y + tile_size:
                    img[y][x] = colour
            else:
                raise ValueError("Invalid direction")


''' Render each Q value which forms a triangle in the grid representation of the Q-function.'''
def renderActionQValue(tileSize, x, y, action, qValues, img, showText=False, text_size=12, h_text_offset=0, v_text_offset=0, rewardMax=1, rewardMin=1):
    value = MDP.getQValue(qValues, (x, y), action=action)
    colour = COLOURS['red'] if value < 0 else COLOURS['green']  # make colour red if value is negative, otherwise make it green
    scalingFactor = rewardMin if value < 0 else rewardMax
    colour = list(map(lambda c: int(c * math.fabs(value/scalingFactor)),
                      colour))  # scale the colour by the reward (make extremes more vivid)
    drawTriangle((x * tileSize, y * tileSize), tileSize, img, colour=colour, direction=action)
    if showText:
        plt.text(x=x * tileSize + tileSize // 2 + h_text_offset, y=y * tileSize + tileSize // 2 + v_text_offset,
                 s=f'{value:.2f}', size=text_size, verticalalignment='center', horizontalalignment='center', color='white')


''' Render each Q value which forms a triangle in the grid representation of the Q-function.'''
def renderActionProbability(tileSize, x, y, action, prob, text_size=6, h_text_offset=0, v_text_offset=0):
    plt.text(x=x * tileSize + tileSize // 2 + h_text_offset, y=y * tileSize + tileSize // 2 + v_text_offset,
             s=f'{prob:.2f}\n{action}', size=text_size, verticalalignment='center', horizontalalignment='center', color='white')


'''render blocked tile as a black and white criss-cross'''
def renderBlockedTile(i, j, img):
    if i % 2 == 0 or j % 2 == 0:
        img[i][j] = COLOURS['black']
    else:
        img[i][j] = COLOURS['white']


def renderFullBlockedTile(x, y, tile_size, img):
    for i in range(x, x+tile_size):
        for j in range(y, y+tile_size):
            if i % 2 == 0 or j % 2 == 0:
                img[j][i] = COLOURS['black']
            else:
                img[j][i] = COLOURS['white']


def renderFullGoalTile(x, y, tile_size, img, reward, rewardMax, rewardMin):
    for i in range(x, x+tile_size):
        for j in range(y, y+tile_size):
            if reward > 0:
                img[j][i] = [0, int(255 * reward / rewardMax), 0]
            else:
                img[j][i] = [int(255 * reward / rewardMin), 0, 0]


'''render the agent as a circle'''
def renderAgent(i, j, img, center_x, center_y, radius):
    h_dist = math.fabs(center_x - j)
    v_dist = math.fabs(center_y - i)
    if h_dist ** 2 + v_dist ** 2 <= radius ** 2:
        img[i][j] = COLOURS['yellow']
    else:
        img[i][j] = COLOURS['black']


'''
Render the goal as a coloured cell. THe color depend on the value of the goal.
Positive values are green, with brighter green representing higher reward.
Negative values are red, with brighter red representing lower reward. 
'''
def renderGoal(i, j, img, reward, reward_max=1, reward_min=-1):
    if reward > 0:
        img[i][j] = [0, int(255 * reward / reward_max), 0]
    else:
        img[i][j] = [int(255 * reward / reward_min), 0, 0]

'''
Matplotlib doesn't have an inbuilt red to green colour map with white in the middle.
So we can just make our own.
'''
def makeRedWhiteGreenCmap():
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (0.5, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 0.0, 0.0))
             }

    # Create the colormap using the dictionary
    return colours.LinearSegmentedColormap('GnRd', cdict)
