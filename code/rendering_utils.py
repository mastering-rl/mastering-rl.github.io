import math
from enum import IntEnum

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

'''Draw the grid lines to differentiate discrete states'''
def drawGridLines(i, j, img):
    img[i][j] = COLOURS['grey']


'''render blocked tile as a black and white criss-cross'''
def renderBlockedTile(i, j, img):
    if i % 2 == 0 or j % 2 == 0:
        img[i][j] = COLOURS['black']
    else:
        img[i][j] = COLOURS['white']


'''render the agent as a circle'''
def renderAgent(i, j, img, center_x, center_y, radius):
    h_dist = math.fabs(center_x - i)
    v_dist = math.fabs(center_y - j)
    if math.sqrt(h_dist ** 2 + v_dist ** 2 <= radius):
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
