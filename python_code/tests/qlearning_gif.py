from gridworld import GridWorld
from gif_maker import GifMaker
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

saves=[1,2,3,4,5,10,50,100,200,300,400,500,600,700,800,900,1000]

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, grid_size=2.0)
qfunction = QTable()
for iteration in range(1, 1001):
    QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=1)
    if iteration in saves:
        title = "After %d iterations" % (iteration)
        image_texts = gridworld.visualise_q_function(qfunction, title=title, grid_size=2.0, gif=True)
        gif_maker.add_frame(image_texts, title=title)

gif_maker.save("../../assets/gifs/gridworld_qfunction.gif")
