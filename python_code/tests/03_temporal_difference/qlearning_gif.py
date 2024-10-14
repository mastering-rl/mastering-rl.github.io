from python_code.gif_makers.gif_maker import GifMaker
from python_code.learners.qlearning import QLearning
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.qfunctions.qtable import QTable

grid_size = 1.5
gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, grid_size=grid_size)
qfunction = QTable()
for episode in range(1, 101):
    QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=1)
    title = "Episode %d" % (episode)
    image_texts = gridworld.visualise_q_function(
        qfunction, title=title, grid_size=grid_size, gif=True
    )
    gif_maker.add_frame(image_texts, title=title)

gif_maker.save("assets/gifs/qlearning.gif")
