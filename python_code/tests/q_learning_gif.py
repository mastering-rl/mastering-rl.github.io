from gridworld import GridWorld
from gif_maker import GifMaker
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, title="Q function Gif", grid_size=2.0)
qfunction = QTable()
for iterations in [1, 2, 3, 4, 5, 10, 100]:
    QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=iterations)
    image_texts = gridworld.visualise_q_function(qfunction, "After %d iterations" % (iterations), grid_size=2.0, gif=True)
    gif_maker.add_frame(image_texts)

# Add policy
qfunction = QTable()
QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=100)
policy = qfunction.extract_policy(gridworld)
image_texts = gridworld.visualise_policy(policy, gif=True)
gif_maker.add_frame(image_texts)

gif_maker.save("../../assets/gifs/q_function.gif")
