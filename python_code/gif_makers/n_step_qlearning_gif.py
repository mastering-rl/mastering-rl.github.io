from gridworld import GridWorld
from gif_maker import GifMaker
from qtable import QTable
from qlearning import QLearning
from n_step_qlearning import NStepQLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

gridworld1 = GridWorld()
gif_maker1 = GifMaker(mdp=gridworld1, grid_size=2.0)
qfunction1 = QTable()
for iteration in range(1, 21):
    QLearning(gridworld1, EpsilonGreedy(), qfunction1).execute(episodes=1)
    title1 = "1-Step Q-Learning after %d iterations" % (iteration)
    image_texts1 = gridworld1.visualise_q_function(qfunction1, title=title1, grid_size=2.0, gif=True)
    gif_maker1.add_frame(image_texts1, title=title1)

gif_maker1.save("../../assets/gifs/1_step_qlearning.gif")

gridworld2 = GridWorld()
gif_maker2 = GifMaker(mdp=gridworld2, grid_size=2.0)
qfunction2 = QTable()
for iteration in range(1, 21):
    NStepQLearning(gridworld2, EpsilonGreedy(), qfunction2, 5).execute(episodes=1)
    title2 = "5-Step Q-Learning after %d iterations" % (iteration)
    image_texts2 = gridworld2.visualise_q_function(qfunction2, title=title2, grid_size=2.0, gif=True)
    gif_maker2.add_frame(image_texts2, title=title2)

gif_maker2.save("../../assets/gifs/5_step_qlearning.gif")

import imageio
import numpy as np

#Create reader object for the gif
gif1 = imageio.get_reader("../../assets/gifs/1_step_qlearning.gif")
gif2 = imageio.get_reader("../../assets/gifs/5_step_qlearning.gif")

#If they don't have the same number of frame take the shorter
number_of_frames = min(gif1.get_length(), gif2.get_length())

#Create writer object
new_gif = imageio.get_writer("../../assets/gifs/1_step_vs_5_step_qlearning.gif")

for frame_number in range(number_of_frames):
    img1 = gif1.get_next_data()
    img2 = gif2.get_next_data()
    # Connect the two images
    new_image = np.hstack((img1, img2))
    new_gif.append_data(new_image)

gif1.close()
gif2.close()
new_gif.close()
