from gridworld import GridWorld
from gif_saver import GifPlayer
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction
import matplotlib.animation as animation
import matplotlib.pyplot as plt


#
gridworld = GridWorld()
imgs = []
fig, ax, grid = gridworld.visualise(gif=True)
for iterations in [0, 1, 2, 3, 4, 5, 10, 100]:
    values = TabularValueFunction()
    ValueIteration(gridworld, values).value_iteration(max_iterations=iterations)
    texts = gridworld.visualise_value_function(values, "After %d iterations" % (iterations), gif=True, grid=grid, ax=ax, fig=fig)
    imgs.append([grid] + texts)

ani = animation.ArtistAnimation(fig, imgs, interval=500, blit=False,
                                repeat_delay=2000)
ani.save("movie.gif")
