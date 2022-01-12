from gridworld import GridWorld
from gif_maker import GifMaker
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld)
values = TabularValueFunction()
for iterations in range(1, 101):
    ValueIteration(gridworld, values).value_iteration(max_iterations=1)
    title = "Iterations %d" % (iterations)
    image_texts = gridworld.visualise_value_function(values, title=title, gif=True)
    gif_maker.add_frame(image_texts, title=title)

gif_maker.save("../../assets/gifs/value_iteration.gif")
