from gridworld import GridWorld
from gif_maker import GifMaker
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, title="Value Iteration Gif")
for iterations in [0, 1, 2, 3, 4, 5, 10, 100]:
    values = TabularValueFunction()
    ValueIteration(gridworld, values).value_iteration(max_iterations=iterations)
    image_texts = gridworld.visualise_value_function(values, "After %d iterations" % (iterations), gif=True)
    gif_maker.add_frame(image_texts)

# Add policy
values = TabularValueFunction()
ValueIteration(gridworld, values).value_iteration(max_iterations=100)
policy = values.extract_policy(gridworld)
image_texts = gridworld.visualise_policy(policy, gif=True)
gif_maker.add_frame(image_texts)

gif_maker.save("../../assets/gifs/value_iteration.gif")
