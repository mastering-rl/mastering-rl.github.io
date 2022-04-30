from gridworld import GridWorld
from gif_maker import GifMaker
from qtable import QTable
from single_agent_mcts import SingleAgentMCTS
from multi_armed_bandit.ucb import UpperConfidenceBounds

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, grid_size=2.0)
qfunction = QTable()
root_node = None
for time in range(1, 101):
    root_node = SingleAgentMCTS(gridworld, qfunction, UpperConfidenceBounds()).mcts(timeout=0.01, root_node=root_node)
    title = "Time = {:.2f}s".format(time/100)
    image_texts = gridworld.visualise_q_function(qfunction, title=title, grid_size=2.0, gif=True)
    gif_maker.add_frame(image_texts, title=title)

gif_maker.save("../../assets/gifs/mcts.gif")
