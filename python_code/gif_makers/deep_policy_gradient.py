from python_code.gif_makers.gif_maker import GifMaker

from python_code.learners.policy_gradient import PolicyGradient
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.policies.deep_nn_policy import DeepNeuralNetworkPolicy

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld)
policy = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)
for episode in range(0, 100):
    title = "Episode %d" % (episode)
    image_texts = gridworld.visualise_stochastic_policy(policy, title=title, gif=True)
    gif_maker.add_frame(image_texts, title=title)
    PolicyGradient(gridworld, policy, alpha=0.1).execute(episodes=1)

gif_maker.save("../../assets/gifs/deep_policy_gradient.gif")
