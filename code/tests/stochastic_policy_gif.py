from gridworld import GridWorld
from gif_maker import GifMaker
from policy_gradient import PolicyGradient
from deep_nn_policy import DeepNeuralNetworkPolicy

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, title="Q function Gif", grid_size=2.0)
policy = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)
for iterations in [10, 20, 30, 40, 50, 100, 1000]:
    policy_gradient = PolicyGradient(gridworld, policy, alpha=0.1).execute(episodes=iterations)
    image_texts = gridworld.visualise_stochastic_policy(policy, gif=True)
    gif_maker.add_frame(image_texts)


# policy = policy.extract_policy(gridworld)
image_texts = gridworld.visualise_policy(policy, gif=True)
gif_maker.add_frame(image_texts)

gif_maker.save("../../assets/gifs/stochastic_policy.gif")
