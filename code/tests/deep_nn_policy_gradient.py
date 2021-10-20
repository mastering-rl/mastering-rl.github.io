import time
from gridworld import GridWorld
from policy_gradient import PolicyGradient
from deep_nn_policy import DeepNeuralNetworkPolicy
from gif_player import GifPlayer

gridworld = GridWorld()
gif_player_stochastic = GifPlayer(mdp=gridworld, title="Deep NN stochastic policy")
gif_player_policy = GifPlayer(mdp=gridworld, title="Deep NN policy")

policy = DeepNeuralNetworkPolicy(gridworld, state_space=len(gridworld.get_initial_state()), action_space=4)
policy_gradient = PolicyGradient(gridworld, policy, alpha=0.1)

# run 10 iterations
for i in range(100):
    policy_gradient.execute(episodes=10)
    print("start time:", time.time())
    gif_player_policy.add_policy_frame(policy)
    gif_player_stochastic.add_stochastic_policy_frame(policy)
    print("end time:", time.time())

# print("hello world 3")
# # run 100 iterations
# policy_gradient.execute(episodes=50)
# fig, ax, img = gridworld.visualise_stochastic_policy_as_image(policy)
# gif_player.add_image(img)
#
gif_player_stochastic.show(block=True)
gif_player_policy.show(block=True)
