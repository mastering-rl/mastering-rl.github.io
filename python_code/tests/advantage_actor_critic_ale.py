import torch

from advantage_actor_critic import AdvantageActorCritic
from deep_nn_policy import DeepNeuralNetworkPolicy
from deep_value_function import DeepValueFunction
from ale_wrapper import ALEWrapper
from tests.plot import Plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
version = 'CartPole-v1'
policy_name = "CartPole.policy"

mdp = ALEWrapper(version=version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 1
episodes = 5
all_rewards = []
for _ in range(runs):

    # Instantiate the critic
    critic = DeepValueFunction(state_space, hidden_dim=16)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space)

    advantage_actor_critic = AdvantageActorCritic(mdp, actor, critic)
    rewards = advantage_actor_critic.execute(episodes, max_episode_length=500)

    all_rewards.append(rewards)

labels = ["Advantage actor critic "  + str(i) for i in range(runs)]
#Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.9)

actor.save(policy_name)
actor = DeepNeuralNetworkPolicy.load(state_space, action_space, policy_name)

import cv2

def create_video(source, fps=60, output_name='output'):
    out = cv2.VideoWriter(output_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (source[0].shape[1], source[0].shape[0]))
    for i in range(len(source)):
        out.write(source[i])
    out.release()

#env = gym.make('SSLGoToBall-v0')
mdp = ALEWrapper(version=version, render_mode="rgb_array")
state = mdp.reset()
frames = []
actor.stochastic = False
for i in range(1):
    done = False
    steps = 0
    while not done and steps < 1000:
        # Step using random actions
        action = actor.select_action(state, mdp.get_actions(state))
        next_state, reward, done = mdp.execute(state, action)
        state = next_state
        frames.append(mdp.render())
        steps += 1

create_video(frames, 60, 'output')
