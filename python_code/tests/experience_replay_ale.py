import torch

from experience_replay_learner import ExperienceReplayLearner
from deep_q_network import DeepQFunction
from stochastic_q_policy import StochasticQPolicy
from ale_wrapper import ALEWrapper
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from tests.plot import Plot


# if GPU is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
version = 'CartPole-v1'
#policy_name = 'Cartpole.policy'
#version = "ALE/VideoCheckers-ram-v5"
#policy_name = "video_checkers.policy"
#version = "BankHeist-ramNoFrameskip-v4"
#version = "ALE/BankHeist-ram-v5"
#policy_name = "bankheist.policy"

mdp = ALEWrapper(version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 5
episodes = 300
all_rewards = []
for _ in range(runs):
    policy_qfunction = DeepQFunction(state_space, action_space)
    target_qfunction = DeepQFunction(state_space, action_space)
    learner = ExperienceReplayLearner(mdp, EpsilonDecreasing(), policy_qfunction, target_qfunction, update_period=1)

    rewards = learner.execute(episodes)
    all_rewards.append(rewards)

labels = ["Deep Experience Replay Q learner " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.0)

policy = StochasticQPolicy(policy_qfunction, EpsilonDecreasing(epsilon=0.04, alpha=1., lower_bound=0.0))

mdp = ALEWrapper(version=version, render_mode="human")
exec_rewards = mdp.execute_policy(policy, episodes=1)
