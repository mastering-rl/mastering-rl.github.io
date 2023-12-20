from freeway_abstraction import FreewayAbstraction
from freeway import Freeway
from qtable import QTable
from reward_shaped_qlearning import RewardShapedQLearning
from qlearning import QLearning
from freeway_potential_function import FreewayPotentialFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.softmax import Softmax
from stochastic_q_policy import StochasticQPolicy
from tests.plot import Plot
from deep_qfunction import DeepQFunction
from ale_wrapper import ALEWrapper
from freeway_hand_policy import FreewayHandPolicy

version="Freeway-ramDeterministic-v4"
#version="ALE/Frogger-ram-v5"
print("==========\nQ-learning: " + version + "\n==========")

#mdp = FreewayAbstraction(render_mode="rgb_array", discount_factor=0.99)
#mdp = Freeway(render_mode="rgb_array", discount_factor=0.99)

mdp = ALEWrapper(version=version, render_mode="rgb_array", discount_factor=0.95)

qfunction = DeepQFunction(mdp, state_space=len(mdp.get_initial_state()), action_space=len(mdp.get_actions()), hidden_dim=20)
#qfunction = QTable()
#RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=1)
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1)
policy = StochasticQPolicy(qfunction, mdp.get_actions(), EpsilonGreedy(epsilon=0.0))
rewards = mdp.execute_policy(policy, episodes=1)

episodes = 50
episodes_per_evaluation = 1
for i in range(int(episodes / episodes_per_evaluation)):
    print("Episode %d" % (i * episodes_per_evaluation))
    #RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=episodes_per_evaluation)
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=episodes_per_evaluation)
    rewards += mdp.execute_policy(policy, episodes=1)

mdp = ALEWrapper(version=version, render_mode="rgb_array", discount_factor=0.95)

potential = FreewayPotentialFunction(mdp)

qfunction = DeepQFunction(mdp, state_space=len(mdp.get_initial_state()), action_space=len(mdp.get_actions()), hidden_dim=20)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=1)
policy = StochasticQPolicy(qfunction, mdp.get_actions(), EpsilonGreedy(epsilon=0.0))
shaped = mdp.execute_policy(policy, episodes=1)
for i in range(int(episodes / episodes_per_evaluation)):
    print("Episode %d" % (i * episodes_per_evaluation))
    RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=episodes_per_evaluation)
    shaped += mdp.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(
    ["Q-learning", "Reward-shaped Q-learning"], [rewards, shaped], smoothing_factor=0.9, episodes_per_evaluation=episodes_per_evaluation
)



#qfunction.save("frogger.policy")

#qfunction.load("frogger.policy")

policy = StochasticQPolicy(qfunction, mdp.get_actions(), EpsilonGreedy(epsilon=0.0))

#policy = FreewayHandPolicy()
#mdp = Freeway(render_mode="human")

mdp = ALEWrapper(version=version, render_mode="human")
exec_rewards = mdp.execute_policy(policy, episodes=1)
print(exec_rewards)