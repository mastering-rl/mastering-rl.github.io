import old_deep_q_function
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from old_deep_q_function import DeepQFunction

# from deep_q_function import DeepQFunction
import python_code.qfunctions.deep_q_function as deep_q_function
from python_code.learners.qlearning import QLearning
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.policies.q_policy import QPolicy

gridworld = GridWorld()
qfunction = deep_q_function.DeepQFunction(
    state_space=len(gridworld.get_initial_state()), action_space=5
)
import time

start = time.time()
new_rewards = QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=300)
end = time.time()
print("new time = " + str(end - start))
print(gridworld.q_function_to_string(qfunction))
# policy = QPolicy(qfunction)
# gridworld.visualise_policy_as_image(policy)

gridworld = GridWorld()
old_qfunction = old_deep_q_function.DeepQFunction(
    state_space=len(gridworld.get_initial_state()), action_space=5
)
start = time.time()
old_rewards = QLearning(gridworld, EpsilonGreedy(), old_qfunction).execute(episodes=300)
end = time.time()
print("old time = " + str(end - start))
print(gridworld.q_function_to_string(old_qfunction))
policy = QPolicy(old_qfunction)
# gridworld.visualise_policy_as_image(policy)

from tests.plot import Plot

labels = ["New", "Old"]
Plot.plot_cumulative_rewards(labels, [new_rewards, old_rewards], smoothing_factor=0.9)
