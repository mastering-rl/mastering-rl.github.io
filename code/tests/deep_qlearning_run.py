from gridworld import GridWorld
from qlearning import QLearning
from deep_qfunction import DeepQFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = GridWorld()
qfunction = DeepQFunction(mdp=mdp, state_space=len(mdp.get_initial_state()), action_space=5, hiddem_dim=16)
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
policy = qfunction.extract_policy(mdp)
print(mdp.q_function_to_string(qfunction))
mdp.visualise_q_function_as_image(qfunction)
print(mdp.policy_to_string(policy))