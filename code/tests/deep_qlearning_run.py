from gridworld import GridWorld
from qlearning import QLearning
from deep_qfunction import DeepQFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = GridWorld(width=4,
                height=3,
                goals=dict(
                    [((3, 2), 1), ((3, 1), -1)]
                ))
qfunction = DeepQFunction(mdp=mdp, state_space=len(mdp.get_initial_state()), action_space=5, hiddem_dim=16)
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=10000)
mdp.visualise_q_function_as_image(qfunction)
policy = qfunction.extract_policy(mdp)
mdp.visualise_policy_as_image(policy)
