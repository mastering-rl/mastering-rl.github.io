from python_code.deep_nn_policy import DeepNeuralNetworkPolicy
from python_code.deep_q_actor_critic import DeepQActorCritic
from python_code.deep_qfunction import DeepQFunction
from python_code.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.qlearning import QLearning

gridworld = GridWorld()

# Instantiate the critic
qfunction = DeepQFunction(gridworld, state_space=len(gridworld.get_initial_state()), action_space=5, hiddem_dim=16)
critic = QLearning(gridworld, EpsilonGreedy(), qfunction, alpha=1.0)

# Instantiate the actor
actor = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)

#  Instantiate the actor critic agent
deep_q_actor_critic = DeepQActorCritic(mdp=gridworld, actor=actor, critic=critic)

gridworld.visualise_q_function_as_image(qfunction, title=f"Q Function: {0} iterations")
gridworld.visualise_stochastic_policy(actor)
gridworld.visualise_policy_as_image(actor)

deep_q_actor_critic.execute(100)
gridworld.visualise_q_function_as_image(qfunction, title=f"Q Function: {100} iterations")
gridworld.visualise_stochastic_policy(actor)
gridworld.visualise_policy_as_image(actor)

deep_q_actor_critic.execute(1000)
gridworld.visualise_q_function_as_image(qfunction, title=f"Q Function: {1000} iterations")
gridworld.visualise_stochastic_policy(actor)
gridworld.visualise_policy_as_image(actor)
