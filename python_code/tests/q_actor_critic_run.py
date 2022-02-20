from python_code.qlearning import QLearning
from python_code.deep_qfunction import DeepQFunction
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.gridworld import GridWorld
from python_code.deep_nn_policy import DeepNeuralNetworkPolicy
from python_code.policy_gradient import PolicyGradient
from python_code.q_actor_critic import QActorCritic

gridworld = GridWorld()

# Instantiate the critic
qfunction = DeepQFunction(gridworld, state_space=len(gridworld.get_initial_state()), action_space=5, hiddem_dim=16)
critic = QLearning(gridworld, EpsilonGreedy(), qfunction, alpha=1.0)
critic.execute(1000)

# Instantiate the actor
policy = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)
actor = PolicyGradient(gridworld, policy, alpha=0.1)

#  Instantiate the actor critic agent
q_actor_critic = QActorCritic(mdp=gridworld, actor=actor, critic=critic)

# Visualise current policy and q-function
gridworld.visualise_q_function_as_image(qfunction)
gridworld.visualise_stochastic_policy(policy)

for i in range(10):
    # execute 100 episodes
    q_actor_critic.execute(100)

    # Visualise current policy and q-function
    gridworld.visualise_q_function_as_image(qfunction)
    gridworld.visualise_stochastic_policy(policy)
    gridworld.visualise_policy_as_image(policy)