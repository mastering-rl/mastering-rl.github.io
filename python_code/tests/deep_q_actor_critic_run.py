from deep_nn_policy import DeepNeuralNetworkPolicy
from q_actor_critic import QActorCritic
from deep_qfunction import DeepQFunction
from deep_q_network import DQN
from gridworld import GridWorld
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from qlearning import QLearning

gridworld = GridWorld()

# Instantiate the critic
qfunction = DQN(
    state_space=len(gridworld.get_initial_state()),
    action_space=5,
    hidden_dim=16,
)
critic = QLearning(gridworld, EpsilonGreedy(), qfunction)

# Instantiate the actor
actor = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)

#  Instantiate the actor critic agent
q_actor_critic = QActorCritic(mdp=gridworld, actor=actor, critic=critic).execute(
    episodes=300
)
gridworld.visualise_stochastic_policy(actor)
gridworld.visualise_q_function(critic.qfunction)
