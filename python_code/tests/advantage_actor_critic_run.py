from python_code.deep_nn_policy import DeepNeuralNetworkPolicy
from python_code.policy_gradient import PolicyGradient
from python_code.gridworld import GridWorld
from python_code.value_iteration import ValueIteration
from python_code.tabular_value_function import TabularValueFunction
from python_code.advantage_actor_critic import AdvantageActorCritic

gridworld = GridWorld()

# Instantiate the critic
critic = TabularValueFunction()
ValueIteration(gridworld, critic).value_iteration(max_iterations=100)
gridworld.visualise_value_function(critic, grid_size=0.8, title="100 iterations")

# Instantiate the actor
policy = DeepNeuralNetworkPolicy(
    gridworld, state_space=len(gridworld.get_initial_state()), action_space=4
)
actor = PolicyGradient(gridworld, policy, alpha=0.1)

advantage_actor_critic = AdvantageActorCritic(mdp=gridworld, actor=actor, critic=critic)

advantage_actor_critic.execute(100)
gridworld.visualise_stochastic_policy(policy)
gridworld.visualise_policy_as_image(policy)

advantage_actor_critic.execute(1000)
gridworld.visualise_stochastic_policy(policy)
gridworld.visualise_policy_as_image(policy)

