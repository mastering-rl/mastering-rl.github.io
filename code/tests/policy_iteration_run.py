from gridworld import GridWorld
from policy_iteration import PolicyIteration
from tabular_policy import TabularPolicy


for iterations in [100]: #[0, 1, 2, 3, 4, 5, 10, 100]:
    mdp = GridWorld()
    policy = TabularPolicy(default_action=mdp.get_actions()[0])
    num_iterations = PolicyIteration(mdp, policy).policy_iteration(iterations=iterations)

    print("After iteration " + str(num_iterations))
    mdp.visualise_policy(policy)

#mdp = GridWorld(width=20, height=15)
#policy = TabularPolicy(default_action=mdp.get_actions()[0])
#PolicyIteration(mdp, policy).policy_iteration(iterations=100)
#mdp.visualise_policy(policy)
