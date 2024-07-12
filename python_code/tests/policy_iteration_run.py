from python_code.learners.policy_iteration import PolicyIteration
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.policies.tabular_policy import TabularPolicy

mdp = GridWorld(width=20, height=15)
policy = TabularPolicy(default_action=mdp.get_actions()[0])
iterations = PolicyIteration(mdp, policy).policy_iteration(max_iterations=100)
print("Number of iterations until convergence: %d" % (iterations))

mdp.visualise_policy(policy)
