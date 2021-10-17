from gridworld import GridWorld
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction


mdp = GridWorld()
mdp.visualise()

for iterations in [0, 1, 2, 3, 4, 5, 10, 100]:
    values = TabularValueFunction()
    ValueIteration(mdp, values).value_iteration(max_iterations=iterations)
    mdp.visualise_value_function(values, "After %d iterations" % (iterations))

values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(max_iterations=100)
policy = values.extract_policy(mdp)
mdp.visualise_policy(policy, "Policy after 100 iterations")

mdp = GridWorld.open('layouts/room.txt')
mdp.visualise(grid_size=0.8)
values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(max_iterations=100)
mdp.visualise_value_function(values, grid_size=0.8, title="100 iterations")
policy = values.extract_policy(mdp)
mdp.visualise_policy(policy, "", grid_size=0.8)

