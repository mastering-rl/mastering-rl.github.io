from gridworld import GridWorld
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction


mdp = GridWorld()
mdp.visualise_image()


for iterations in [1, 2, 3, 4, 5, 10, 100]:
    values = TabularValueFunction()
    ValueIteration(mdp, values).value_iteration(iterations=iterations)
    print("After iteration " + str(iterations))
    print(mdp.value_function_to_string(values) + "\n")


values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(iterations=100)
mdp.visualise_value_function(values, title="100 iterations")
policy = values.extract_policy(mdp)
print("Policy after 100 iterations")
print(mdp.policy_to_string(policy))
mdp.visualise_policy(policy, "")


mdp = GridWorld.open('layouts/room.txt')
mdp.visualise_image(grid_size=0.8)
values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(iterations=100)
mdp.visualise_value_function(values, grid_size=0.8, title="100 iterations")
policy = values.extract_policy(mdp)
mdp.visualise_policy(policy, "", grid_size=0.8)
