from gridworld import GridWorld
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction


mdp = GridWorld()

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
