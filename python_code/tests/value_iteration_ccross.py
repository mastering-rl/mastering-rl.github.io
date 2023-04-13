from contested_crossing import ContestedCrossing
from value_iteration import ValueIteration
from tabular_value_function import TabularValueFunction

mdp = ContestedCrossing()
values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(max_iterations=10)
enemy_health = direction = 1
for x in [1,2]:
    for y in [1,2]:
        for ship_health in [1,2]:
            print("state: {0} - value: {1}".format((x,y, ship_health, enemy_health, direction),
                                                   round(values.value_table[(x, y, ship_health, enemy_health, direction)], 3)))

for iterations in [10]:
    values = TabularValueFunction()
    ValueIteration(mdp, values).value_iteration(max_iterations=iterations)
    mdp.visualise_value_function(values, "After %d iterations" % (iterations), mode=3)

values = TabularValueFunction()
ValueIteration(mdp, values).value_iteration(max_iterations=2)
policy = values.extract_policy(mdp)
mdp.visualise_policy(policy, "Policy plot after 100 iterations", mode=0)
mdp.visualise_policy(policy, "Path Plot after 100 iterations", mode=1)
