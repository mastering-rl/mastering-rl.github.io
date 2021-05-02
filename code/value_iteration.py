from gridworld import *

class ValueIteration():

    def __init__(self, mdp):
        self.mdp = mdp

    ''' Implmentation of value iteration '''
    def valueIteration(self, iterations = 100, theta = 0.001):

        # Initialise the value function V with all 0s
        values = self.initialiseValueFunction()
        for i in range(iterations):

           delta = 0.0
           for state in mdp.getStates():
               qValues = dict()
               for action in mdp.getActions(state):
                   # Calculate the value of Q(s,a)
                   newValue = 0.0
                   for (newState, probability) in mdp.getTransitions(state, action):
                       reward = mdp.getReward(state, action, newState)
                       newValue += probability * (reward + (mdp.getDiscountFactor() * values[newState]))
                   qValues.update({action: newValue})

               # V(s) = max_a Q(s,a)
               maxQ = max(qValues.values())
               delta = max(delta, abs(values[state] - maxQ))
               values.update({state: maxQ})

           # terminate if the value function has converged
           if delta < theta:
               print("%d iterations" % i)
               break

        return values

    def initialiseValueFunction(self):
        values = dict()
        for state in self.mdp.getStates():
            values.update({state: 0.0})
        return values

if __name__ == "__main__":
    #mdp = GridWorld(discountFactor=0.9, width = 16, height = 12)
    mdp = GridWorld(discountFactor=0.9, width = 40, height = 24, goals = [((35,20), 1), ((35,19), -1)])
    valueIteration = ValueIteration(mdp)

    for iterations in []: #[1, 2, 3, 4, 5, 10, 100]:
        print("After iteration " + str(iterations))
        print(mdp.valueFunctionToString(valueIteration.valueIteration(iterations = iterations)) + "\n")

    print("Policy after 100 iterations")
    import time
    start = time.time_ns()
    values = valueIteration.valueIteration(iterations = 100000000)
    finish = time.time_ns()
    policy = mdp.extractPolicyFromValueFunction(values)
    print(mdp.policyToString(policy))
    print("Value iteration execution time = %f" % ((finish - start) / 1000000))

