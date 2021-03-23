from gridworld import *

class ValueIteration():

    def __init__(self, mdp):
        self.mdp = mdp

    ''' Implmentation of value iteration '''
    def valueIteration(self, iterations = 100, theta = 0.001):

        # Initialise the value function V with all 0s
        values = self.initialiseValueFunction()
        for _ in range(iterations):

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
               break

        return values

    def initialiseValueFunction(self):
        values = dict()
        for state in self.mdp.getStates():
            values.update({state: 0.0})
        return values


mdp = GridWorld(discountFactor=0.9, width = 4, height = 3)
valueIteration = ValueIteration(mdp)


for iterations in [1, 2, 3, 4, 5, 10, 100]:
    print("After iteration " + str(iterations))
    print(mdp.valueFunctionToString(valueIteration.valueIteration(iterations = iterations)) + "\n")

print("Policy after 100 iterations")
print(mdp.policyToString(mdp.extractPolicyFromValueFunction(valueIteration.valueIteration(iterations = 100))))


