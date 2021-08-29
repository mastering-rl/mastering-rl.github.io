from gridworld import *
from vtable import *
from qtable import *

class ValueIteration():

    def __init__(self, mdp, values):
        self.mdp = mdp
        self.values = values

    ''' Implmentation of value iteration '''
    def valueIteration(self, iterations = 100, theta = 0.001):

        for i in range(iterations):
            delta = 0.0
            newValues = VTable()
            for state in mdp.getStates():
                qtable = QTable()
                for action in mdp.getActions(state):
                    # Calculate the value of Q(s,a)
                    newValue = 0.0
                    for (newState, probability) in mdp.getTransitions(state, action):
                        reward = mdp.getReward(state, action, newState)
                        newValue += probability * (reward + (mdp.getDiscountFactor() * self.values.getValue(newState)))
                    qtable.update(state, action, newValue)

                # V(s) = max_a Q(s,a)
                (_, maxQ) = qtable.getMaxQ(state, mdp.getActions(state))
                delta = max(delta, abs(self.values.getValue(state) - maxQ))
                newValues.update(state, maxQ)

            self.values.merge(newValues)
            
            # terminate if the value function has converged
            if delta < theta:
                break


if __name__ == "__main__":
    mdp = GridWorld()
    #mdp.visualiseImage()
    
    for iterations in [1, 2, 3, 4, 5, 10, 100]:
        values = VTable()
        valueIteration = ValueIteration(mdp, values)
        print("After iteration " + str(iterations))
        valueIteration.valueIteration(iterations = iterations)
        print(mdp.valueFunctionToString(values) + "\n")

    print("Policy after 100 iterations")
    values = VTable()
    valueIteration = ValueIteration(mdp, values)
    valueIteration.valueIteration(iterations = 100)
    mdp.visualiseValueFunction(values, title="100 iterations")
    policy = mdp.extractPolicyFromValueFunction(values)
    print(mdp.policyToString(policy))

