
class Policy(): pass

class DeterministicPolicy(Policy):
    def selectAction(state): abstract
    def update(state, action, something): abstract

import random

class TabularPolicy(DeterministicPolicy):
    def __init__(self, mdp, initialPolicy = None):
        self.mdp = mdp
        if initialPolicy is not None: 
            self.policyTable = initialPolicy
        else:
            self.policyTable = self.initialise()

    def initialise(self):
        action = random.choice(self.mdp.getActions())
        return defaultdict(lambda: action)

    def selectAction(self, state):
        return self.policyTable[state]

    def update(self, state, action):
        self.policyTable[state] = action

from collections import defaultdict

class PolicyIteration():

    def __init__(self, mdp):
        self.mdp = mdp

    ''' Calculate Q(s,a) of an action '''
    def qValue(self, state, action, values):
        newValue = 0.0
        for (newState, probability) in mdp.getTransitions(state, action):
            reward = mdp.getReward(state, action, newState)
            newValue += probability * (reward + (mdp.getDiscountFactor() * values[newState]))
        return newValue

    ''' Implmentation of policy iteration iteration '''
    def policyEvaluation(self, policy, values, theta = 0.001):

        while True:
            delta = 0.0
            newValues = dict()
            for state in mdp.getStates():
                # Calculate the value of V(s)
                oldValue = values[state]
                values[state] = self.qValue(state, policy.selectAction(state), values)
                delta = max(delta, abs(oldValue - values[state]))
            
            # terminate if the value function has converged
            if delta < theta:
                break

        return values


    ''' Implmentation of policy iteration iteration '''
    def policyIteration(self, iterations = 100, theta = 0.001):

        # Initialise the value function V and policy function pi
        values = self.initialiseValueFunction()
        policy = TabularPolicy(self.mdp)
        
        for i in range(iterations):
            policyChanged = False
            values = self.policyEvaluation(policy, values, theta)
            for state in mdp.getStates():
                oldAction = policy.selectAction(state)
                
                qValues = dict()
                for action in mdp.getActions(state):
                    # Calculate the value of Q(s,a)
                    newValue = self.qValue(state, action, values)
                    qValues.update({action: newValue})

                # V(s) = argmax_a Q(s,a)
                newAction = max(qValues, key = lambda i: qValues[i])
                policy.update(state, newAction)

                policyChanged = True if newAction is not oldAction else policyChanged

            if not policyChanged:
                print("iterations = %d" % i)
                break
                
        return policy

    def initialiseValueFunction(self):
        return defaultdict(lambda: 0.0)
        

if __name__ == "__main__":
    from gridworld import GridWorld
    mdp = GridWorld(width = 8, height = 6)
    policyIteration = PolicyIteration(mdp)

    for iterations in [0, 1, 2, 3, 4, 5, 10, 100]:
        print("After iteration " + str(iterations))
        print(mdp.policyToString(policyIteration.policyIteration(iterations = iterations).policyTable) + "\n")

    mdp = GridWorld(width = 20, height = 15)
    policyIteration = PolicyIteration(mdp)
    print(mdp.policyToString(policyIteration.policyIteration(iterations = 100).policyTable) + "\n")
