import random

class MDP:
    ''' Return all states of this MDP '''
    def getStates(self): abstract

    ''' Return all actions with non-zero probability from this state '''
    def getActions(self, state): abstract

    ''' Return all non-zero probability transitions for this action from this state '''
    def getTransitions(self, state, action): abstract

    ''' Return the reward for transitioning from state to nextState via action '''
    def getReward(self, state, action, nextState): abstract

    ''' Return true if and only if state is a terminal state of this MDP '''
    def isTerminal(self, state): abstract
    
    ''' Return the discount factor for this MDP '''
    def getDiscountFactor(self): abstract

    ''' Return the initial state of this MDP '''
    def getInitialState(self): abstract

    ''' Return all goal states of this MDP '''
    def getGoalStates(self): abstract


    ''' Return a policy given a value function '''
    def extractPolicyFromValueFunction(self, values):
        policy = dict()
        for state in mdp.getStates():
            maxQ = float('-inf')
            for action in mdp.getActions(state):
               # Calculate the value of Q(s,a)
               qValue = 0.0
               for (newState, probability) in mdp.getTransitions(state, action):
                   reward = mdp.getReward(state, action, newState)
                   qValue += probability * (reward + (mdp.getDiscountFactor() * values[newState]))

               # if this is the maximum Q-value so far, set the policy for this state
               if qValue > maxQ:
                   policy.update({state: action})
                   maxQ = qValue

        return policy

    ''' Return a policy given a Q function '''
    def extractPolicyFromQFunction(self, qValues):
        policy = dict()
        for state in mdp.getStates():
            # Get the QValues for this state only
            sQValues = dict(filter(lambda sa: sa[0][0] == state, qValues.items()))

            # Find the maximum Q-value
            maxQ = float('-inf')
            for (_, action) in sQValues:
               # if this is the maximum Q-value so far, set the policy for this state
               qValue = sQValues[(state, action)]
               if qValue > maxQ:
                   policy.update({state: action})
                   maxQ = qValue

        return policy

    ''' 
       Return a new state and a reward for executing action in state, 
       based on the underlying probability. This can be used for 
       model-free method
    '''
    def simulate(self, state, action):
        r = random.random()
        cumulativeProbability = 0.0
        for (newState, probability) in mdp.getTransitions(state, action):
            if r >= cumulativeProbability and r <= probability + cumulativeProbability:
                return (newState, self.getReward(state, action, newState))
            cumulativeProbability += probability
            if cumulativeProbability >= 1.0:
                raise "Cumulative probability >= 1.0 for action " + str(action) + " from " + str(state)
        print("No outcome state in simulation for action " + str(action) + " from " + str(state))
        raise "No outcome state in simulation for action"
        return None


class NavigationMDP(MDP):

    # labels for terminate action and terminal state
    TERMINATE = 'terminate'
    TERMINAL = ('terminal', 'terminal')

    def __init__(self, width = 4, height = 3,
                 discountFactor = 0.9, 
                 blockedStates = [(1,1)],
                 goals = [((3,2), 1), ((3,1), -1)]):
        self.width = width
        self.height = height
        self.blockedStates = blockedStates
        self.discountFactor = discountFactor
        self.goalStates = dict(goals)

    def getStates(self):
        states = [self.TERMINAL]
        for x in range(self.width):
            for y in range(self.height):
                if not (x, y) in self.blockedStates:
                    states.append((x,y))
        return states

    def getActions(self, state=None):

        if (state == None):
            return ['N', 'S', 'E', 'W', self.TERMINATE]

        actions = []
        for action in ['N', 'S', 'E', 'W', self.TERMINATE]:
            for (newState, probability) in self.getTransitions(state, action):
                if probability > 0:
                    actions.append(action)
                    break
        return actions

    def getInitialState(self):
        return (0,0)

    def getGoalStates(self):
        return self.goalStates

    def validAdd(self, state, newState, probability):
        # if the next state is blocked, stay in the same state
        if (newState in self.blockedStates):
            return [(state, probability)]

        # move to the next space if it is not off the grid
        (x, y) = newState
        if (x >= 0 and x < self.width and y >= 0 and y < self.height):
            return [((x, y), probability)]
 
        # if off the grid, state in the same state
        return [(state, probability)]

    def getTransitions(self, state, action):
        transitions = []

        if state == self.TERMINAL:
            return [(self.TERMINAL, 1.0)]

        (x, y) = state
        if state in self.getGoalStates().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]

        elif action == 'N':
            transitions += self.validAdd(state, (x, y + 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == 'S':
            transitions += self.validAdd(state, (x, y - 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == 'E':
            transitions += self.validAdd(state, (x + 1, y), 0.8)
            transitions += self.validAdd(state, (x, y - 1), 0.1)
            transitions += self.validAdd(state, (x, y + 1), 0.1)

        elif action == 'W':
            transitions += self.validAdd(state, (x - 1, y), 0.8)
            transitions += self.validAdd(state, (x, y - 1), 0.1)
            transitions += self.validAdd(state, (x, y + 1), 0.1)

        return transitions

    def getReward(self, state, action, newState):
       reward = 0.0
       if state in self.getGoalStates().keys() and newState == self.TERMINAL:
          reward = self.getGoalStates().get(state)
       return reward

    def getDiscountFactor(self):
        return self.discountFactor

    def isTerminal(self, state):
        if state == self.TERMINAL:
            return True
        return False

    ''' Convert a grid world value function to a formatted string '''
    def valueFunctionToString(self, values):
        line = " {:-^{n}}\n".format("", n=len(" | +0.00")*self.width + 1)
        result = line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " | #####"
                else:
                    result += " | {:+0.2f}".format(values[(x, y)])
            result += " |\n"
            result += line

        return result

    ''' Convert a grid world Q function to a formatted string '''
    def qFunctionToString(self, qValues):
        space = " |               "

        line = "  "
        for x in range(self.width):
            line += "---------------- "
        line += "\n"
        
        result = line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in mdp.getGoalStates().keys():
                    result += space
                else:
                    result += " |       /\      "
            result += " |\n"
            
            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in mdp.getGoalStates().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(qValues[((x, y), 'N')])
            result += " |\n"
            
            for x in range(self.width):
                result += space
            result += " |\n"
            
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " |     #####     "
                elif (x, y) in mdp.getGoalStates().keys():
                    result += " |     {:+0.2f}     ".format(qValues[((x, y), mdp.TERMINATE)])
                else:
                    result += " | <{:+0.2f}  {:+0.2f}>".format(qValues[((x, y), 'W')], qValues[((x, y), 'E')])
            result += " |\n"

            for x in range(self.width):
                result += space
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in mdp.getGoalStates().keys():
                    result += space
                else:
                    result += " |     {:+0.2f}     ".format(qValues[((x, y), 'S')])
            result += " |\n"

            for x in range(self.width):
                if (x, y) in self.blockedStates or (x, y) in mdp.getGoalStates().keys():
                    result += space
                else:
                    result += " |       \/      "
            result += " |\n"
            result += line        
        return result

    ''' Convert a grid world policy to a formatted string '''
    def policyToString(self, policy):
        line = " {:-^{n}}\n".format("", n=len(" |  N ")*self.width + 1)
        result = line 
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " | ###"
                else:
                    action = "T" if policy[(x, y)] == self.TERMINATE else policy[(x, y)]
                    result += " |  " + action + " "
            result += " |\n"
            result += line

        return result

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

class MultiArmedBandits():

    def epsilonGreedy(actions, state, qValues, epsilon=0.1):
        r = random.random()

        # select a random action with epsilon probability
        if r < epsilon:
            index = random.randint(0, len(actions) - 1)
            return actions[index]
        else:
            # find the action with maximum Q value
            maxAction = None
            maxValue = float('-inf')
            result = str(actions) + "\n"
            for action in actions:
                value = qValues[(state, action)]
                result += "\t Q(" + str(state) + ", " + str(action) + ") = " + str(value) + "\n"
                result += "\t maxValue = " + str(maxValue)
                if value > maxValue:
                    maxAction = action
                    maxValue = value
            if maxAction == None:
                print(result)
            return maxAction

import math

class QLearning():

    def __init__(self, mdp):
        self.mdp = mdp

    def qLearning(self, episodes = 1000, alpha = 0.05, decay = 0.2, epsilon = 0.1):
        qValues = self.initialiseQFunction()

        decay = 0.2 
        for i in range(episodes):
            state = mdp.getInitialState()

            while not mdp.isTerminal(state):
                validActions = mdp.getActions(state)
                action = MultiArmedBandits.epsilonGreedy(validActions, state, qValues)
                (newState, reward) = mdp.simulate(state, action)
                newValue = self.update(qValues, state, action, newState, reward, alpha)
                qValues[(state, action)] = newValue
                state = newState

            alpha = max(0.05, alpha * math.exp(-decay * i))
            print(alpha)
        return qValues


    def update(self, qValues, state, action, newState, reward, alpha):
        (_, maxQValue) = self.getMaxQ(qValues, newState)
        qValue = qValues[(state, action)]
        return qValue + alpha * (reward + mdp.discountFactor * maxQValue - qValue)

    def getMaxQ(self, qValues, state):
        argmaxQ = None
        maxQ = float('-inf')
        for action in self.mdp.getActions(state):
            value = qValues[(state, action)]
            if maxQ < value:
                argMaxQ = action
                maxQ = value
        return (argmaxQ, maxQ)

    def initialiseQFunction(self):
        qValues = dict()
        for state in self.mdp.getStates():
            for action in self.mdp.getActions():
                qValues.update({(state, action): 0.0})
        return qValues

mdp = NavigationMDP(discountFactor=0.9)
valueIteration = ValueIteration(mdp)

for iterations in []: #1, 2, 3, 4, 5, 10, 100]:
    print("After iteration " + str(iterations))
    #mdp.valueFunctionToString(valueIteration.valueIteration(iterations = iterations))
    print(mdp.valueFunctionToString(valueIteration.valueIteration(iterations = iterations)) + "\n")

print("Policy after 100 iterations")
print(mdp.policyToString(mdp.extractPolicyFromValueFunction(valueIteration.valueIteration(iterations = 100))))


qLearning = QLearning(mdp)
print("qLearning")
print(mdp.qFunctionToString(qLearning.qLearning(episodes = 1000)) + "\n")
