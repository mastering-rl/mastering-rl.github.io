import random

class MDP:
    ''' Return all states of this MDP '''
    def getStates(self): abstract

    ''' Return all actions with non-zero probability from this state '''
    def getActions(self, state): abstract

    '''
        Return all non-zero probability transitions for this action from this state,
        as a list of (state, probability) pairs
    '''
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

    def getQValue(qValues, state, action):
        qValue = 0.0
        if (state, action) in qValues.keys():
            qValue = qValues[(state, action)]
        return qValue
    
    ''' Return a policy given a value function '''
    def extractPolicyFromValueFunction(self, values):
        policy = dict()
        for state in self.getStates():
            maxQ = float('-inf')
            for action in self.getActions(state):
               # Calculate the value of Q(s,a)
               qValue = 0.0
               for (newState, probability) in self.getTransitions(state, action):
                   reward = self.getReward(state, action, newState)
                   qValue += probability * (reward + (self.getDiscountFactor() * values[newState]))

               # if this is the maximum Q-value so far, set the policy for this state
               if qValue > maxQ:
                   policy.update({state: action})
                   maxQ = qValue

        return policy

    ''' Return a policy given a Q function '''
    def extractPolicyFromQFunction(self, qValues):
        policy = dict()
        for state in self.getStates():

            # Find the maximum Q-value
            maxQ = float('-inf')
            for action in self.getActions(state):
               # if this is the maximum Q-value so far, set the policy for this state
                qValue = MDP.getQValue(qValues, state, action)
            
                if qValue > maxQ:
                    policy.update({state: action})
                    maxQ = qValue

        return policy

    ''' 
       Return a new state and a reward for executing action in state, 
       based on the underlying probability. This can be used for 
       model-free method, but requires a model to operator.
       Override for simulation-based learning
    '''
    def simulate(self, state, action):
        r = random.random()
        cumulativeProbability = 0.0
        for (newState, probability) in self.getTransitions(state, action):
            if r >= cumulativeProbability and r <= probability + cumulativeProbability:
                return (newState, self.getReward(state, action, newState))
            cumulativeProbability += probability
            if cumulativeProbability >= 1.0:
                raise "Cumulative probability >= 1.0 for action " + str(action) + " from " + str(state)
        print("No outcome state in simulation for action " + str(action) + " from " + str(state))
        raise "No outcome state in simulation for action"
        return None


