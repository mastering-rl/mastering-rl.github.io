import random

class QFunction():

    '''
        Update this Q-Function with a new value
    '''
    def update(self, state, action, value): abstract

    '''
        Get a Q value for a given state-action pair
    '''
    def getQValue(self, state, action): abstract

    '''
        Return a pair containing the action and Q-value, where the
        action has the maximum Q-value in state
    '''
    def getMaxQ(self, state, actions):
        argmaxQ = None
        maxQ = float('-inf')
        for action in actions:
            value = self.getQValue(state, action)
            if maxQ < value:
                argMaxQ = action
                maxQ = value
            # if these actions have the same Q-value, randomly choose one
            elif maxQ == value:
                argMaxQ = random.choice([argMaxQ, action])
        return (argMaxQ, maxQ)
