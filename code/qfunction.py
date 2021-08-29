import random

class QFunction():

    '''
        Update the Q-value of (state, action) by delta
    '''
    def update(self, state, action, delta): abstract

    '''
        Get a Q value for a given state-action pair
    '''
    def getQValue(self, state, action): abstract

    '''
        Return a pair containing the action and Q-value, where the
        action has the maximum Q-value in state
    '''
    def getMaxQ(self, state, actions):
        argMaxQ = None
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
