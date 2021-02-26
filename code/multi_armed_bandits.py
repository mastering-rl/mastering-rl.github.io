import random

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
            for action in actions:
                value = qValues[(state, action)]
                if value > maxValue:
                    maxAction = action
                    maxValue = value
            return maxAction

    def uct(actions, state, qValues):
        return MultiArmedBandits.epsilonGreedy(actions, state, qValues)
