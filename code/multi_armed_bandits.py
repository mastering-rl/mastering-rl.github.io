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
            maxActions = []
            maxValue = float('-inf')
            for action in actions:
                value = qValues[(state, action)]
                if value > maxValue:
                    maxActions = [action]
                    maxValue = value
                elif value == maxValue:
                    maxActions += [action]

            # if there are multiple actions with the highest value
            # choose one randomly
            return random.choice(maxActions)

    def uct(actions, state, qValues):
        return MultiArmedBandits.epsilonGreedy(actions, state, qValues)
