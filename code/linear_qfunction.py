from qfunction import QFunction

class LinearQFunction(QFunction):

    def __init__(self, features, weights = None, default = 0.0):
        self.features = features
        if weights == None:
            self.weights = [default for _ in range(0, features.numActions()) for _ in range(0, features.numFeatures())]

    def update(self, state, action, delta):
        # update the weights
        featureValues = self.features.extract(state, action)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (delta * featureValues[i])        

    def getQValue(self, state, action):
        qValue = 0.0
        featureValues = self.features.extract(state, action)
        for i in range(len(featureValues)):
            qValue += featureValues[i] * self.weights[i]
        return qValue
    
