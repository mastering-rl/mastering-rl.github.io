from collections import defaultdict
from qfunction import QFunction

class QTable(QFunction):
    
    def __init__(self, default = 0.0):
        self.qTable = defaultdict(lambda : default)

    def update(self, state, action, delta):
        self.qTable[(state, action)] = self.qTable[(state, action)] + delta

    def getQValue(self, state, action):
        return self.qTable[(state, action)]
