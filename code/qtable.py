from collections import defaultdict
from qfunction import *

class QTable(QFunction):
    
    def __init__(self, default = 0.0):
        self.qTable = defaultdict(lambda : 0.0)

    def update(self, state, action, delta):
        self.qTable[(state, action)] = self.qTable[(state, action)] + delta

    def getQValue(self, state, action):
        return self.qTable[(state, action)]
