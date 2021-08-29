from collections import defaultdict

class VTable():

    def __init__(self, default = 0.0):
        self.vtable = defaultdict(lambda : default)

    def update(self, state, value):
        self.vtable[state] = value

    def merge(self, vtable):
        for state in vtable.vtable.keys():
            self.update(state, vtable.getValue(state))

    def getValue(self, state):
        return self.vtable[state]
