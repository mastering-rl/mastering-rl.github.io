
class MDP:
   def getStates(self): abstract
   def getActions(self, state): abstract
   def getTransitions(self, state, action): abstract
   def getRewards(self, state, action, nextState): abstract


   def states():
     return [(0,0), (0,1), (0,2), (0,3),
          (1,0), (1,1), (1,2), (1,3),
          (2,0), (2,1), (2,2), (2,2)]

def actions():
  return ['N', 'S', 'E', 'W']



#V=[[0.0, 0.0, 0.0, 0.0],
#   [0.0, 0.0, 0.0, 0.0],
#   [0.0, 0.0, 0.0, 0.0]]

transitions(states()[0])