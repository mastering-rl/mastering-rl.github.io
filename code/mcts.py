import math
import random
import time
from multi_armed_bandits import *

class Node():    

    # record a unique node id to distinguish duplicated states
    nextNodeID = 0
    
    def __init__(self, mdp, parent, state):
        self.mdp = mdp  
        self.parent = parent
        self.state = state
        self.id = Node.nextNodeID
        Node.nextNodeID += 1

        # the value and the total visits to this node
        self.visits = 0
        self.value = 0.0

    '''
    Return the value of this node
    '''
    def getValue(self):
        return self.value

class StateNode(Node):
    
    def __init__(self, mdp, parent, state, reward = 0, probability = 1.0, bandit = UpperConfidenceBounds()):
        super().__init__(mdp, parent, state)
        
        # a dictionary from actions to an environment node
        self.children = {}

        # the reward received for this state
        self.reward = reward
        
        # the probability of this node being chosen from its parent
        self.probability = probability

        # a multi-armed bandit for this node
        self.bandit = bandit

    '''
    Return true if and only if all child actions have been expanded
    '''
    def isFullyExpanded(self):
        validActions = self.mdp.getActions(self.state)
        if len(validActions) == len(self.children):
            return True
        else:
            return False

    def select(self):
        if not self.isFullyExpanded():
            return self
        else:
            actions = list(self.children.keys())    
            qValues = dict()
            for action in actions:
                #get the Q values from all outcome nodes
                qValues[action] = self.children[action].getValue()
            bestAction = self.bandit.select(actions, qValues)
            return self.children[bestAction].select()

    def expand(self):
        #randomly select an unexpanded action to expand
        actions = self.mdp.getActions(self.state) - self.children.keys()
        action = random.choice(list(actions))

        #choose an outcome
        newChild = EnvironmentNode(self.mdp, self, self.state, action)
        newStateNode = newChild.expand()
        self.children[action] = newChild
        return newStateNode

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits) 
        
        if self.parent != None:
            self.parent.backPropagate(reward)

    def getQFunction(self):
        qValues = {}
        for action in self.children.keys():
            qValues[(self.state, action)] = round(self.children[action].getValue(), 3)
        return qValues

class EnvironmentNode(Node):
    
    def __init__(self, mdp, parent, state, action):
        super().__init__(mdp, parent, state)
        self.outcmes = {}
        self.action = action
        
        # a set of outcomes
        self.children = []

    def select(self):
        # choose one outcome based on transition probabilities
        (newState, reward) = self.mdp.execute(self.state, self.action)

        #find the corresponding state
        for child in self.children:
            if newState == child.state:
                return child.select()

    def addChild(self, action, newState, reward, probability):
        child = StateNode(self.mdp, self, newState, reward, probability)
        self.children += [child]
        return child

    def expand(self):
        # choose one outcome based on transition probabilities
        (newState, reward) = self.mdp.execute(self.state, self.action)

        # expand all outcomes
        selected = None
        transitions = self.mdp.getTransitions(self.state, self.action)
        for (outcome, probability) in transitions:
            newChild = self.addChild(self.action, outcome, reward, probability)
            # find the child node correponding to the new state
            if outcome == newState:
                selected = newChild
        return selected

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((reward - self.value) / self.visits)
        self.parent.backPropagate(reward * self.mdp.getDiscountFactor())

class MCTS():

    def __init__(self, mdp):
        self.mdp = mdp

    '''
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    '''
    def mcts(self, timeout = 1):
        rootNode = StateNode(self.mdp, None, self.mdp.getInitialState())
        
        startTime = int(time.time() * 1000)
        currentTime = int(time.time() * 1000)
        while currentTime < startTime + timeout * 1000:
            # find a state node to expand
            selectedNode = rootNode.select()
            if not self.mdp.isTerminal(selectedNode):
                child = selectedNode.expand()
                reward = self.simulate(child)
                child.backPropagate(reward)
                
            currentTime = int(time.time() * 1000)

        return rootNode

    '''
        Choose a random action. Heustics can be used here to improve simulations.
    '''
    def choose(self, state):
        return random.choice(self.mdp.getActions(state))

    '''
        Simulate until a terminal state
    '''
    def simulate(self, node):
        state = node.state
        cumulativeReward = 0.0
        depth = 0
        while not self.mdp.isTerminal(state):
            #choose an action to execute
            action = self.choose(state)
            
            # execute the action
            (newState, reward) = self.mdp.execute(state, action)

            # discount the reward 
            cumulativeReward += pow(self.mdp.getDiscountFactor(), depth) * reward
            depth += 1

            state = newState
            
        return cumulativeReward

if __name__ == "__main__":
    from gridworld import *
    
    mdp = GridWorld()
    rootNode = MCTS(mdp).mcts(timeout=10.0)
    print("mcts")
    print(rootNode.getQFunction())
