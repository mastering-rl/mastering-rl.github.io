import math
import random
import time
from gridworld import *
from multi_armed_bandits import *

class Node():
    
    def __init__(self, mdp, parent, state):
        self.mdp = mdp  
        self.parent = parent
        self.state = state

        # the total rewards seen from this node
        self.totalRewards = 0.0
        self.visits = 0

        if parent == None:
            self.level = 0
        else:
            self.level = parent.level + 1


    '''
    Return the value of this node
    '''
    def getValue(self):
        value = 0.0
        if self.visits > 0:
            value = self.totalRewards / self.visits
        return value


class StateNode(Node):
    
    def __init__(self, mdp, parent, state):
        super().__init__(mdp, parent, state)
        
        # a dictionary from actions to an environment node
        self.children = {}

        # A multi-armed bandit for this node
        self.bandit = UpperConfidenceBounds()

    '''
    Return true if and only if all child actions have been expanded
    '''
    def isFullyExpanded(self):
        validActions = self.mdp.getActions(self.state)
        if len(validActions) == len(self.children):
            return True
        else:
            return False

    def select(self, level):
        if not self.isFullyExpanded():
            return self
        else:
            actions = list(self.children.keys())    
            qValues = dict()
            for action in actions:
                #get the Q values from all outcome nodes
                qValues[action] = self.children[action].getValue()
            bestAction = self.bandit.select(actions, qValues)
            if level == 0:
                print("select " + bestAction)
            return self.children[bestAction].select(level + 1)

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
        self.totalRewards += reward
        if self.parent != None:
            self.parent.backPropagate(reward)

    def toString(self, level = 0):
        result = ""
        for action in self.children.keys():
            child = self.children[action]
            result += "\"" + self.nodeString() + "\" -> \"" +  child.nodeString() + " \" [label = \"" + action + "\"]\n"
        for action in self.children:
            result += self.children[action].toString()
        return result

    def nodeString(self):
        return "State" + str(self.state)+ "." + str(self.level) + "[" + str(self.totalRewards) + "]"

    def getQFunction(self):
        qValues = {}
        for action in self.children.keys():
            qValues[(self.state, action)] = self.children[action].getValue()
        return qValues

class EnvironmentNode(Node):
    def __init__(self, mdp, parent, state, action):
        super().__init__(mdp, parent, state)
        self.outcmes = {}
        self.action = action
        
        # a set of outcomes
        self.children = []

    def select(self, level):
        # choose one outcome based on transition probabilities
        (newState, reward) = self.mdp.execute(self.state, self.action)

        #find the corresponding state
        for child in self.children:
            if newState == child.state:
                return child.select(level + 1)

        # unreachable
        return None

    def addChild(self, action, newState):
        child = StateNode(self.mdp, self, newState)
        self.children += [child]
        return child

    def expand(self):
        # choose one outcome based on transition probabilities
        (newState, reward) = self.mdp.execute(self.state, self.action)

        seleted = None
        transitions = self.mdp.getTransitions(self.state, self.action)
        for (outcome, probability) in transitions:
            newChild = self.addChild(self.action, outcome)
            if outcome == newState:
                selected = newChild

        self.expanded = True

        return selected

    def backPropagate(self, reward):
        self.visits += 1
        self.totalRewards += reward
        self.parent.backPropagate(reward * self.mdp.getDiscountFactor())
    
    def toString(self, level = 0):
        result = ""
        for child in self.children:
            result += "\"" + self.nodeString() + "\"" + " -> \"" + child.nodeString() + "\"\n"
        for child in self.children:
            result += child.toString(level + 1)
        return result

    def nodeString(self):
        return "Env" + str(self.state) + "." + str(id(self))

class MCTS():

    # Initialise with a base policy (default to empty)
    def __init__(self, mdp, qValues = dict()):
        self.mdp = mdp
        self.qValues = qValues

    '''
    Execute the MCTS algorithm from the initial state given, with timeOut in seconds
    '''
    def mcts(self, timeOut = 10, epsilon = 0.1):

        rootNode = StateNode(mdp, None, mdp.getInitialState())

        startTime = int(time.time() * 1000)
        currentTime = int(time.time() * 1000)
        while currentTime < startTime + timeOut * 1000:

            # find a state node to expand
            selectedNode = rootNode.select(level = 0)
            print("select %s.%d" % (str(selectedNode.state), selectedNode.level))
            if not self.mdp.isTerminal(selectedNode):

                child = selectedNode.expand()
                
                print("\tsimulate from %s" % str(child.state))
                reward = self.simulate(child)
                print("reward = " + str(reward))
                self.backPropagate(selectedNode, reward)
                
            currentTime = int(time.time() * 1000)

        print("digraph mcts {")
        print(rootNode.toString())
        print("}")
        print("Q = " + str(rootNode.getQFunction()))
        return self.qValues

    def getQValue(qValues, state, action):

        if (state, action) in qValues.keys():
            return qValues[(state,action)]
        else:
            return 0.0

    def choose(self, outcomes):
        return random.choice(list(outcomes))

    def simulate(self, node):
        state = node.state
        cumulativeReward = 0.0
        depth = 0
        while not self.mdp.isTerminal(state):
            #choose a random action
            actions = self.mdp.getActions(state)
            action = self.choose(actions)
            (newState, reward) = self.mdp.execute(state, action)
            cumulativeReward += pow(self.mdp.getDiscountFactor(), depth) * reward
            depth += 1
            state = newState
        node.value = cumulativeReward
        return cumulativeReward

    '''
        Backpropoate the reward from a simulate node to the root
    '''
    def backPropagate(self, node, reward):
        node.backPropagate(reward)

            
    def getMaxQ(self, qValues, state):
        argmaxQ = None
        maxQ = float('-inf')
        for action in self.mdp.getActions(state):
            value = float('-inf')
            if (state, action) in qValues.keys():
                value = qValues[(state, action)]
            if maxQ < value:
                argMaxQ = action
                maxQ = value
        return (argmaxQ, maxQ)
    
    def getMaxQ(self, qValues):
        maxQ = float('-inf')
        for action in qValues.keys():
            value = qValues[action]
            if maxQ < value:
                maxQ = value
        return maxQ


mdp = GridWorld(discountFactor=0.9, noise=0.1) #, blockedStates=[(1,1), (2,1)])
mcts = MCTS(mdp)
qFunction = mcts.mcts()
print("mcts")
policy = mdp.extractPolicyFromQFunction(qFunction)
print(mdp.qFunctionToString(qFunction) + "\n")
print(mdp.policyToString(policy))
