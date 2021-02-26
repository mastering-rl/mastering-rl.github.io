import math
import random
import time
from navigation_mdp import *
from multi_armed_bandits import *

class Node():
    def __init__(self, mdp, state, parent, probability, reward):
        self.mdp = mdp

        self.state = state

        # a dictionary from actions to a set of potential outcome nodes
        self.parent = parent
        self.children = {}
        
        # this actions probability and reward in the MDP
        self.probability = probability
        self.reward = reward

        # the value of this node and number of times it has been visited
        self.value = 0.0
        self.visits = 0

    def isFullyExpanded(self):
        validActions = self.mdp.getActions(self.state)
        if len(validActions) == len(self.children):
            return True
        else:
            return False

class ModelBasedMCTS():

    # Initialise with a base policy (default to empty)
    def __init__(self, mdp, qValues = dict()):
        self.mdp = mdp
        self.qValues = qValues

    '''
    Execute the MCTS algorithm from the initial state given, with timeOut in seconds
    '''
    def mcts(self, timeOut = 100, epsilon = 0.1):

        rootNode = Node(mdp, mdp.getInitialState(), None, 1.0, 0)

        startTime = int(time.time() * 1000)
        currentTime = int(time.time() * 1000)
        while currentTime < startTime + timeOut * 1000:

            expandedNode = self.select(rootNode)
            print("expand", expandedNode.state)
            outcomes = self.expand(expandedNode)
            child = self.choose(outcomes)
            reward = self.simulate(child)
            self.backPropagate(expandedNode, reward)
            currentTime = int(time.time() * 1000)

        print(self.qValues)
        return self.qValues

    def getQValue(qValues, state, action):
        if (state, action) in qValues.keys():
            return qValues[(state,action)]
        else:
            return 0.0

    def select(self, node):

        node.visits += 1
        if node.isFullyExpanded():
            return self.selectChild(node)
        else:
            return node
            
    def selectChild(self, node):
        actions = list(node.children.keys())
        for action in actions:
            #get the Q values from all outcome nodes
            qValue = 0.0
            for outcome in node.children[action]:
                 qValue += outcome.value * outcome.probability
            self.qValues[(node.state, action)] = qValue
        bestAction = MultiArmedBandits.uct(actions, node.state, self.qValues)
        outcome = self.selectOutcome(node.children[bestAction])
        return self.select(outcome)

    '''
        Select an outcome based on the probability of the child nodes
    '''
    def selectOutcome(self, outcomes):
        r = random.random()
        cumulativeProbability = 0.0
        for outcome in outcomes:
            if r >= cumulativeProbability and r <= outcome.probability + cumulativeProbability:
                return outcome
            cumulativeProbability += outcome.probability
            if cumulativeProbability >= 1.0:
                raise "Cumulative probability >= 1.0 for outcome " + str(outcome.state)
        raise "No selected outcome: " + str(outcomes)
        return None

    def expand(self, node):

        #randomly select an unexpanded action to expand
        actions = self.mdp.getActions(node.state) - node.children.keys()
        action = random.choice(list(actions))
        
        #expand all outcomes nodes
        transitions = self.mdp.getTransitions(node.state, action)
        for (newState, probability) in transitions:
            reward = self.mdp.getReward(node.state, action, newState)
            if action in node.children:
                node.children[action].add(Node(mdp, newState, node, probability, reward))
            else:
                node.children.update({action : set([Node(mdp, newState, node, probability, reward)])})
        return node.children[action]

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
            (newState, reward) = self.mdp.simulate(state, action)
            cumulativeReward += pow(self.mdp.getDiscountFactor(), depth) * reward
            depth += 1
            state = newState
        node.value = cumulativeReward
        return cumulativeReward
    
    def backPropagate(self, node, reward):
        while node != None:
            for action in node.children.keys():
                value = 0.0
                for outcome in node.children[action]:
                    value += outcome.probability * (outcome.reward + mdp.getDiscountFactor() * outcome.value)
                self.qValues.update({action: value})
            node.value = self.getMaxQ(self.qValues)
            node = node.parent
            
    def getMaxQ(self, qValues, state):
        argmaxQ = None
        maxQ = float('-inf')
        for action in self.mdp.getActions(state):
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


mdp = NavigationMDP(discountFactor=0.9)
mcts = ModelBasedMCTS(mdp)
qFunction = mcts.mcts()
print("mcts")
policy = mdp.extractPolicyFromQFunction(qFunction)
print(mdp.qFunctionToString(qFunction) + "\n")
print(mdp.policyToString(policy))
