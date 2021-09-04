import math
import random
import time
from multi_armed_bandits import *

class Node():    

    # record a unique node id to distinguish duplicated states
    next_node_id = 0
    
    def __init__(self, mdp, parent, state):
        self.mdp = mdp  
        self.parent = parent
        self.state = state
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # the value and the total visits to this node
        self.visits = 0
        self.value = 0.0

    '''
    Return the value of this node
    '''
    def get_value(self):
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
    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state)
        if len(valid_actions) == len(self.children):
            return True
        else:
            return False

    def select(self):
        if not self.is_fully_expanded():
            return self
        else:
            actions = list(self.children.keys())    
            q_values = dict()
            for action in actions:
                #get the Q values from all outcome nodes
                q_values[action] = self.children[action].getValue()
            best_action = self.bandit.select(actions, q_values)
            return self.children[best_action].select()

    def expand(self):
        #randomly select an unexpanded action to expand
        actions = self.mdp.getActions(self.state) - self.children.keys()
        action = random.choice(list(actions))

        #choose an outcome
        new_child = EnvironmentNode(self.mdp, self, self.state, action)
        new_state_node = new_child.expand()
        self.children[action] = new_child
        return new_state_node

    def back_propagate(self, reward):
        self.visits += 1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits) 
        
        if self.parent != None:
            self.parent.backPropagate(reward)

    def get_q_function(self):
        q_values = {}
        for action in self.children.keys():
            q_values[(self.state, action)] = round(self.children[action].getValue(), 3)
        return q_values

class EnvironmentNode(Node):
    
    def __init__(self, mdp, parent, state, action):
        super().__init__(mdp, parent, state)
        self.outcmes = {}
        self.action = action
        
        # a set of outcomes
        self.children = []

    def select(self):
        # choose one outcome based on transition probabilities
        (new_state, reward) = self.mdp.execute(self.state, self.action)

        #find the corresponding state
        for child in self.children:
            if new_state == child.state:
                return child.select()

    def add_child(self, action, new_state, reward, probability):
        child = StateNode(self.mdp, self, new_state, reward, probability)
        self.children += [child]
        return child

    def expand(self):
        # choose one outcome based on transition probabilities
        (new_state, reward) = self.mdp.execute(self.state, self.action)

        # expand all outcomes
        selected = None
        transitions = self.mdp.getTransitions(self.state, self.action)
        for (outcome, probability) in transitions:
            new_child = self.add_child(self.action, outcome, reward, probability)
            # find the child node correponding to the new state
            if outcome == new_state:
                selected = new_child
        return selected

    def back_propagate(self, reward):
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
        root_node = StateNode(self.mdp, None, self.mdp.get_initial_state())
        
        start_time = int(time.time() * 1000)
        current_time = int(time.time() * 1000)
        while current_time < start_time + timeout * 1000:
            # find a state node to expand
            selected_node = root_node.select()
            if not self.mdp.is_terminal(selected_node):
                child = selected_node.expand()
                reward = self.simulate(child)
                child.back_propagate(reward)
                
            current_time = int(time.time() * 1000)

        return root_node

    '''
        Choose a random action. Heustics can be used here to improve simulations.
    '''
    def choose(self, state):
        return random.choice(self.mdp.get_actions(state))

    '''
        Simulate until a terminal state
    '''
    def simulate(self, node):
        state = node.state
        cumulative_reward = 0.0
        depth = 0
        while not self.mdp.is_terminal(state):
            #choose an action to execute
            action = self.choose(state)
            
            # execute the action
            (new_state, reward) = self.mdp.execute(state, action)

            # discount the reward 
            cumulative_reward += pow(self.mdp.get_discount_factor(), depth) * reward
            depth += 1

            state = new_state
            
        return cumulative_reward

if __name__ == "__main__":
    from gridworld import *
    
    mdp = GridWorld()
    root_node = MCTS(mdp).mcts(timeout=10.0)
    print("mcts")
    print(root_node.get_q_function())
