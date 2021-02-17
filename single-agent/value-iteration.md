---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
## Value Iteration

**Learning outcomes**

1.  Apply value iteration to solve small-scale MDP problems manually and program value iteration algorithms to 
    solve medium-scale MDP problems automatically

2.  Construct a policy from a value function

3.  Compare and contrast value iteration to policy iteration

4.  Discuss the strengths and weaknesses of value iteration and policy iteration algorithms

### Overview

*Value Iteration* is a method for finding the optimal value function $V^*$ by solving the
Bellman equations iteratively. It uses the concept of dynamic programming to keep a value function $V$ that approximates the optimal value function $V^**$, iteratvely improving $V$ until it converges to $V^*$ (or close to it). 

### Algorithm

Once we understand the Bellman equation, the value iteration algorithm is straightforward.

:::{admonition} Algorithm -- Value Iteration

**Input:** MDP $M = \langle S, s_0, A, P_a(s' \mid s), r(s,a,s')\rangle$\
**Output:** Value function $V$

Set $V$ to arbitrary value function; e.g., $V(s)=0$ for all $s$

$\text{Repeat}$\
$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \underbrace{V'(s) \leftarrow \max_{a \in A(s)} \sum_{s' \in S}  P_a(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]}_{\text{Bellman equation}}$\
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |V'(s) - V(s)|)$\
$\quad\quad V \leftarrow V'$\
$\text{Until}~ \Delta \leq \theta$
:::

As we can see, this is just applying the Bellman equation iteratively until either the value function $V$ doesn't change anymore, or until it changes in by a very small amount ($\theta$).

We could also write the algorithm using the idea of Q-functions, which is closer to a code-based implementation. For this, the loop is:

$\quad\quad \Delta \leftarrow 0$\
$\quad\quad \text{For each}~ s \in S$\
$\quad\quad\quad\quad \text{For each}~ a \in A(s)$\
$\quad\quad\quad\quad\quad\quad Q(s,a) \leftarrow \sum_{s' \in S}  P_a(s' \mid s)\ [r(s,a,s') +  \gamma\ V(s') ]$\_
$\quad\quad\quad\quad \Delta \leftarrow \max(\Delta, |\max_{a \in A(s)} Q(s,a) - V(s)|)$\
$\quad\quad\quad\quad V(s) \leftarrow \max_{a \in A(s)} Q(s,a)$

Value iteration converges to the optimal policy as iterations continue: $V \mapsto V^*$ as $i \mapsto \infty$, where $i$ is the number of iterations. So, given an infinite amount of iterations, it will be optimal.


Value iteration converges to the optimal value function $V^*$ asymptotically, but in practice, the algorithm is stopped when the *residual*  $\Delta$ reaches some pre-determined threshold $\theta$ -- that is, when the largest change in the values between iterations is "small enough".

A policy can now be easily defined: in a state $s$, given $V$, choose the action with the highest expected reward using policy extraction. The resulting greedy policy $\pi_V$ has it's *loss* bounded by $2 \gamma  \Delta / 1-\gamma$.

### Complexity

The complexity of each iteration is $O(|S|^2 |A|)$. On each iteration, we iterate in an outer loop over all states in $S$, and in each outer loop iteration, we need to iterate over all states ($\sum_{s' \in S}$), meaning $|S|^2$ iterations. But also within each outer loop iteration, we need to calculate the value for every action to find the maximum.

It is clear to see that the value iteration can be easily parallelised by updating the value of many states at once: the values of states at step $t + 1$ are dependent only on the value of other states at step $t$.

### Implementation

```{code-cell} ipython3
:tags: [hide-input]

class MDP:
    ''' Return all states of this MDP '''
    def getStates(self): abstract

    ''' Return all actions with non-zero probability from this state '''
    def getActions(self, state): abstract

    ''' Return all non-zero probability transitions for this action from this state '''
    def getTransitions(self, state, action): abstract

    ''' Return the reward for transitioning from state to nextState via action '''
    def getReward(self, state, action, nextState): abstract

    ''' Return the discount factor for this MDP '''
    def getDiscountFactor(self): abstract

    ''' Return the initial state of this MDP '''
    def getInitialState(self): abstract

    ''' Return all goal states of this MDP '''
    def getGoalStates(self): abstract

    ''' Return a policy given a value function '''
    def extractPolicy(self, values):
        policy = dict()
        for state in mdp.getStates():
            maxQ = float('-inf')
            for action in mdp.getActions(state):
               # Calculate the value of Q(s,a)
               qValue = 0.0
               for (newState, probability) in mdp.getTransitions(state, action):
                   reward = mdp.getReward(state, action, newState)
                   qValue += probability * (reward + (mdp.getDiscountFactor() * values[newState]))

               # if this is the maximum Q-value so far, set the policy for this state
               if qValue > maxQ:
                   policy.update({state: action})
                   maxQ = qValue

        return policy

    ''' 
       Return a new state and a reward for executing action in state, 
       based on the underlying probability. This can be used for 
       model-free method
    '''
    def simulate(self, state, action): abstract

class NavigationMDP(MDP):

    # labels for terminate action and terminal state
    TERMINATE = 'terminate'
    TERMINAL = ('terminal', 'terminal')

    def __init__(self, width = 4, height = 3,
                 discountFactor = 0.9, 
                 blockedStates = [(1,1)],
                 goals = [((3,2), 1), ((3,1), -1)]):
        self.width = width
        self.height = height
        self.blockedStates = blockedStates
        self.discountFactor = discountFactor
        self.goalStates = dict(goals)

    def getStates(self):
        states = [self.TERMINAL]
        for x in range(self.width):
            for y in range(self.height):
                if not (x, y) in self.blockedStates:
                    states.append((x,y))
        return states

    def getActions(self, state=None):

        if (state == None):
            return ['N', 'S', 'E', 'W', self.TERMINATE]

        actions = []
        for action in ['N', 'S', 'E', 'W', self.TERMINATE]:
            for (newState, probability) in self.getTransitions(state, action):
                if probability > 0:
                    actions.append(action)
                    break
        return actions

    def getGoalStates(self):
        return self.goalStates

    def validAdd(self, state, newState, probability):
        # if the next state is blocked, stay in the same state
        if (newState in self.blockedStates):
            return [(state, probability)]

        # move to the next space if it is not off the grid
        (x, y) = newState
        if (x >= 0 and x < self.width and y >= 0 and y < self.height):
            return [((x, y), probability)]
 
        # if off the grid, state in the same state
        return [(state, probability)]

    def getTransitions(self, state, action):
        transitions = []

        if state == self.TERMINAL:
            return [(self.TERMINAL, 1.0)]

        (x, y) = state
        if state in self.getGoalStates().keys():
            if action == self.TERMINATE:
                transitions += [(self.TERMINAL, 1.0)]

        elif action == 'N':
            transitions += self.validAdd(state, (x, y + 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == 'S':
            transitions += self.validAdd(state, (x, y - 1), 0.8)
            transitions += self.validAdd(state, (x - 1, y), 0.1)
            transitions += self.validAdd(state, (x + 1, y), 0.1)

        elif action == 'E':
            transitions += self.validAdd(state, (x + 1, y), 0.8)
            transitions += self.validAdd(state, (x, y - 1), 0.1)
            transitions += self.validAdd(state, (x, y + 1), 0.1)

        elif action == 'W':
            transitions += self.validAdd(state, (x - 1, y), 0.8)
            transitions += self.validAdd(state, (x, y - 1), 0.1)
            transitions += self.validAdd(state, (x, y + 1), 0.1)

        return transitions

    def getReward(self, state, action, newState):
       reward = 0.0
       if state in self.getGoalStates().keys() and newState == self.TERMINAL:
          reward = self.getGoalStates().get(state)
       return reward

    def getDiscountFactor(self):
        return self.discountFactor

    ''' Convert a grid world value function to a formatted string '''
    def valueFunctionToString(self, values):
        line = " {:-^{n}}\n".format("", n=len(" | +0.00")*self.width + 1)
        result = line
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " | #####"
                else:
                    result += " | {:+0.2f}".format(values[(x, y)])
            result += " |\n"
            result += line

        return result

    ''' Convert a grid world policy to a formatted string '''
    def policyToString(self, policy):
        line = " {:-^{n}}\n".format("", n=len(" |  N ")*self.width + 1)
        result = line 
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if (x, y) in self.blockedStates:
                    result += " | ###"
                else:
                    action = "T" if policy[(x, y)] == self.TERMINATE else policy[(x, y)]
                    result += " |  " + action + " "
            result += " |\n"
            result += line

        return result
```

Below is a Python implementation for value iteration. In this implementation, the parameters `iterations` is the number of iterations around the loop, which will terminate before convergence is the maximum number of iterations is reach. The parameter `theta` is $\theta$ in the value iteration algorithm above. Once the difference ($\Delta$) is less than `theta` , the loop will terminate.

```{code-cell} ipython3
class ValueIteration():

    def __init__(self, mdp):
        self.mdp = mdp

    ''' Implmentation of value iteration '''
    def valueIteration(self, iterations = 100, theta = 0.001):

        # Initialise the value function V with all 0s
        values = self.initialiseValueFunction()
        for _ in range(iterations):

           delta = 0
           for state in mdp.getStates():
               qValues = dict()
               for action in mdp.getActions(state):
                   # Calculate the value of Q(s,a)
                   newValue = 0.0
                   for (newState, probability) in mdp.getTransitions(state, action):
                       reward = mdp.getReward(state, action, newState)
                       newValue += probability * (reward + (mdp.getDiscountFactor() * values[newState]))
                   qValues.update({action: newValue})

               # V(s) = max_a Q(s,a)
               maxQ = max(qValues.values())
               delta = max(delta, abs(values[state] - maxQ))
               values.update({state: maxQ})

           # terminate if the value function has converged
           if delta < theta:
               break

        return values

    def initialiseValueFunction(self):
        values = dict()
        for state in self.mdp.getStates():
            values.update({state: 0.0})
        return values
```

Given this, we can create a GridWorld MDP, and solve using value iteration. The code below prints the value function for value iteration after 1, 2, 3, 4, 5, 10, and 100 iterations:


```{code-cell} ipython3
mdp = NavigationMDP()
valueIteration = ValueIteration(mdp)

for iterations in [1, 2, 3, 4, 5, 10, 100]:
    print("After iteration " + str(iterations))
    values = valueIteration.valueIteration(iterations = iterations)
    print(mdp.valueFunctionToString(values) + "\n")
```

We can then extract  this using the `extractPolicy` method  from the MDP class:

```{code-cell} ipython3
print("Policy after 100 iterations")
values = valueIteration.valueIteration(iterations = 100)
print(mdp.policyToString(mdp.extractPolicy(values)))
```

### Strengths and Limitations

**Guarantees** Value iteration is guaranteed to converge to the optimal policy, given an infinite amount of time. In practice, for problems with a small-to-medium size state space, value iteration converges in a "reasonable" amount of time, returning a close-to-optimal value function, and quite often an optimal policy.

**The Curse of Dimensionality** Solving MDPs using value iteration is polynomial  in the size of the state space, but exponential in the number of variables if we use a factored representation such as a PDDL-like language to represent our problem. If there are $N$ number of variables, each a Boolean, there are $2^N$ number of states. Value iteration requires us to keep a vector of size $|2^N|$.

**Question:** Can we do better?

**Answer:** Yes! Using function approximation, which we will see later.