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
# Q-function approximation

## Learning Outcomes

1.  Manually apply linear Q-function approximation to solve small-scall MDP problems given some known features
    
2.  Select suitable features and design & implement Q-function approximation for model-free reinforcement learning techniques to solve medium-scale MDP problems automatically
    
3.  Argue the strengths and weaknesses of function approximation approaches
    
4.  Compare and contrast linear Q-learning with deep Q-learning


## Overview
Using a Q-table has two main limitations:

1. It requires that we visit every reachable state many times and apply every action many times to get a good estimate of $Q(s,a)$. Thus, if we never visit a state $s$, we have no estimate of $Q(s,a)$, even if we have visited states that are very similar to $s$.

2. It requires us to maintain a table of size $|A| \times |S|$, which is prohibitively large for any non-trivial problem.


To get around these we will look at how to use machine learning to approximate Q-functions. In particular, we will look at *linear function approximation* and approximation using *deep learning* (deep Q-learning). Instead of calculating an exact Q-function, we approximate it using simple methods that both eliminate the need for a large Q-table (therefore the methods scale better), and also allowing use to provide reasonable estimates of $Q(s,a)$ *even if we have not applied action $a$ in state $s$ previously*. 

:::{admonition} Example --- Freeway
Conside the game *Freeway*, in which a kangaroo needs to cross several lanes on a freeway without being run over by a car. A screenshot of the game is shown below:

![image](./figs/freeway_screenshot.png)

Let us assume that there are 12 rows and about 40 columns. This grossly underestimates the actual number of rows and columns because the cars move a few pixels at a time, not in columns. This means there are 480 different positions that a kangaroo can be in, and there are two kangaroos. We also need to record whether there is a car in each location. 

This leads to:

$$480^2 \times  2^{480} \approx 3 \times 10^{147} \text{ states}$$ 

There are four actions: left, right, up, down.

A Q-table would need to store $12\times 10^{147}$ entries. This is a huge Q-table for what is a trivial example compared to many other problems.
:::

## Linear Q-learning (Linear Function Approximation) 

The key idea is to *approximate* the Q-function using a linear combination of *features* and their weights. Instead of recording everything in detail, we think about what is most important to know, and model that.

What are some features that are relevant to the Freeway example?

The overall process is:

1.  For the states, consider what are the features that determine its representation.

2.  During learning, perform updates based on the *weights* of
    *features* instead of states.

3.  Estimate $Q(s,a)$ by summing the features and their weights.


:::{admonition} Example --- Features for *Freeway*

Instead of recording the position of both kangaroos and whether there is a car in every position, we just record the following features:

- the number of rows each kangaroo is away from the other side of the road in (two features -- one for each kangaroo); and
- how far away the *closest* car is in the row above and below each kangaroo (four features --- two for each kangaroo).

This requires just six features. 

:::

### Linear Q-function Representation 
In linear Q-learning, we store features and weights, not states. What we need to learn is how important each feature is (its *weight*) for each action.

To represent this, we have two vectors:

1.  A *feature vector*, $f(s,a)$, which is a vector of $n \cdot |A|$
    different functions, where $n$ is the number of state features and
    $|A|$ the number of actions. Each function extracts the value of a
    feature for state-action pair $(s,a)$. We say $f_i(s,a)$ extracts
    the $i$th feature from the state-action pair $(s,a)$:

    $$f(s,a) = \begin{pmatrix} 
        f_1(s,a) \\
        f_2(s,a) \\
        \ldots\\
        f_{n \times |A|}(s,a) \\
        \end{pmatrix}$$

    In the Freeway example, we have a vector with six state features
    times four actions. The function $f_1(s,Up)$ returns value of the feature that represents the distance kangaroo 1 is away from the goal. The function $f_{3}(s, Up)$
     returns the distance to the nearest car in the row above the first kangaroo. 

2.  A *weight* vector $w$ of size $n \times |A|$: one weight for each
    feature-action pair. $w^a_i$ defines the weight of a feature $i$ for
    action $a$.

### Defining State-Action Features 

Often it is easier to just define features for states, rather than state-action pairs. The features are just a vector of $n$ functions of the form $f_i(s)$.

However, for most applications, the weight of a feature is related to the action. The weight of being one step away from the end in Freeway is different if we go Up to if we go Right.

It is straightforward to construct $n \times |A|$ state-pair features from just $n$ state features:

$$
f_{i,k}(s,a) = \Bigg \{
\begin{array}{ll}
 f_i(s) & \text{if } a=a_k\\
 0      & \text{otherwise}
 ~~~ 1 \leq i \leq n, 1 \leq k \leq |A|
 \end{array}
$$

This effectively results in $|A|$ different weight vectors:

$$
 f(s,a_1) = \begin{pmatrix} 
f_{1,a_1}(s,a) \\
f_{2,a_1}(s,a) \\
0\\
0\\
0\\
0\\
\ldots
\end{pmatrix}~~
f(s,a_2) = \begin{pmatrix} 
0\\
0\\
f_{1,a_2}(s,a) \\
f_{2,a_2}(s,a) \\
0\\
0\\
\ldots
\end{pmatrix}~~
f(s,a_3) = \begin{pmatrix} 
0\\
0\\
0\\
0\\
f_{1,a_3}(s,a) \\
f_{2,a_3}(s,a) \\
\ldots
\end{pmatrix}~~\ldots
$$

### Linear Q-function Computation 
Give a feature vector $f$ and a weight vector $w$, the Q-value of a state is a simple linear combination of features and weights:

$$
\begin{array}{lll}
  Q(s,a) & = & f_1(s,a) \cdot w^a_1 + f_2(s,a)\cdot w^a_2 + \ldots  + f_{n}(s,a) \cdot w^a_n\\
         & = & \sum_{i=0}^{n} f_i(s,a) w^a_i
\end{array}
$$

In practice, we also multiple the feature vector for weights $w^b_n$ for all actions $b \neq a$, but as the feature values will be 0, we know that it does not influence the result.

:::{admonition} Example --- Approximate Q-function computation for Freeway

For the Freeway example, we would assume that moving up would give a better score than moving down, all else equal (that is, if the closest car in the next row up is the same distance away than the closest in the next row down). So, for state $s$ where the kangaroo is in row 1:

$$
\begin{array}{lll}
  Q(s,Up)   & = &  f_1(s,Up)\cdot 0.31  + \ldots + f_{6}(s,Up) \cdot 0.04
\end{array}
$$
:::

### Linear Q-function Update 

To use approximate Q-functions in reinforcement learning, there are two steps we need to change from the standard algorithsm: (1) initialisation; and (2) update.

For initialisation, initialise all weights to 0. Alternatively, you can try Q-function initialisation and assign weights that you think will be `good' weights.

For update, we now need to update the weights instead of the actions. For Q-learning, the update rule is now:

$$w^a_i \leftarrow w^a_i + \alpha [r + \gamma max_a' Q(s',a') - Q(s,a)]\ f_i(s,a)$$

For SARSA:

$$w^a_i \leftarrow w^a_i + \alpha [r + \gamma Q(s',a') - Q(s,a)]\ f_i(s,a)$$

Note: we need to update for each feature $i$ for the last executed action $a$.

As this is linear, it is therefore convex, so the weights will converge.

```{admonition} Note --- Q-value propagation
Note that this has the effect of updating Q-values to states that have never been visited! 

In Freeway, for example, if we receive our first reward by crossing the road (going Up from the final row), this will update the weight all features for Up, and now we have a Q-value for going Up from *any* position on the final row.
```

```{admonition} Example --- Q-value update for Freeway
Assume that all weights are 0, therefore, $Q(s,a) = 0$ for every state and action. Now, we receive the reward of 10 for getting to the other side of the road. If feature 6 is has the value $\frac{r}{D}$, where $r$ is the current row and $D$ is the distance to the other side, then
we have:

$$
\begin{array}{lll}
  w^a_i & \leftarrow & w^a_i + \alpha[r + \gamma \max_a Q(s',a') - Q(s,a)] f_i(s,a)\\
  w^{Up}_{6} & \leftarrow & 0 + 0.5[10 + 0.9 \times 0] \frac{10}{10}\\
              & = &5
\end{array}
$$

From this, we now can get an estimate of $Q(s,Up)$ from any state because we have some weights in our linear function. Those that are closer to the other size of the road will get a higher Q-value than those further away (all other things being equal).
```

### Challenges and tips

The key challenge in linear function approximation for Q-learning is the feature engineering: selecting features that are meaningful and helpful in learning a good Q function. As well as estimating the Q-values of each action in a state, it also has to estimate the value of future states. As with any machine learning problem, feature engineering requires some experimentation and a careful combination of art and science.

**Tip:** Note that to be effective, our feature values can be *normalised* using e.g. min-max normalisation or mean normalisation. 

## Deep Q-learning

The latest hype in reinforcement learning is all about the use of deep neural networks to approximate value and Q-functions. 

### Deep Q-function representation

In deep Q-learning, Q-functions are represented using deep neural networks. Instead of selecting features and training weights, we learn the parameters $\theta$ to a neural network. The Q-function is $Q(s,a; \theta)$, so takes the parameters as an argument.

This has the advantage (over linear Q-function approximation) that feature engineering is not required, the 'features' will be learnt as part of the hidden layers of the neural network. 

A further advantage is that states can be non-structured (or less structured), rather than using a factored state representation. This means that states can be images, videos (sequences of images), or unstructured text.

### Deep Q-function update

The update rule for deep Q-learning looks similar to that of updating a linear Q-function.

The deep Q-learning  TD update for Q-learning is just:

$$\theta \leftarrow \theta + \alpha[r + \gamma \max_{a'} Q(s',a'; \theta) - Q(s,a ;\theta)]
\nabla_{\theta} Q(s,a; \theta)$$

where $\nabla_{\theta} Q(s,a; \theta)$ is the *gradient* of the Q-function. In these notes, we will not cover how to calculate the gradient of the Q-function: there are many excellent text books that cover gradients.

For SARSA, the TD update is:

$$\theta \leftarrow \theta + \alpha[r + \gamma Q(s',a'; \theta) - Q(s,a ;\theta)]
\nabla_{\theta} Q(s,a; \theta)$$


### Advantages and disadvantages

**Advantages** of deep Q-function approximation  (compared to linear Q-function approximation):

- We do not need to select features -- the 'features' will be learnt as part of the hidden layers of the neural network. 
- The state $s$ can be less structured, such as images or sequences of images (video).

Disadvantages:

- There are no convergence guarantees.
- Deep neural networks are data hungry because they need to learn features as well as "the Q-function", so compared to a linear approximation with good features, learning good Q-functions can be  difficult. Large amounts of computation are often required.

Despite this, deep Q-learning  works remarkably well in some areas, especially for tasks that require vision (see the robotic arm grasping unknown objects).

## Strengths and Limitations of Q-function Approximation

Approximating Q-functions using machine learning techniques such as linear functions or deep learning  has advantages and disadvantages.

**Advantages:**

-   Memory: More efficient representation compared with Q-tables because we only store weights/parameters for the Q-function, rather than the the $|A| \times |S|$ entries for a Q-table.

-   Q-value propagation: we do not need to apply action $a$ in state $s$ to get a value for $Q(s,a)$ because the Q-function generalises.

**Disadvantages:**

-   The Q-function is now only an approximation of the real Q-function: states that share feature values will have the same Q-value according to the Q-function, but the actual Q-value according to the (unknown) optimal Q-function may be different.

## Implementation

In this section, we present an implementation of linear Q-function approximation for SARSA and run it on a modified GridWorld problem. For simplicity, we will first demonstrate this on the GridWorld with just one goal state: the one in the top right that returns +1. The -1 will just become a normal cell. We will see later that the original GridWorld problem is not easy to define features for given a linear approximation.

```{code-cell} ipython3
---
tags: [remove-cell]
---
import sys
sys.path.append('/mnt/c/Users/tmiller/OneDrive - The University of Melbourne/Documents/subjects/COMP90054/rl-notes/code')
```

The first thing we need to do is define some features for the task. As discussed above, feature engineering is not always straightforward. However, for the GridWorld task, it is reasonably clear that the distance from the goal cell is important. As such, we define three features here: 

1. The distance from the goal on the X-axis.
2. The distance from the goal on the Y-axis.
3. The total distance from the goal as a Manhattan distance.

As noted above, normalising features is important to ensure that they are in the same magnitude. So, given the current position $(x,y)$ as the state, we extract the values of the state features as follows:

1. $1 - ((x(g) - x(s)) / width)$
2. $1 - ((y(g) - y(s)) / height)$
3. $1 - ((x(g) - x(s) + y(g) - y(s) / (x(g) + y(g))$

where $x(s)$ and $y(s)$ return the x and y coordinates of the agent respectively, and $g$ is the goal state.

These expressions normalise the feature values to the range $[0,1]$. Subtracting from 1 means that states that are closer to the goal have a higher value, which is intuitively easier to consider, but is technically not necessary.

Then, to extract state-action features, we need to define these as $f(s,a)$ is defined from $f(s)$ above.

We can implement these in a feature extractor class:

```{code-cell} ipython3
class FeatureExtractor():
    def extractFeatures(self, state, action): abstract
    
class GridWorldFeatureExtractor(FeatureExtractor):
    def __init__(self, mdp):
        self.mdp = mdp

    def initialiseWeights(self):
        weights = []
        for action in self.mdp.getActions():
            weights += [0.0, 0.0, 0.0]
        return weights
        
    def extractFeatures(self, state, action):
        goal = (self.mdp.width, self.mdp.height)
        x = 0
        y = 1
        featureValues = []
        for a in self.mdp.getActions():
            if a == action and state != GridWorld.TERMINAL:
                featureValues += [1 - ((goal[x] - state[x]) / goal[x])]
                featureValues += [1 - ((goal[y] - state[y]) / goal[y])]
                featureValues += [1 - ((goal[x] - state[x] + goal[y] - state[y]) / (goal[x] + goal[y]))]
            else:
                featureValues += [0.0, 0.0, 0.0]
        return featureValues
```

Now, we implement our learning using this. Recall that the superclass ``ModelFreeReinforcementLearner`` is used for all of our model-free techniques; unhide to see the code for this below.

```{code-cell} ipython3
---
tags: [hide-cell]
---
class ModelFreeReinforcementLearner():

    # how many episodes to take an average for determining convergence
    length = 30

    def __init__(self, mdp, bandit, alpha = 0.1, convergenceEpsilon = float('-inf'), initQValues = None):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.convergenceEpsilon = convergenceEpsilon
        self.initQValues = initQValues

        self.latestRewards = []
        self.previousAverage = 0.0

    def execute(self, episodes = 2000): abstract

    def getMaxQ(self, qValues, state):
        argmaxQ = None
        maxQ = float('-inf')
        for action in self.mdp.getActions(state):
            value = qValues[(state, action)]
            if maxQ < value:
                argMaxQ = action
                maxQ = value
        return (argmaxQ, maxQ)

    def initialiseQFunction(self):
        if self.initQValues == None:
            qValues = dict()
            for state in self.mdp.getStates():
                for action in self.mdp.getActions():
                    qValues.update({(state, action): 0.0})
            return qValues
        else:
            return self.initQValues

    '''
        Return the Q-values only for this state
    '''
    def getQValues(self, state, qValues):
        return {k[1]:v for (k,v) in qValues.items() if k[0] == state}   
```

Our SARSA implementation with a linear Q-function is similar to our first SARSA algorithm, except that we update and retrieve Q-values from the linear function instead of a Q-table:

```{code-cell} ipython3
class LinearSARSA(ModelFreeReinforcementLearner):
    def __init__(self, mdp, bandit, featureExtractor, alpha = 0.1, initQValues = None, weights = None):
        super().__init__(mdp, bandit, alpha = alpha, initQValues = initQValues)
        self.featureExtractor = featureExtractor
        self.weights = weights
        
    def execute(self, episodes = 100):
        self.initialiseQFunction()

        for i in range(episodes):
            state = self.mdp.getInitialState()
            actions = self.mdp.getActions(state)
            action = self.bandit.select(actions, self.getQValues(actions, state))

            while not self.mdp.isTerminal(state):
                (nextState, reward) = self.mdp.execute(state, action)
                actions = self.mdp.getActions(nextState)
                nextAction = self.bandit.select(actions, self.getQValues(actions, nextState))
                newValue = self.update(state, action, nextState, nextAction, reward)
                state = nextState
                action = nextAction

    '''
        Return the Q-value for a state-action pair
    '''
    def getQValue(self, state, action):
        qValue = 0.0
        featureValues = self.featureExtractor.extractFeatures(state, action)
        for i in range(len(featureValues)):
            qValue += featureValues[i] * self.weights[i]
        return qValue
    
    def update(self, state, action, nextState, nextAction, reward):
        qValue = self.getQValue(state, action)
        qValueNext = self.getQValue(nextState, nextAction)
        delta = self.alpha * (reward + self.mdp.discountFactor * qValueNext - qValue)

        # update the weights
        featureValues = self.featureExtractor.extractFeatures(state, action)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (delta * featureValues[i])
        
    def initialiseQFunction(self):
        if self.weights == None:
            self.weights = self.featureExtractor.initialiseWeights()

    '''
        Return the Q-values only for this state
    '''
    def getQValues(self, actions, state):
        qValues = dict()
        for action in actions:
            qValues[action] = self.getQValue(state, action)
        return qValues

    def getQTable(self):
        qValues = dict()
        for state in self.mdp.getStates():
            for action in self.mdp.getActions(state):
                qValues[(state, action)] = self.getQValue(state, action)
        return qValues
```

Let's see how this goes on Gridworld. First, we extract the Q-values:

```{code-cell} ipython3
from gridworld import *
from multi_armed_bandits import *
mdp = GridWorld(discountFactor = 0.9, noise=0.1, goals=[((3,2),1)])
featureExtractor = GridWorldFeatureExtractor(mdp)
linearSarsa = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor)
linearSarsa.execute(episodes = 100)
qFunction = linearSarsa.getQTable()
print(mdp.qFunctionToString(qFunction))
```

We can see that this gives quite good Q-values, but these are not optimal (compare them to the Q-values for Q-table-based learning).

Despite this, it still extracts a good policy, albeit not one that is necessarily optimal:

```{code-cell} ipython3
policy = mdp.extractPolicyFromQFunction(qFunction)
print(mdp.policyToString(policy))
```

The choice of features is key to solving the problem. Above, we have define features that assume there is just the goal in the top-right corner. However, if we return to the original GridWorld problem that has another terminating state with reward -1, our features no longer work particularly well:

```{code-cell} ipython3
from gridworld import *
from multi_armed_bandits import *
mdp = GridWorld()
featureExtractor = GridWorldFeatureExtractor(mdp)
linearSarsa = LinearSARSA(mdp, EpsilonGreedy(), featureExtractor)
linearSarsa.execute(episodes = 100)
qFunction = linearSarsa.getQTable()
print(mdp.qFunctionToString(qFunction))
```

This is because our linear approximation learns one weight for going right, left, up, and down. Going right at the state $(2,2)$ is clearly good, so the weight will be learnt as positive, but every time an update is performed after the agent tranisitions from $(2,2)$ to the goal state $(3,2)$, the weight updates the value of $Q(s, Right)$ for all states $s$, including $(2,1)$.

We could solve this by encoding specific features that learn that we are e.g. in state $(2,1)$, but the more of these features we engineer, the more domain knowledge we are encoding into our solution. It is fine to encode domain knowledge, but we want to avoid a situation where we have to encode enough knowledge that we may as well encode the entire solution by hand..

## Summary

1.  We can scale reinforcement learning by approximating Q-functions, rather than storing complete Q-tables.

2.  Using simple linear methods in which we select features and learn weights are effective and guarantee convergence.

3.  Deep Q-learning offers alternatives in which we do not need to select features, but requires more training data (more episodes) and has no convergence guarantees.


## Further Reading

- Chapter 9 (Approximate Solution Methods) of *Introduction to Reinforcement Learning* \[*Sutton and Barto*\]: <https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html>

- Deep Q-learning for Atari. This uses Convolutional Neural Networks (NN) to estimate $\mathcal{Q}(s,a)$. The input for the NN is the state, and the output is the estimated reward for each action. There are two papers worth reading on this:

  - [Human-level control through deep reinforcement learning](http://www.davidqiu.com:8888/research/nature14236.pdf). Mnih, V., et al. Nature 529 (2015).
  - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf). Mnih, V., et al. arXiV: preprint arXiv:1312.5602 (2013).

-   Before AlphaGo there was TD-gammon, which was the first paper to
    combine reinforcement learning and neural networks:
    [TD-Gammon, A Self-Teaching Backgammon Program, Achieves Master-Level Play](http://www.aaai.org/Papers/Symposia/Fall/1993/FS-93-02/FS93-02-003.pdf), : AAAI Technical Report FS-93-02 (1993).
