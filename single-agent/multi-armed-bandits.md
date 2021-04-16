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

## Multi-armed bandits

### Learning outcomes

The learning outcomes of this chapter are:

1. Select and apply multi-armed bandit algorithms

### Overview

*Multi-armed bandit* techniques are not techniques for solving MDPs, but it turns out that they are used throughout a lot of reinforcement learning techniques that do solve for MDPs.

The problem of multi-armed bandits can be illustrated as follows:

> Imagine that you have $N$ number of slot machines (or poker machines in Australia), which are sometimes called *one-armed bandits*. Over  time, each bandit pays a random reward from an unknown probability distribution. Some bandits pay higher rewards than others. The goal is to maximize the sum of the rewards of a sequence of lever pulls of the  machine.

The question is: over an infinite period of time, without knowing the probability distribution beforehand, how should we select the arms. Multi-armed bandit techniques aim to solve this problem. 

### The multi-armed bandit problem

:::{admonition} Definition

A **multi-armed bandit** (also known as an **$N$-armed bandit**) is defined by a set of *random variables* $X_{i,k}$ where:

-   $1 \leq i \leq N$, such that $i$ is the *arm* of the bandit; and

-   $k$ the index of the *play* of arm $i$;

Successive plays $X_{i,1}, X_{j,2}, X_{k,3}\ldots$ are assumed to be independently distributed, but we do not know the probability distributions of the random variables.

The idea is that a gambler iteratively plays rounds, observing the reward from the arm after each round, and can adjust their strategy each time. The aim is to maximise the sum of the rewards collected over all  rounds.

:::

Given that we do not know the distributions, a simple strategy is simply to select the arm given a uniform distribution; that is, select each arm with the same probability. This is just uniform sampling.

Then, the Q-value for an action $a$ can be estimated using the following formula:

$$Q(a) = \frac{1}{N(a)} \sum_{i=1}^{t} {\mathbb I}_{i}(a) r_i$$

where $t$ is the number of rounds so far, $N(a)$ is the number of times $a$ selected in previous rounds, $r_i$ is the *reward* obtained in the $i$-th round, and  $\mathbb{I}_{i}(a)$ is $1$ if $a$ was selected on the $i$-th round, and is $0$ otherwise.

The idea here is that for a multi-armed bandit problem, we explore the options uniformly for some time, and then once we are confident we have enough samples (when the changes to the values of $Q(a)$ start to stabilise), we start selecting $\max_a Q(a)$. This is a strategy known as *$\epsilon$-first* strategy, where the parameter $\epsilon$ (epsilon), determines how many rounds to select random actions before moving to the greedy action.

**But what is the issue?** Time is wasted equally in all actions using the uniform distribution. Why not focus also on the *most promising actions* given the rewards we have received so far.

### Exploration vs. Exploitation

What we want is to play only the good actions; so just keep playing the actions that have given us the best reward so far. However, our selection is randomised, so what if we just haven't sampled the best action enough times? Thus, we want strategies that *exploit* what we think are the best actions so far, but still *explore* other actions.

But how much should we exploit and how much should we explore? This is known as the *exploration vs. exploitation dilemma*. It is driven by the *The Fear of Missing Out* (FOMO). FOMO drives us to search for strategies that *minimise regret*.

:::{definition}(Pseudo)--Regret 
Pseudo-regret is defined formally as:

 $$\mathcal {R_{N,b} }  =  \max_a Q(a) N(s) - \mathbb{E} [ \sum_{i}^{t} Q(b) \mathbb{I}_{i}(b) ]$$

where $t$ is the number of rounds, $\mathbb{I}_{i}(a)$ is $1$ if $a$ was selected on the $i$-th round and $0$ otherwise, and ${\mathbb E}[ \sum_{i}^{t} Q(b) \mathbb{I}_{i}(b)] > 0$ for every $b$.
:::

Informally: If I take action $b$, my regret is the *best possible expected reward* minus the *expected reward of playing $b$*. If I take action $a$ (the best action), my regret is 0. So, regret is the *expected loss* from not taking the best action.

A *zero-regret* strategy is a strategy whose average regret each round approaches zero as the number of rounds approached infinity. So, this means that a zero-regret strategy will converge to an optimal strategy given enough rounds.

### Solutions for minimising regret

There are several basic strategies for minimising regret. In each of these techniques, we record the average return of each arm over time. For consistency with the rest of these notes, we will call these Q-values, and will use $Q(a)$ to represent the Q value of arm $a$.

### Epsilon-greedy strategy

The $\epsilon$-greedy strategy  is a simple and effective way of balancing exploration and exploitation. In this algorithm, the parameter $\epsilon \in [0,1]$ (pronounced "epsilon") controls how much we explore and how much we exploit. 

Each time we need to choose an action, we do the following:

- With probability $\epsilon$ we choose the arm with the maximum Q value: $\textrm{argmax}_a Q(a)$. If there is a tie between multiple actions with the larget Q-value, break the tie.
- With probability $1-\epsilon$ we choose a random arm with uniform probability.

The best value for $\epsilon$ depends on the particular problem, but typically, values around 0.05-0.1 work well as they exploit what they have learnt.

#### Implementation

```{code-cell} ipython3
---
tags: [remove-cell]
---
import sys
sys.path.append('/mnt/c/Users/tmiller/OneDrive - The University of Melbourne/Documents/subjects/COMP90054/rl-notes/code')
```
Each multi-armed bandit strategy we discuss inherits from a class ```MultiArmedBandit```, which contains an action ```select``` that takes the list of available actions and their Q-values:

```{code-cell} 
import random
import math

class MultiArmedBandit():

    '''
        Select an action given Q-values for each action.
    '''
    def select(self, actions, qValues): abstract
```

```{code-cell} ipython3
---
tags: [remove-cell]
---
import random
import math

class MultiArmedBandit():

    '''
        Select an action given Q-values for each action.
    '''
    def select(self, actions, qValues): abstract
    
    '''
        Reset a multi-armed bandit to its initial configuration.
    '''
    def reset(self):
        self.__init__()

    '''
        Run a bandit algorithm for a number of episodes, with each
        episode being a set length.
    '''
    def runBandit(self, episodes = 50, episodeLength = 1000, drift = True):

        #the actions available
        actions = [0, 1, 2, 3, 4]

        rewards = []
        for episode in range(0, episodes):
            self.reset()

            # The probability of receiving a payoff of 1 for each action
            probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]
        
            qValues = dict()
            N = dict()
            for action in actions:
                qValues[action] = 0.0
                N[action] = 0

            episodeRewards = []
            for step in range(0, episodeLength):

                # Halfway through the episode, change the probabilities
                if drift and step == episodeLength / 2:
                    probabilities = [0.5, 0.2, 0.0, 0.3, 0.3]
                
                #select an action
                action = self.select(actions, qValues)

                r = random.random()
                reward = 0
                if r < probabilities[action]:
                    reward = 5

                episodeRewards += [reward]

                N[action] = N[action] + 1

                qValues[action] = qValues[action] - (qValues[action] / N[action])
                qValues[action] = qValues[action] + reward / N[action]

            rewards += [episodeRewards]

        return rewards
```

The implementation for epsilon greedy then using ```random()``` to select a random number between 0 and 1. If that number is less than epsilon, an action is randomly selected. If r is greater than or equal to epsilon, it finds the actions with the maximum Q value, breaking ties randomly:

```{code-cell} ipython3
class EpsilonGreedy(MultiArmedBandit):

    def __init__(self, epsilon = 0.1):
        self.epsilon = epsilon

    def reset(self):
        None

    def select(self, actions, qValues):
        r = random.random()
        # select a random action with epsilon probability
        if r < self.epsilon:
            return random.choice(actions)
        else:
            maxActions = []
            maxValue = float('-inf')
            for action in actions:
                value = qValues[action]
                if value > maxValue:
                    maxActions = [action]
                    maxValue = value
                elif value == maxValue:
                    maxActions += [action]
                    
            # if there are multiple actions with the highest value
            # choose one randomly
            return random.choice(maxActions)
```

The ```reset``` method is used to run the simulations for the plots in this section.

The following plot shows the average reward of each step in an episode over 500 episodes, for a set of different values of epsilon. Each episode is 1000 steps long. In each step of each episode, the bandit chooses one of five actions, in which each actions has a probability of giving a payoff. The action probabilities are different for each action. If an action has 0.2 probability of a reward, then with probability, the action returns the reward, and with probability 0.8 it returns 0 reward:

```{code-cell} ipython3
from plot import Plot

def plotEpsilonGreedy(drift = False):
    epsilon000 = EpsilonGreedy(epsilon = 0.00).runBandit(drift = drift)
    epsilon005 = EpsilonGreedy(epsilon = 0.05).runBandit(drift = drift)
    epsilon01 = EpsilonGreedy(epsilon = 0.1).runBandit(drift = drift)
    epsilon02 = EpsilonGreedy(epsilon = 0.2).runBandit(drift = drift)
    epsilon04 = EpsilonGreedy(epsilon = 0.4).runBandit(drift = drift)
    epsilon08 = EpsilonGreedy(epsilon = 0.8).runBandit(drift = drift)
    epsilon10 = EpsilonGreedy(epsilon = 1.0).runBandit(drift = drift)

    Plot.plotRewards(["epsilon = 0.0", "epsilon = 0.05", "epsilon = 0.1", "epsilon = 0.2", "epsilon = 0.4", "epsilon = 0.8", "epsilon = 1.0"],
                     [epsilon000, epsilon005, epsilon01, epsilon02, epsilon04, epsilon08, epsilon10])
                     
plotEpsilonGreedy()
```

As one can see, lower values of epsilon tend to have a lower reward over time, except that we need some non-zero value of epsilon. The actual choice of parameter is entirely dependent on the particular application: there is no magic number. However, as in this particular case, an epsilon between 0.05-0.1 is usually a reasonable choice.

### Epsilon-decreasing

This follows a similar idea to epsilon greedy, however, it recognises that initially, we have very little feedback so exploiting is not a good strategy to being with: we need to explore first. Then, it recognises that as we gather more data, we should exploit more.

It does this by taking the basic epsilon greedy strategy and introducing another parameter $\alpha$ (pronounced "alpha"), which is used to decrease $\epsilon$ over time. For this reason, $\alpha$ is called the *decay$.

The selection mechanism is the same as epsilon greedy, but then after each selection, we set  $\epsilon := \epsilon \times \alpha$. We start initially with a higher value of $\epsilon$ to explore, and it will slowly decay to a low number such that we explore less and less as we gather more feedback.

#### Implementation

The implementation for the epsilon decreasing strategy uses the epsilon greedy strategy, just changing the epsilon value each step:

```{code-cell} ipython3
class EpsilonDecreasing(MultiArmedBandit):

    def __init__(self, epsilon = 0.2, alpha = 0.999):
        self.epsilonGreedyBandit = EpsilonGreedy(epsilon)
        self.initialEpsilon = epsilon
        self.alpha = alpha

    def reset(self):
        self.epsilonGreedyBandit = EpsilonGreedy(self.initialEpsilon)
    
    def select(self, actions, qValues):
        
        result = self.epsilonGreedyBandit.select(actions, qValues)
        self.epsilonGreedyBandit.epsilon *= self.alpha
        return result
```

The following plot shows the average reward over 500 episodes, each 1000 steps long, varying the value of $\alpha$:

```{code-cell} ipython3
def plotEpsilonDecreasing(drift = False):
    alpha09 = EpsilonDecreasing(alpha = 0.9).runBandit(drift = drift)
    alpha099 = EpsilonDecreasing(alpha = 0.99).runBandit(drift = drift)
    alpha0999 = EpsilonDecreasing(alpha = 0.999).runBandit(drift = drift)
    alpha1 = EpsilonDecreasing(alpha = 1.0).runBandit(drift = drift)
    

    Plot.plotRewards(["alpha = 0.9", "alpha = 0.99", "alpha= 0.999", "alpha = 1.0"],
                     [alpha09, alpha099, alpha0999, alpha1])
                     
plotEpsilonDecreasing()
```

This indicates that for this particular simulator, a value of 0.99 for alpha has a better average return  than lower values. This is because a lower value, such as 0.9, will very quickly result in epsilon approaching zero. However, the choice of alpha depends both on the application, and also the expected length of each episode: for longer episodes, decreasing slower would be more beneficial so we do not stop exploring too early.

### Softmax

Softmax  is *probability matching strategy*, which means that the probability of each action being chosen is dependent on its Q-value so far. Formally, softmax chooses an action because on the *Boltzman* distribution for that action:

$$\frac{e^{Q(a)/\tau}}{\sum_{b=1}^{N} e^{Q(b)/\tau}}$$ 

where $N$ is the number of arms, and $\tau$ (pronounced "tau") is the *temperature*, a positive number that dictates how much of an influence the past data has on the decision. A higher value of $\tau$ would mean that the probability of selecting each action is close to each other, while a lower value of $\tau$ would imply that the probabilities are closer to their Q values. When $\tau=1$, the probabilities are just $e^{Q(a)}$.

As with epsilon decreasing, we can add a decay parameter $\alpha$ that allows the value of $\tau$ to decay until it reaches 1. This encourages exploration in earlier phases, and exploration less as we gather more feedback.

#### Implementation

The following implementation of the softmax strategy uses ```random()``` to generate a random number between 0 and 1, and divides this space 0-1 among the set of actions because on the value of $e^{Q(a)/\tau}$:

```{code-cell} ipython3
class Softmax(MultiArmedBandit):
    
    def __init__(self, tau = 1.0):
        self.tau = tau

    def reset(self):
        None
    
    def select(self, actions, qValues):

        # calculate the denominator for the softmax strategy
        sum = 0.0
        for action in actions:
            sum += math.exp(qValues[action] / self.tau)

        r = random.random()
        cumulativeProbability = 0.0
        result = None
        for action in actions:
            probability = math.exp(qValues[action] / self.tau) / sum
            if r >= cumulativeProbability and r <= cumulativeProbability + probability:
                result = action
            cumulativeProbability += probability

        return result
```

```{code-cell} ipython3
def plotSoftmax(drift = False):
    tau10 = Softmax(tau = 1.0).runBandit(drift = drift)
    tau11 = Softmax(tau = 1.1).runBandit(drift = drift)
    tau15 = Softmax(tau = 1.5).runBandit(drift = drift)
    tau20 = Softmax(tau = 2.0).runBandit(drift = drift)

    Plot.plotRewards(["tau = 1.0", "tau = 1.1", "tau = 1.5", "tau = 2.0"],
                     [tau10, tau11, tau15, tau20])
                    
plotSoftmax()
```

### Upper Confidence Bounds (UCB1)

A highly effective multi-armed bandit strategy is the *Upper Confidence Bounds* (UCB1) strategy.

Using the UCB1 strategy, we select the next action  using the following:

$$\textrm{argmax}_{a}\left(Q(a)   +   \sqrt{\frac{2 \ln t}{N(a)}}\right)$$

where $t$ is the number of rounds so far, and $N(a)$ is the number of times times $a$ has been chosen in all previous rounds. The term inside the square root is undefined if $N(a) = 0$. The avoid this, the typical strategy is to spend the first $N$ rounds to select each of the $N$ bandits once.

The left--hand side encourages exploitation: the Q-value is high for actions that have had a high reward.

The right--hand side encourages exploration: it is high for actions that have been explored less -- that is, when $N(a)$ relative to other actions.

Interesting, the UCB formula is not a weighted formula -- that is, there is no parameter giving weight to the $Q(a)$ or the square root expressions to balance exploration vs. exploitation. So how does it work? We will not get into all the details, but instead just give some intuition.

We want to learn the Q-function, which gives us the average return on each action $a$, such that it approximates the real (unknown) Q-function, which we will call $Q^*$. At each round, we select the action $a$ that maximises the expression inside the brackets. If arm $a$ is optimal, then we want the following to hold for all actions $b \neq a$:

$$Q(b) + sqrt{\frac{2 \ln t}{N(a)}} \leq Q^*(a)$$

If this holds, we have some confidence that $Q(a)$ is optimal. If $N(b)$ is low for some actions, we do not have this confidence. 

If by chance the above expression does NOT hold for the optimal action $a$, then $a$ is disregarded, but should not be. We want this to occur only with probability $\frac{1}{N}$ to maximise pseudo-regret. This leads us to $\ln t$ in the expression. We will not go into the technicalities of why $\ln t$ in these notes.

#### Implementation

```{code-cell} ipython3 
class UpperConfidenceBounds(MultiArmedBandit):

    def __init__(self):
        self.total = 0
        self.N = dict() #number of times each action has been chosen

    def select(self, actions, qValues):

        # First execute each action one time
        for action in actions:
            if not action in self.N.keys():
                self.N[action] = 1
                self.total += 1
                return action

        maxActions = []
        maxValue = float('-inf')
        for action in actions:
            N = self.N[action]
            value = qValues[action] + math.sqrt((2 * math.log(self.total)) / N)
            if value > maxValue:
                maxActions = [action]
                maxValue = value
            elif value == maxValue:
                maxActions += [action]
                    
        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(maxActions)
        self.N[result] = self.N[result] + 1
        self.total += 1
        return result
```

Because UCB does not have parameters, there is no exploration to be done, however, in the next section we compare UCB with the other three strategies.

### Comparison

```{code-cell} ipython3
def plotComparison(drift = False):
    epsilonGreedy = EpsilonGreedy(epsilon = 0.1).runBandit(drift = drift)
    epsilonDecreasing = EpsilonDecreasing(alpha = 0.99).runBandit(drift = drift)
    softmax = Softmax(tau = 1.0).runBandit(drift = drift)
    ucb = UpperConfidenceBounds().runBandit(drift = drift)

    Plot.plotRewards(["Epsilon Greedy (epsilon = 0.1)", "Epsilon Decreasing (alpha = 0.99)", "Softmax (tau = 1.0)", "UCB"],
                     [epsilonGreedy, epsilonDecreasing, softmax, ucb])


plotComparison()
```

From this, we may answer the question who softmax is considered a good strategy at all: it clearly does not perform well compared to the others.

```{code-cell} ipython3
plotComparison(drift=True)
```