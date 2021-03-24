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

-   $k$ the index of the *play* of arm $i$.

Successive plays $X_{i,1}, X_{j,2}, X_{k,3}\ldots$ are assumed to be independently distributed, but we do not know the probability distributions of the random variables.
:::

Given that we do not know the distributions, a simple strategy is simply to select the arm given a uniform distribution; that is, select each arm with the same probability. This is just uniform sampling.

Then, the Q-value for an action $a$ in a given state $s$ can be approximated using the following formula:

$$Q(s,a) = \frac{1}{N(s,a)} \sum_{t=1}^{N(s)} {\mathbb I}_{t}(s,a) r_t$$

where $N(s,a)$ is the number of times $a$ executed in $s$, $N(s)$ is the number of times $s$ is visited, $r_t$ is the *reward* obtained by the $t$-th simulation from $s$, and  $\mathbb{I}_{t}(s,a)$ is $1$ if $a$ was selected on the $t$-th simulation from $s$, and is $0$ otherwise

**But what is the issue?** Time is wasted equally in all actions using the uniform distribution. Why not focus also on the *most promising actions* given the rewards we have received so far.

### Exploration vs. Exploitation

What we want is to play only the good actions; so just keep playing the actions that have given us the best reward so far. However, our selection is randomised, so what if we just haven't sampled the best action enough times? Thus, we want strategies that *exploit* what we think are the best actions so far, but still *explore* other actions.

But how much should we exploit and how much should we explore? This is known as the *exploration vs. exploitation dilemma*. It is driven by the *The Fear of Missing Out* (FOMO).

FOMO drives us to search for policies $\pi$ that *minimise regret*.

:::{definition}(Pseudo)--Regret 
Pseudo-regret is defined formally as:

 $$\mathcal {R_{N(s),b} }  =  Q(\pi^*(s),s) N(s) - \mathbb{E} [ \sum_{t}^{N(s)} Q(b,s) \mathbb{I}_{t}(s,b) ]$$

where $Q(\pi^*(s),s)$ is the $Q$-value for the optimal policy $\pi^*(s)$ (which we do not know), $N(s)$ is the number of visits to state $s$, $\mathbb{I}_{i}(s,a)$ is $1$ if $a$ was selected on the $i$-th visit from $s$, and $0$ otherwise, and ${\mathbb E}[ \sum_{t}^{N(s)} Q(b,s) \mathbb{I}_{t}(s,b)] > 0$ for every $b$.
:::

Informally: If I play arm $b$, my regret is the *best possible expected reward* minus the *expected reward of playing $b$*. If I play arm $a$ (the best arm), my regret is 0. So, regret is the *expected loss* from not taking the best action.

### Solutions for minimising regret

There are several basic strategies for minimising regret. In each of these techniques, we record the average return of each arm over time. For consistency with the rest of these notes, we will call these Q-values, and will use $Q(a)$ to represent the Q value of arm $a$.

### Epsilon-greedy strategy

The $\epsilon$-greedy strategy  is a simple and effective way of balancing exploration and exploitation. In this algorithm, the parameter $\epsilon \in \[0,1\]$ (pronounced "epsilon") controls how much we explore and how much we exploit. 

Each time we need to choose an arm, we do the following:

- With probability $\epsilon$ we choose the arm with the maximum Q value: $\argmax_a Q(a)$
- With probability $1-\epsilon$ we choose a random arm with uniform probability.

The best value for $\epsilon$ depends on the particular problem, but typically, values around 0.05-0.1 work well as they exploit what they have learnt.

### Epsilon-decreasing

This follows a similar idea to epsilon greedy, however, it recognises that initially, we have very little feedback so exploiting is not a good strategy to being with: we need to explore first. Then, it recognises that as we gather more data, we should exploit more.

It does this by taking the basic epsilon greedy strategy and introducing another parameter $\alpha$ (pronounced "alpha"), which is used to decrease $\epsilon$ over time. For this reason, $\alpha$ is called the *decay$.

The selection mechanism is the same as epsilon greedy, but then after each selection, we set  $\epsilon := \epsilon \times \alpha$. We start initially with a higher value of $\epsilon$ to explore, and it will slowly decay to a low number such that we explore less and less as we gather more feedback.

### Softmax

Softmax  is *probability matching strategy*, which means that the probability of each action being chosen is dependent on its Q-value so far. Formally, softmax chooses an action because on the *Boltzman* distribution for that action:

$$\frac{e^{Q(a)/\tau}}{\sum_{b=1}^{n} e^{Q(b)/\tau}}$$ 

in which $\tau$ (pronounced "tau") is the *temperature*, a positive number that dictates how much of an influence the past data has on the decision. A higher value of $\tau$ woudl mean that the probability of selecting each action is close to each other, while a lower value of $\tau$ would imply that the probabilities are closer to their Q values. When $\tau=1$, the probabilities are just $e^(Q(a))$.

As with epsilon decreasing, we can add a decay parameter $\alpha$ that allows the value of $\tau$ to decay until it reaches 1. This encourages exploration in earlier phases, and exploration less as we gather more feedback.

### Upper Confidence Bounds (UCB1)

A highly effective multi-armed bandit strategy is the *Upper Confidence Bounds* (UCB1) strategy.

Using the UCB1 strategy, we select the next action  using the following:

$\textrm{argmax}_{a}(Q(a)   +   \sqrt{\frac{2 \ln N}{N(a)}})$

where $N$ is the number of times we have made a multi-armed bandit decision, and $N(a)$ is the number of times times $a$ has been chosen.

The left--hand side encourages exploitation: the Q-value is high for actions that have had a high reward.

The right--hand side encourages exploration: it is high for actions that have been explored less.